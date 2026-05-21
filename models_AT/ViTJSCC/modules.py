import numpy as np
import torch.nn as nn
import torch
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Attention, Block


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """
        x: B, H*W+1, C
        """
        x0 = x[:, 0::4, :]
        x1 = x[:, 1::4, :]
        x2 = x[:, 2::4, :]
        x3 = x[:, 3::4, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class PatchMerging4x(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.patch_merging1 = PatchMerging(dim, dim, norm_layer)
        self.patch_merging2 = PatchMerging(dim, out_dim, norm_layer)

    def forward(self, x):
        x = self.patch_merging1(x)
        x = self.patch_merging2(x)
        return x


class PatchReverseMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.proj = nn.Linear(dim, out_dim)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.norm(x)
        x = self.increment(x).permute(0, 2, 1).unsqueeze(-1)
        x = nn.PixelShuffle(2)(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class PatchReverseMerging4x(nn.Module):
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.patch_reverse_merging1 = PatchReverseMerging(dim, dim, norm_layer)
        self.patch_reverse_merging2 = PatchReverseMerging(dim, out_dim, norm_layer)

    def forward(self, x):
        x = self.patch_reverse_merging1(x)
        x = self.patch_reverse_merging2(x)
        return x


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, pixel_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    if pixel_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)


class RelativePosition2D(nn.Module):
    def __init__(self, num_heads, max_distance=16):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.rel_pos_table = nn.Parameter(
            torch.randn(2 * max_distance + 1, 2 * max_distance + 1, num_heads) * 0.02
        )

    def forward(self, H, W):
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel_coords = torch.clamp(rel_coords, -self.max_distance, self.max_distance)
        rel_coords += self.max_distance
        rel_pos_bias = self.rel_pos_table[rel_coords[0], rel_coords[1]]
        return rel_pos_bias.permute(2, 0, 1)


class AttentionWithRelPos(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, max_rel_dist=16):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.rel_pos = RelativePosition2D(num_heads, max_rel_dist)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rel_pos_bias = self.rel_pos(H, W)
        attn = attn + rel_pos_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class CustomBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, max_rel_dist=16, **kwargs):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, **kwargs)
        self.attn = AttentionWithRelPos(dim, num_heads, qkv_bias, max_rel_dist)

    def forward(self, x, H, W):
        x = x + self.drop_path1(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
