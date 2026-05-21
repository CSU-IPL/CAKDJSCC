import torch
from IFKD.utils.projector import *


class FFM(nn.Module):
    def __init__(self, shape_s, shape_t):
        super(FFM, self).__init__()
        self.shape_s = shape_s
        self.shape_t = shape_t
        self.proj_t = create_projector(shape_t, shape_s)

        if shape_s.__len__() == 3:
            self.embed_dim = shape_s[-1]
        else:
            self.embed_dim = shape_s[1]

        self.linear1 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Tanh()
        )

        self.mlp = nn.Sequential(
            nn.Linear(3 * self.embed_dim, self.embed_dim),
            nn.GELU(),
        )
        self.attn_net = nn.Linear(self.embed_dim, 1)

        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, s, t, x):
        assert s.shape[1:] == self.shape_s[1:] and t.shape[1:] == self.shape_t[1:] and s.shape == x.shape, "shape mismatch"
        is_conv_features = (s.dim() == 4)
        if is_conv_features:
            B, C, H, W = s.shape
            reshape = lambda v: v.permute(0, 2, 3, 1).reshape(B, H * W, C)
            reshapeT = lambda v: v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            B, L, C = s.shape
            reshape = lambda v: v
            reshapeT = lambda v: v

        z_s = self.linear1(reshape(s))
        z_t = self.linear2(reshape(self.proj_t(t)))
        z_x = self.linear3(reshape(x))

        combined = self.mlp(torch.cat([z_s, z_t, z_x], dim=-1)) # B, L, embed_dim
        attn = self.softmax(self.attn_net(combined))  # B, L, 1
        fused = self.norm(attn * combined + combined)

        return reshapeT(fused)
