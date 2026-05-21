import timm.models.vision_transformer
from utils.distortion import Distortion
from utils.channel import Channel
from random import choice
from IFKD.models_AT.VitJSCC.modules import *
from IFKD.models_AT.FFM_Module import FFM


class BasicBlockEnc(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, depth, **kwargs):
        super().__init__()
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']

        self.blocks = nn.ModuleList([
            CustomBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)]
        )

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class BasicBlockDec(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, depth, **kwargs):
        super().__init__()
        norm_layer = kwargs['norm_layer']
        embed_dim = kwargs['embed_dim']
        num_heads = kwargs['num_heads']
        mlp_ratio = kwargs['mlp_ratio']

        self.blocks = nn.ModuleList([
            CustomBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)]
        )

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class Encoder(nn.Module):
    """
    JSCC encoder with VisionTransformer backbone
    """

    def __init__(self, img_size, patch_size, C, embed_dim, num_Blocks, num_head, shape_s_list, shape_t_list,
                 norm_layer=nn.LayerNorm, mlp_ratio=4, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.C = C
        self.embed_dim = embed_dim
        self.num_Blocks = num_Blocks
        self.num_head = num_head

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.layers = nn.ModuleList()
        for i in range(len(num_Blocks)):
            layer = BasicBlockEnc(depth=int(num_Blocks[i]), embed_dim=embed_dim,
                                  norm_layer=norm_layer, num_heads=num_head, mlp_ratio=mlp_ratio)
            self.layers.append(layer)
        self.ffms = nn.ModuleList()
        for i in range(len(num_Blocks)):
            ffm = FFM(shape_s_list[i], shape_t_list[i])
            self.ffms.append(ffm)

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, C),
            nn.LayerNorm(C)
        )
        self.memory = {}
        self.initialize_weights()

    def forward(self, x, s_list, t_list, H_Patch, W_Patch):
        x = self.patch_embed(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, H_Patch, W_Patch)
            x = self.ffms[i](s=s_list[i], t=t_list[i], x=x)
        x = self.norm(x)
        x = self.proj(x)
        self.memory['enc_output'] = x
        return x

    def get_feature(self, key):
        return self.memory[key]

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Decoder(nn.Module):
    """
    JSCC encoder with VisionTransformer backbone
    """

    def __init__(self, img_size, patch_size, C, embed_dim, num_Blocks, num_head, norm_layer=nn.LayerNorm,
                 mlp_ratio=4, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.C = C
        self.embed_dims = embed_dim
        self.num_Blocks = num_Blocks
        self.num_heads = num_head

        self.num_patches = int(img_size // patch_size) ** 2

        self.proj = nn.Linear(C, embed_dim)
        self.layers = nn.ModuleList()
        for i in range(len(num_Blocks)):
            layer = BasicBlockDec(depth=int(num_Blocks[i]), embed_dim=embed_dim,
                                  norm_layer=norm_layer, num_heads=num_head, mlp_ratio=mlp_ratio)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(embed_dim)
        self.memory = {}
        self.initialize_weights()

    def forward(self, x, H_Patch, W_Patch):
        x = self.proj(x)
        x = self.norm(x)

        for i, layer in enumerate(self.layers):
            x = layer(x, H_Patch, W_Patch)
        self.memory['enc_output'] = x
        return x

    def get_feature(self, key):
        return self.memory[key]

    def initialize_weights(self):
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class VitJSCC_AT(nn.Module):
    def __init__(self, img_size, C, patch_size,
                 embed_dim, num_head, shape_s_list, shape_t_list,
                 encoder_num_blocks, decoder_num_blocks,
                 distortin_metric='MSE', logger=None, device=torch.device('cuda:0'),
                 is_channel=True, multiple_snr='1,4,7,10,13', channel_type='awgn'):
        super().__init__()
        self.encoder = Encoder(img_size, patch_size, C, embed_dim, encoder_num_blocks, num_head, shape_s_list, shape_t_list)
        self.decoder = Decoder(img_size, patch_size, C, embed_dim, decoder_num_blocks, num_head)

        self.C = C
        self.patch_size = patch_size

        self.is_channel = is_channel
        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.channel = Channel(channel_type, multiple_snr)

        self.distortion_loss = Distortion(distortin_metric, logger)
        self.squared_difference = torch.nn.MSELoss(reduction='none')

        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * 3, bias=True)

        self.apply(self._init_weights)

    def unpatchify(self, x, h, w):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, s_list, t_list, chan_param=None):
        B, _, H, W = input_image.size()
        if chan_param is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR

        H_Patch = int(H // self.patch_size)
        W_Patch = int(W // self.patch_size)
        feature = self.encoder(input_image, s_list, t_list, H_Patch, W_Patch)

        if self.is_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        CBR = feature.numel() / 2 / input_image.numel()
        output = self.decoder(noisy_feature, H_Patch, W_Patch)
        imgs = self.decoder_pred(output)
        recon_image = self.unpatchify(imgs, H_Patch, W_Patch)

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))
        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


if __name__ == '__main__':
    data = torch.rand(48, 3, 256, 256).to('cuda:0')
    model = VitJSCC_AT(img_size=256, C=128, patch_size=16,
                    encoder_num_blocks=[2, 2, 6, 2],
                    decoder_num_blocks=[2, 6, 2, 2],
                    embed_dim=768,
                    num_head=8).to('cuda:0')
    recon_image, mse, loss_G, CBR, chan_param = model(data)
    print(1)
