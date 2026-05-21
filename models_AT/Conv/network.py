import torch
from torch import nn

from IFKD.models_AT.FFM_Module import FFM
from utils.distortion import Distortion
from utils.channel import Channel
from random import choice


class BasicBlockEnc(nn.Module):
    def __init__(self, depth, embed_dim, out_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.PReLU()
            ) for i in range(depth)]
        )
        self.conv = nn.Conv2d(embed_dim, out_dim, 3, stride=2, padding=1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    def __init__(self, depth, embed_dim, out_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, 3, stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(embed_dim),
                nn.PReLU()
            ) for i in range(depth)]
        )
        self.convT = nn.ConvTranspose2d(embed_dim, out_dim, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.convT(x)
        return x


class Encoder(nn.Module):
    def __init__(self, C, embed_dims, num_blocks, shape_s_list, shape_t_list,):
        super().__init__()
        self.C = C
        self.embed_dims = embed_dims
        self.num_blocks = num_blocks
        depth = len(num_blocks)

        self.conv0 = nn.Conv2d(3, embed_dims[0], 3, stride=1, padding=1)
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = BasicBlockEnc(depth=num_blocks[i], embed_dim=embed_dims[i - 1] if i != 0 else embed_dims[0],
                                  out_dim=embed_dims[i])
            self.layers.append(layer)
        self.ffms = nn.ModuleList()
        for i in range(len(num_blocks)):
            ffm = FFM(shape_s_list[i], shape_t_list[i])
            self.ffms.append(ffm)

        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.memory = {}

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dims[-1], C, 3, stride=1, padding=1),
            nn.BatchNorm2d(C)
        )

    def forward(self, x, s_list, t_list):
        x = self.conv0(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.ffms[i](s=s_list[i], t=t_list[i], x=x)
        x = self.norm(x)
        x = self.proj(x)
        self.memory['enc_output'] = x
        return x

    def get_feature(self, key):
        return self.memory[key]

class Decoder(nn.Module):
    def __init__(self, C, embed_dims, num_blocks):
        super().__init__()
        self.C = C
        self.embed_dims = embed_dims
        self.num_blocks = num_blocks
        depth = len(num_blocks)

        self.conv0 = nn.ConvTranspose2d(C, embed_dims[0], 3, stride=1, padding=1, output_padding=0)
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = BasicBlockDec(depth=num_blocks[i], embed_dim=embed_dims[i - 1] if i != 0 else embed_dims[0],
                                  out_dim=embed_dims[i])
            self.layers.append(layer)
        self.norm = nn.BatchNorm2d(embed_dims[0])
        self.memory = {}

        self.proj = nn.ConvTranspose2d(embed_dims[-1], 3, 3, stride=1, padding=1, output_padding=0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm(x)

        featurelist = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            featurelist.append(x)
        self.memory['feature'] = featurelist

        x = self.proj(x)
        return x


class ConvJSCC_AT(nn.Module):
    def __init__(self, C, encoder_embed_dims, encoder_num_blocks, decoder_embed_dims, decoder_num_blocks,
                 shape_s_list, shape_t_list,
                 distortion_metric='MSE', channel_type='awgn',
                 multiple_snr='1,4,7,10,13', device=torch.device("cuda:0"), logger=None,
                 CUDA=True, pass_channel=True):
        super(ConvJSCC_AT, self).__init__()
        self.encoder = Encoder(C, encoder_embed_dims, encoder_num_blocks, shape_s_list, shape_t_list).to(device)
        self.decoder = Decoder(C, decoder_embed_dims, decoder_num_blocks).to(device)
        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.distortion_loss = Distortion(distortion_metric, logger)
        self.channel = Channel(channel_type, multiple_snr, device, logger, CUDA)
        self.pass_channel = pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.device = device

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, s_list, t_list, given_SNR=None):
        B, _, H, W = input_image.shape

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, s_list, t_list)

        CBR = feature.numel() / 2 / input_image.numel()

        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        recon_image = self.decoder(noisy_feature)

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))

        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()


if __name__ == '__main__':
    data = torch.rand(48, 3, 256, 256).to('cuda:0')
    model = ConvJSCC_AT(C=128, encoder_embed_dims=[128, 192, 256, 320], encoder_num_blocks=[2, 2, 6, 2],
                     decoder_embed_dims=[320, 256, 192, 128], decoder_num_blocks=[2, 6, 2, 2]).to('cuda:0')
    recon_image, mse, loss_G, CBR, chan_param = model(data)
    print(1)
