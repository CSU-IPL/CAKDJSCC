from models.SwinJSCC.decoder import *
from models.SwinJSCC.encoder import *
from utils.distortion import Distortion
from utils.channel import Channel
from random import choice
import torch.nn as nn


class SwinJSCC_AT(nn.Module):
    def __init__(self, encoder_kwargs, decoder_kwargs, downsample, distortion_metric='MSE',
                 channel_type='awgn',  model='WITT', multiple_snr='1,4,7,10,13',
                 device=torch.device("cuda:0"), logger=None, CUDA=True, pass_channel=True, norm=False):
        super(SwinJSCC_AT, self).__init__()
        encoder_kwargs = encoder_kwargs
        decoder_kwargs = decoder_kwargs
        self.encoder = create_encoder(**encoder_kwargs)
        self.decoder = create_decoder(**decoder_kwargs)
        if logger is not None:
            logger.info("Network config: ")
            logger.info("Encoder: ")
            logger.info(encoder_kwargs)
            logger.info("Decoder: ")
            logger.info(decoder_kwargs)
        self.distortion_loss = Distortion(distortion_metric, logger)
        self.channel = Channel(channel_type, multiple_snr, device, logger, CUDA)
        self.pass_channel = pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.downsample = downsample
        self.model = model
        self.norm = norm

    def distortion_loss_wrapper(self, x_gen, x_real):
        distortion_loss = self.distortion_loss.forward(x_gen, x_real, normalization=self.norm)
        return distortion_loss

    def feature_pass_channel(self, feature, chan_param, avg_pwr=False):
        noisy_feature = self.channel.forward(feature, chan_param, avg_pwr)
        return noisy_feature

    def forward(self, input_image, given_SNR=None):
        B, _, H, W = input_image.shape

        if H != self.H or W != self.W:
            self.encoder.update_resolution(H, W)
            self.decoder.update_resolution(H // (2 ** self.downsample), W // (2 ** self.downsample))
            self.H = H
            self.W = W

        if given_SNR is None:
            SNR = choice(self.multiple_snr)
            chan_param = SNR
        else:
            chan_param = given_SNR

        feature = self.encoder(input_image, chan_param, self.model)

        CBR = feature.numel() / 2 / input_image.numel()
        # Feature pass channel
        if self.pass_channel:
            noisy_feature = self.feature_pass_channel(feature, chan_param)
        else:
            noisy_feature = feature

        recon_image = self.decoder(noisy_feature, chan_param, self.model)
        recon_image = recon_image.reshape(B, H, W, _).permute(0, 3, 1, 2)

        mse = self.squared_difference(input_image * 255., recon_image.clamp(0., 1.) * 255.)
        loss_G = self.distortion_loss.forward(input_image, recon_image.clamp(0., 1.))

        return recon_image, CBR, chan_param, mse.mean(), loss_G.mean()

if __name__ == '__main__':
    data = torch.rand(16, 3, 256, 256).to('cuda:0')
    model = SwinJSCC(
        encoder_kwargs=dict(
            img_size=(256, 256), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=128, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        ),
        decoder_kwargs=dict(
            img_size=(256, 256),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=128, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        ),
        downsample=4).to('cuda:0')
    recon_image, CBR, chan_param, mse, loss_G = model(data)
    print(1)