import torch
from torch import nn
import torch.nn.functional as F

from IFKD.models_AT.FFM_Module import FFM
from utils.distortion import Distortion
from utils.channel import Channel
from random import choice

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, planes=None, stride=1):
        super().__init__()
        if planes is None:
            planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act = nn.PReLU()
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, planes=None, stride=1):
        super().__init__()
        if planes is None:
            planes = int(in_planes / stride)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.act(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


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


class ResNetEnc(nn.Module):
    def __init__(self, C, embed_dims, num_Blocks, shape_s_list, shape_t_list):
        super().__init__()
        self.num_blocks = num_Blocks
        self.embed_dims = embed_dims

        self.in_planes = self.embed_dims[0]
        self.conv1 = nn.Conv2d(3, self.embed_dims[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.embed_dims[0])
        self.layer0 = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.PReLU()
        )
        self.layers = nn.ModuleList()
        self.layer1 = self._make_layer(BasicBlockEnc, self.embed_dims[0], num_Blocks[0], stride=1)
        self.layers.append(self.layer1)
        self.layer2 = self._make_layer(BasicBlockEnc, self.embed_dims[1], num_Blocks[1], stride=2)
        self.layers.append(self.layer2)
        self.layer3 = self._make_layer(BasicBlockEnc, self.embed_dims[2], num_Blocks[2], stride=2)
        self.layers.append(self.layer3)
        self.layer4 = self._make_layer(BasicBlockEnc, self.embed_dims[3], num_Blocks[3], stride=2)
        self.layers.append(self.layer4)
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dims[-1], C, 3, stride=1, padding=1),
            nn.BatchNorm2d(C)
        )
        self.ffms = nn.ModuleList()
        for i in range(len(num_Blocks)):
            ffm = FFM(shape_s_list[i], shape_t_list[i])
            self.ffms.append(ffm)
        self.memory = {}

        self.norm = nn.BatchNorm2d(C)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, s_list, t_list):
        x = self.layer0(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.ffms[i](s=s_list[i], t=t_list[i], x=x)

        x = self.proj(x)
        x = self.norm(x)
        self.memory['enc_output'] = x
        return x

    def get_feature(self, key):
        return self.memory[key]


class ResNetDec(nn.Module):

    def __init__(self, C, embed_dims, num_Blocks):
        super().__init__()
        self.num_Blocks = num_Blocks
        self.embed_dims = embed_dims

        self.in_planes = self.embed_dims[3]
        self.conv0 = nn.ConvTranspose2d(C, embed_dims[3], 3, stride=1, padding=1, output_padding=0)
        self.norm = nn.BatchNorm2d(self.embed_dims[3])
        self.layers = nn.ModuleList()
        self.layer4 = self._make_layer(BasicBlockDec, self.embed_dims[2], num_Blocks[3], stride=2)
        self.layers.append(self.layer4)
        self.layer3 = self._make_layer(BasicBlockDec, self.embed_dims[1], num_Blocks[2], stride=2)
        self.layers.append(self.layer3)
        self.layer2 = self._make_layer(BasicBlockDec, self.embed_dims[0], num_Blocks[1], stride=2)
        self.layers.append(self.layer2)
        self.layer1 = self._make_layer(BasicBlockDec, self.embed_dims[0], num_Blocks[0], stride=1)
        self.layers.append(self.layer1)
        self.conv1 = ResizeConv2d(self.embed_dims[0], 3, kernel_size=3, scale_factor=2)
        self.layer0 = nn.Sequential(
            self.conv1,
            nn.PReLU()
        )

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.norm(self.conv0(x))

        for i, layer in enumerate(self.layers):
            x = layer(x)
        recon_image = self.layer0(x)
        return recon_image


class ResNetAE(nn.Module):
    def __init__(self, C, embed_dims, num_blocks, shape_s_list, shape_t_list,
                 distortion_metric='MSE', channel_type='awgn',
                 multiple_snr='1,4,7,10,13', device=torch.device("cuda:0"), logger=None,
                 CUDA=True, pass_channel=True, norm=False):
        super(ResNetAE, self).__init__()
        self.encoder = ResNetEnc(C, embed_dims, num_blocks, shape_s_list, shape_t_list).to(device)
        self.decoder = ResNetDec(C, embed_dims, num_blocks).to(device)
        self.multiple_snr = multiple_snr.split(",")
        for i in range(len(self.multiple_snr)):
            self.multiple_snr[i] = int(self.multiple_snr[i])
        self.distortion_loss = Distortion(distortion_metric, logger)
        self.channel = Channel(channel_type, multiple_snr, device, logger, CUDA)
        self.pass_channel = pass_channel
        self.squared_difference = torch.nn.MSELoss(reduction='none')
        self.H = self.W = 0
        self.norm = norm
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
    data = torch.rand(32, 3, 256, 256).to('cuda:2')
    model = ResNetAE(C=128, embed_dims=[128, 192, 256, 320], num_blocks=[2, 2, 18, 2]).to('cuda:2')
    recon_image, mse, loss_G, CBR, chan_param = model(data)
    print(1)
