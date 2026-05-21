import torch.nn as nn


class projector_c2v(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(projector_c2v, self).__init__()
        self.B_in, self.C_in, self.H_in, self.W_in = dim_in
        self.B_out, self.L_out, self.C_out = dim_out
        H_p = W_p = int(self.L_out ** .5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((H_p, W_p))
        self.proj = nn.Linear(self.C_in, self.C_out)
        self.norm = nn.LayerNorm(self.C_out)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = self.norm(x)
        return x


class projector_c2c(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(projector_c2c, self).__init__()
        B_in, C_in, H, W = dim_in
        B_out, C_out, H_, W_ = dim_out
        self.adaptive_pool = nn.AdaptiveAvgPool2d((H_, W_))
        self.proj = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class projector_v2c(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(projector_v2c, self).__init__()
        self.B_in, self.L_in, self.C_in = dim_in
        self.B_out, self.C_out, self.H_out, self.W_out = dim_out
        H_p = W_p = int(self.L_in ** .5)
        self.upsample = nn.Upsample(scale_factor=(self.H_out // H_p, self.W_out // W_p), mode='bilinear')
        self.proj = nn.Conv2d(self.C_in, self.C_out, kernel_size=1, stride=1)
        self.norm = nn.BatchNorm2d(self.C_out)

    def forward(self, x):
        B, L, C = x.shape
        assert L == self.L_in, 'error'
        H_p = W_p = int(L ** .5)
        x = x.view(B, H_p, W_p, C).permute(0, 3, 1, 2)
        x = self.upsample(x)
        x = self.proj(x)
        x = self.norm(x)
        return x


class projector_v2v(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(projector_v2v, self).__init__()
        B, L_in, C_in = dim_in
        B, self.L_out, C_out = dim_out
        self.proj = nn.Linear(C_in, C_out)
        H_p = W_p = int(self.L_out ** .5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((H_p, W_p))
        self.norm = nn.LayerNorm(C_out)

    def forward(self, x):
        B, L, C = x.size()
        H = W = int(L ** .5)
        assert H * W == L, 'error'
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.adaptive_pool(x).permute(0, 2, 3, 1)
        x = self.proj(x).view(B, self.L_out, -1)
        x = self.norm(x)
        return x


def create_projector(dim_in, dim_out):
    projector = None
    if dim_in.__len__() == 3 and dim_out.__len__() == 4:
        projector = projector_v2c(dim_in, dim_out)
    elif dim_in.__len__() == 4 and dim_out.__len__() == 4:
        projector = projector_c2c(dim_in, dim_out)
    elif dim_in.__len__() == 4 and dim_out.__len__() == 3:
        projector = projector_c2v(dim_in, dim_out)
    elif dim_in.__len__() == 3 and dim_out.__len__() == 3:
        projector = projector_v2v(dim_in, dim_out)
    return projector

# if __name__ == '__main__':
#     data = (torch.randint(0, 255, [4, 256, 256], dtype=torch.float32) / 255.)
#     proj = projector_v2v(dim_in=[4, 256, 256], dim_out=[4, 196, 256])
#     x = proj(data)
#     print(1)
