import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from model.DFSAFNet.SMSF import Block
from model.DFSAFNet.TCDAF import CoDiAF
from model.DFSAFNet.FDFE import MDAFF


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()    #Weighted Fusion,具有可学习的融合权重
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):

        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, with_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_channels) if with_bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class BlockL(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        #垂直条纹卷积
        self.conv1_1_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        #self.conv1_1_2 = nn.Conv2d(dim, dim, (1, 9), padding=(0, 4), groups=dim)
        self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        #self.conv1_2_2 = nn.Conv2d(dim, dim, (9, 1), padding=(4, 0), groups=dim)
        #self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        #self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1_1_1(x)+self.conv1_2_1(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.conv1_1_1(self.g(x))+self.conv1_2_1(self.g(x))
        x = input + self.drop_path(x)
        return x


class Reswaveattention(nn.Module):
    def __init__(self, conv=default_conv, dim=48):
        super(Reswaveattention, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.head = conv(dim, dim, kernel_size)
        self.headl = conv(dim, dim, kernel_size)

        self.body1L = BlockL(dim, mlp_ratio=3)
        self.body1H = BlockL(dim, mlp_ratio=3)
        self.body2L = BlockL(dim, mlp_ratio=3)
        self.body2H = BlockL(dim, mlp_ratio=3)
        self.body3L = BlockL(dim, mlp_ratio=3)
        self.body3H = BlockL(dim, mlp_ratio=3)
        self.body4L = BlockL(dim, mlp_ratio=3)
        self.body4H = BlockL(dim, mlp_ratio=3)
        self.convout = conv(2 * dim, dim, kernel_size)
        self.out_dim = dim

    def forward(self, x, xl):   #低频高频
        #x = self.head(x)
        #xl = self.headl(xl)

        resx = self.body1L(x)
        resxl = self.body1H(xl)
        resx = self.body2L(resx)
        resxl = self.body2H(resxl)
        resx = self.body3L(resx)
        resxl = self.body3H(resxl)
        resx = self.body4L(resx)
        resxl = self.body4H(resxl)

        rescat = torch.cat([resx, resxl], dim=1)
        res = self.convout(rescat)
        final = res + x+xl
        return final


class Dwt(nn.Module):
    def __init__(self, in_ch, out_ch,decode_channels=48):
        super(Dwt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)
        self.rsewave = Reswaveattention(conv=default_conv)
        self.glblock = Block(dim=48)
        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat2, feat3, feat4, feat5):
        res1h, res1w = feat2.size()[-2:]
        feat3 = self.conv2(feat3)
        feat4 = self.conv3(feat4)
        feat5 = self.conv4(feat5)
        feat3 = F.interpolate(feat3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        # bicubic是双三次插值（Bicubic interpolation）
        feat4 = F.interpolate(feat4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        # align_corners=False 是深度学习中的常见选择，确保数值稳定，避免插值误差累积
        feat5 = F.interpolate(feat5, size=(res1h, res1w), mode='bicubic', align_corners=False)
        x = torch.cat([feat3, feat4, feat5], dim=1)
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)

        yL = self.outconv_bn_relu_L(yL)
        yH = self.outconv_bn_relu_H(yH)
        final = self.rsewave(yL, yH)
        final = self.upsample(final)
        return final

class DFSAFNet(nn.Module):
    def __init__(self,
                 decode_channels=48,
                 dropout=0.1,
                 #backbone_name="convnextv2_base.fcmae_ft_in22k_in1k_384",
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=True,
                 window_size=8,
                 num_classes=6,
                 use_aux_loss=True
                 ):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained,
                                          output_stride=32, out_indices=(0, 1, 2, 3))

        #in_filters = [192, 192, 384, 1152]
        #in_filters = [192, 256, 512, 1536]
        out_filters = [24, 48, 96, 192]
        #out_filters = [32, 64, 128, 256]
        self.WF1 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.WF2 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        # self.MDAF_L = CoDiAF(channels=decode_channels, num_heads=4, down_factor=8, proj_dim=None, reduction=8, attn_dropout=0.0)
        # self.MDAF_H = CoDiAF(channels=decode_channels, num_heads=4, down_factor=8, proj_dim=None, reduction=8, attn_dropout=0.0)
        self.MDAF_L = CoDiAF(dim=decode_channels)
        self.MDAF_H = CoDiAF(dim=decode_channels)
        self.MDAF = MDAFF(dim=decode_channels,decode_channels=16,num_heads=8,LayerNorm_type='WithBias')
        self.dwt = Dwt(in_ch=3*48, out_ch=48)
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[1], out_filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_filters[1], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.convjw = nn.Sequential(
            nn.Conv2d(in_channels=out_filters[2], out_channels=out_filters[1], kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(out_filters[1]),
        )
        self.upsample2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        [feat2, feat3, feat4,feat5] = self.backbone(inputs)[0:]   #C:
        up2_dwt = self.dwt(feat2, feat3, feat4,feat5)
        up2_glb = self.glblock(feat2, feat3, feat4, feat5)
        up2_Fourier = self.MDAF(feat2, feat3, feat4, feat5)
        feat2_2 = self.convjw(feat2)
        # up2_L = self.MDAF_L(up2_dwt, up2_glb)
        # up2_H = self.MDAF_H(up2_Fourier,up2_glb)
        up2_L = self.MDAF_L (up2_dwt, up2_glb,up2_Fourier)
        up2_L = up2_L+feat2_2
        #up2_H = up2_H + feat2_2
        #up2 = self.WF1(up2_L, up2_H)
        up1 = self.up_conv(up2_L)
        up1=self.upsample2(up1)
        final = self.final(up1)
        return final



if __name__ == '__main__':
    input = torch.randn(1, 3, 1024,1024)
    sft = DFSAFNet(pretrained=True, num_classes=6)
    output=sft(input)
    print(output.shape)
