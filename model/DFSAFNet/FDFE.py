import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numbers
import torch.fft
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, with_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_channels) if with_bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))


class modReLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        # z: complex tensor, shape [B, C, H, W]
        mag = torch.abs(z)
        phase = torch.angle(z)
        activated_mag = torch.relu(mag + self.bias)
        return activated_mag * torch.exp(1j * phase)


class FilterGenerator(nn.Module):

    def __init__(self, in_channels=1, hidden=32, out_stride=8, base_init=0.5):

        super().__init__()
        self.out_stride = out_stride


        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, bias=True),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1, bias=True),  # H/4
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
        )


        bias_val = math.log(base_init / (1.0 - base_init))
        self.register_buffer("initial_bias", torch.tensor(bias_val, dtype=torch.float32))

        self.out_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_mag):

        B, C, H, W = x_mag.shape
        feat = self.encoder(x_mag)  # shape [B, hidden, H/4, W/4] (depending on encoder)
        feat = F.interpolate(feat, size=(max(1, H // self.out_stride), max(1, W // self.out_stride)),
                             mode='bilinear', align_corners=False)
        raw = self.decoder(feat)  # [B,1,H/out_stride,W/out_stride]


        raw = raw + (self.initial_bias + self.out_bias)
        lowres = torch.sigmoid(raw)  # [B,1,h0,w0] in (0,1)


        filter_map = F.interpolate(lowres, size=(H, W), mode='bilinear', align_corners=False)
        return filter_map


class DynamicFreqEnhance(nn.Module):

    def __init__(self, base_filter='directional_highpass', gen_hidden=32, gen_stride=8, base_scale=1.0):
        super().__init__()
        self.modrelu = modReLU()
        self.filter_gen = FilterGenerator(in_channels=1, hidden=gen_hidden, out_stride=gen_stride, base_init=0.5)
        self.base_scale = base_scale


        base = self._create_base_filter(size=256, kind=base_filter)  # base size 256，后续会 resize
        self.register_buffer('base_filter', base)

    def _create_base_filter(self, size=256, kind='directional_highpass'):
        freqs = torch.fft.fftfreq(size)
        fx, fy = torch.meshgrid(freqs, freqs, indexing='ij')
        if kind == 'directional_highpass':
            directional_filter = torch.exp(-((fx)**2 + (4 * fy)**2))
            highpass_mask = 1 - torch.exp(-20 * (fx**2 + fy**2))
            base = (directional_filter * highpass_mask)
        elif kind == 'gaussian_lowpass':
            base = torch.exp(- (fx**2 + fy**2) * 10.0)
        else:
            base = torch.ones_like(fx)
        base = base.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
        return base

    def forward(self, x):

        B, C, H, W = x.shape


        x_fft = torch.fft.fft2(x, norm='ortho')           # complex [B,C,H,W]
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2,-1))


        mag = torch.abs(x_fft_shifted)                    # [B, C, H, W]
        mag_mean = mag.mean(dim=1, keepdim=True)

        gen_input = torch.log1p(mag_mean)
        gen_input = (gen_input - gen_input.mean(dim=(-2,-1), keepdim=True)) / (gen_input.std(dim=(-2,-1), keepdim=True) + 1e-6)

        dyn_filter = self.filter_gen(gen_input)           # [B,1,H,W], in (0,1)


        if self.base_filter.shape[-2:] != (H,W):
            base_resized = F.interpolate(self.base_filter, size=(H,W), mode='bilinear', align_corners=False)
        else:
            base_resized = self.base_filter  # [1,1,H,W]


        final_filter = dyn_filter * base_resized

        final_filter = final_filter * self.base_scale

        x_filtered = x_fft_shifted * final_filter

        x_activated = self.modrelu(x_filtered)

        x_ifft = torch.fft.ifftshift(x_activated, dim=(-2,-1))
        x_spatial = torch.fft.ifft2(x_ifft, norm='ortho').real  # [B,C,H,W], real

        out = x + x_spatial
        return out


class MDAFF(nn.Module):
    def __init__(self, dim, decode_channels=16,num_heads=8, LayerNorm_type='WithBias'):
        super(MDAFF, self).__init__()
        self.num_heads = num_heads
        self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.fftseglayer1 = DynamicFreqEnhance(base_scale=1.0, gen_hidden=24, gen_stride=8)
        #self.fftseglayer2 = FFTSegLayer(dim, num_heads)
    def forward(self,feat2,feat3, feat4,feat5):
        res1h, res1w = feat2.size()[-2:]
        feat3 = self.conv2(feat3)
        feat4 = self.conv3(feat4)
        feat5 = self.conv4(feat5)
        feat3 = F.interpolate(feat3, size=(res1h, res1w), mode='bicubic', align_corners=False)

        feat4 = F.interpolate(feat4, size=(res1h, res1w), mode='bicubic', align_corners=False)

        feat5 = F.interpolate(feat5, size=(res1h, res1w), mode='bicubic', align_corners=False)
        input = torch.cat([feat3, feat4, feat5], dim=1)
        x1 = self.norm1(input)
        #x2 = self.norm2(input)

        att1_fft = self.fftseglayer1(x1)
        #att2_fft = self.fftseglayer1(x2)
        #att3_fft = self.fftseglayer2(x1)
        #att4_fft = self.fftseglayer2(x2)


        out = (self.project_out(att1_fft)+ x1)

        return out



