import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, with_bn=True, bias=None):
        super().__init__()
        if bias is None:
            bias = not with_bn
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if with_bn else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(x))


class HighFreqEnhance(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.ks = kernel_size

    def forward(self, x):

        pad = self.ks // 2
        low = F.avg_pool2d(x, kernel_size=self.ks, stride=1, padding=pad)
        return x - low

class GlobalContextPPM(nn.Module):

    def __init__(self, dim, scales=(1, 2, 3, 6)):
        super().__init__()
        assert len(scales) > 0
        red = max(1, dim // len(scales))
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(dim, red, kernel_size=1, bias=True),
                nn.ReLU(inplace=True)
            ) for s in scales
        ])
        self.bottleneck = nn.Conv2d(dim + red * len(scales), dim, kernel_size=1, bias=True)

    def forward(self, x):
        H, W = x.size(-2), x.size(-1)
        feats = [x]
        for stage in self.stages:
            feats.append(F.interpolate(stage(x), size=(H, W), mode='bilinear', align_corners=False))
        out = torch.cat(feats, dim=1)
        return self.bottleneck(out)

class WeightedFusion(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        w = self.attn(x)
        x = x * w
        return self.proj(x)

class LightSelfAttention(nn.Module):

    def __init__(self, channels, num_heads=4, factor=8, proj_dim=None, attn_dropout=0.0):
        super().__init__()
        self.factor = factor
        self.num_heads = num_heads
        embed_dim = proj_dim if proj_dim is not None else channels
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.q = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.k = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.v = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.out = nn.Conv2d(embed_dim, channels, 1, bias=False)
        self.drop = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.factor > 1:
            x_low = F.avg_pool2d(x, kernel_size=self.factor, stride=self.factor)
        else:
            x_low = x
        Hl, Wl = x_low.shape[-2:]

        q = self.q(x_low).view(B, self.num_heads, self.head_dim, Hl * Wl).permute(0, 1, 3, 2)   # [B,h,N,hd]
        k = self.k(x_low).view(B, self.num_heads, self.head_dim, Hl * Wl).permute(0, 1, 3, 2)   # [B,h,N,hd]
        v = self.v(x_low).view(B, self.num_heads, self.head_dim, Hl * Wl).permute(0, 1, 3, 2)   # [B,h,N,hd]

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,h,N,N]
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = torch.matmul(attn, v)  # [B,h,N,hd]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.num_heads * self.head_dim, Hl, Wl)
        out = self.out(out)

        if self.factor > 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out


class Block(nn.Module):

    def __init__(self, dim=48, dilations=(1, 2, 4), sa_heads=4, sa_factor=8):
        super().__init__()
        self.dim = dim

        self.proj1 = ConvBN(96,  dim, kernel_size=1, with_bn=True)
        self.proj2 = ConvBN(192, dim, kernel_size=1, with_bn=True)
        self.proj3 = ConvBN(384, dim, kernel_size=1, with_bn=True)
        self.proj4 = ConvBN(768, dim, kernel_size=1, with_bn=True)

        self.fusion = WeightedFusion(in_channels=dim * 4, out_channels=dim, reduction=4)

        self.local_convs = nn.ModuleList([
            nn.Sequential(
                ConvBN(dim, dim, kernel_size=3, padding=d, dilation=d, groups=dim, with_bn=True),
                nn.ReLU(inplace=True),
                ConvBN(dim, dim, kernel_size=1, with_bn=True),
            ) for d in dilations
        ])
        self.local_merge = nn.Sequential(
            nn.Conv2d(dim * len(dilations), dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.hf = HighFreqEnhance(kernel_size=3)
        self.alpha_hf = nn.Parameter(torch.tensor(1.0))

        self.ppm = GlobalContextPPM(dim, scales=(1, 2, 3, 6))
        self.self_attn = LightSelfAttention(dim, num_heads=sa_heads, factor=sa_factor, proj_dim=None, attn_dropout=0.0)

        self.out_proj = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, f1, f2, f3, f4):

        B, _, H, W = f1.shape

        p1 = self.proj1(f1)
        p2 = F.interpolate(self.proj2(f2), size=(H, W), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=(H, W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.proj4(f4), size=(H, W), mode='bilinear', align_corners=False)

        fused = self.fusion(torch.cat([p1, p2, p3, p4], dim=1))  # [B,dim,H,W]

        local_feats = []
        for block in self.local_convs:
            local_feats.append(block(fused))
        local_agg = self.local_merge(torch.cat(local_feats, dim=1))  # [B,dim,H,W]
        local_out = local_agg + self.alpha_hf * self.hf(local_agg)

        global_ppm = self.ppm(fused)         # [B,dim,H,W]
        global_sa  = self.self_attn(fused)   # [B,dim,H,W]
        global_out = 0.5 * global_ppm + 0.5 * global_sa


        out = self.out_proj(torch.cat([fused, local_out, global_out], dim=1))
        return out



