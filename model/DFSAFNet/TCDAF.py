import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm(nn.Module):
    def __init__(self, dim, bias_free=True):
        super().__init__()
        if bias_free:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = None
        else:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        self.bias_free = bias_free
        self.eps = 1e-5

    def forward(self, x):
        h, w = x.shape[-2:]
        x_3d = rearrange(x, 'b c h w -> b (h w) c')
        if self.bias_free:
            var = x_3d.var(-1, keepdim=True, unbiased=False)
            x_3d = x_3d / torch.sqrt(var + self.eps) * self.weight
        else:
            mean = x_3d.mean(-1, keepdim=True)
            var = x_3d.var(-1, keepdim=True, unbiased=False)
            x_3d = (x_3d - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
        return rearrange(x_3d, 'b (h w) c -> b c h w', h=h, w=w)

class DifferenceGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_ws = nn.Conv2d(dim, dim, 1)
        self.conv_fs = nn.Conv2d(dim, dim, 1)
        self.conv_fw = nn.Conv2d(dim, dim, 1)

    def forward(self, F_wav, F_fft, F_spa):
        D_ws = torch.abs(F_wav - F_spa)
        D_fs = torch.abs(F_fft - F_spa)
        D_fw = torch.abs(F_fft - F_wav)
        G_ws = torch.sigmoid(self.conv_ws(D_ws))
        G_fs = torch.sigmoid(self.conv_fs(D_fs))
        G_fw = torch.sigmoid(self.conv_fw(D_fw))
        Gwav = (G_ws + G_fw) / 2
        Gfft = (G_fs + G_fw) / 2
        Gspa = (G_ws + G_fs) / 2
        return F_wav * Gwav, F_fft * Gfft, F_spa * Gspa

class AxialAttention(nn.Module):
    def __init__(self, dim, axis='h', heads=4):
        super().__init__()
        self.axis = axis
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)
        if self.axis == 'h':  # 垂直依赖
            q = rearrange(q, 'b (head c) h w -> b head h (w c)', head=self.heads)
            k = rearrange(k, 'b (head c) h w -> b head h (w c)', head=self.heads)
            v = rearrange(v, 'b (head c) h w -> b head h (w c)', head=self.heads)
        else:  # 水平依赖
            q = rearrange(q, 'b (head c) h w -> b head w (h c)', head=self.heads)
            k = rearrange(k, 'b (head c) h w -> b head w (h c)', head=self.heads)
            v = rearrange(v, 'b (head c) h w -> b head w (h c)', head=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        if self.axis == 'h':
            out = rearrange(out, 'b head h (w c) -> b (head c) h w', head=self.heads, h=H, w=W)
        else:
            out = rearrange(out, 'b head w (h c) -> b (head c) h w', head=self.heads, h=H, w=W)
        return self.proj_out(out)

class HybridDirectionalConv(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()

        self.horiz_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, (1,7), padding=(0,3), groups=dim),  # 连续大核
            nn.Conv2d(dim, dim, (1,3), padding=(0,3), dilation=(1,3), groups=dim),
            nn.Conv2d(dim, dim, (1,3), padding=(0,5), dilation=(1,5), groups=dim)
        ])
        self.horiz_attn = AxialAttention(dim, axis='w', heads=heads)


        self.vert_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, (7,1), padding=(3,0), groups=dim),  # 连续大核
            nn.Conv2d(dim, dim, (3,1), padding=(3,0), dilation=(3,1), groups=dim),
            nn.Conv2d(dim, dim, (3,1), padding=(5,0), dilation=(5,1), groups=dim)
        ])
        self.vert_attn = AxialAttention(dim, axis='h', heads=heads)

        self.diag_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, (3,3), padding=(1,1), groups=dim),
            nn.Conv2d(dim, dim, (5,5), padding=(2,2), groups=dim)
        ])


        self.proj = nn.Conv2d(dim * 3, dim, kernel_size=1)
        self.local_mix = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        horiz_out = sum([conv(x) for conv in self.horiz_convs]) + self.horiz_attn(x)
        vert_out = sum([conv(x) for conv in self.vert_convs]) + self.vert_attn(x)
        diag_out = sum([conv(x) for conv in self.diag_convs])
        merged = torch.cat([horiz_out, vert_out, diag_out], dim=1)
        merged = self.proj(merged)
        return self.local_mix(merged)

class TripletAxialAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.proj_q = nn.Conv2d(dim, dim, 1)
        self.proj_k = nn.Conv2d(dim*2, dim, 1)
        self.proj_v = nn.Conv2d(dim*2, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward_axial(self, q_in, kv_in, axis='h'):
        B, C, H, W = q_in.shape
        q = self.proj_q(q_in)
        k = self.proj_k(kv_in)
        v = self.proj_v(kv_in)
        if axis == 'h':
            q = rearrange(q, 'b (head c) h w -> b head h (w c)', head=self.heads)
            k = rearrange(k, 'b (head c) h w -> b head h (w c)', head=self.heads)
            v = rearrange(v, 'b (head c) h w -> b head h (w c)', head=self.heads)
        else:
            q = rearrange(q, 'b (head c) h w -> b head w (h c)', head=self.heads)
            k = rearrange(k, 'b (head c) h w -> b head w (h c)', head=self.heads)
            v = rearrange(v, 'b (head c) h w -> b head w (h c)', head=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        if axis == 'h':
            out = rearrange(out, 'b head h (w c) -> b (head c) h w', head=self.heads, h=H, w=W)
        else:
            out = rearrange(out, 'b head w (h c) -> b (head c) h w', head=self.heads, h=H, w=W)
        return out

    def forward(self, f1, f2, f3):
        o1 = self.proj_out(self.forward_axial(f1, torch.cat([f2, f3], 1), 'h') +
                           self.forward_axial(f1, torch.cat([f2, f3], 1), 'w'))
        o2 = self.proj_out(self.forward_axial(f2, torch.cat([f1, f3], 1), 'h') +
                           self.forward_axial(f2, torch.cat([f1, f3], 1), 'w'))
        o3 = self.proj_out(self.forward_axial(f3, torch.cat([f1, f2], 1), 'h') +
                           self.forward_axial(f3, torch.cat([f1, f2], 1), 'w'))
        return o1, o2, o3

class CoDiAF(nn.Module):
    def __init__(self, dim, heads=4, bias_free=True):
        super().__init__()
        self.norm_wav = LayerNorm(dim, bias_free)
        self.norm_fft = LayerNorm(dim, bias_free)
        self.norm_spa = LayerNorm(dim, bias_free)
        self.diff_gate = DifferenceGating(dim)

        self.hdc_wav = HybridDirectionalConv(dim, heads)
        self.hdc_fft = HybridDirectionalConv(dim, heads)
        self.hdc_spa = HybridDirectionalConv(dim, heads)

        self.tattn_full = TripletAxialAttention(dim, heads)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, F_wav, F_fft, F_spa):
        F_wav, F_fft, F_spa = self.norm_wav(F_wav), self.norm_fft(F_fft), self.norm_spa(F_spa)
        F_wav, F_fft, F_spa = self.diff_gate(F_wav, F_fft, F_spa)

        F_wav_en = self.hdc_wav(F_wav)
        F_fft_en = self.hdc_fft(F_fft)
        F_spa_en = self.hdc_spa(F_spa)

        o1, o2, o3 = self.tattn_full(F_wav_en, F_fft_en, F_spa_en)
        merged = (o1 + o2 + o3) / 3
        return self.proj_out(merged) + F_wav_en + F_fft_en + F_spa_en



