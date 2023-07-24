import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from .lraspp import ChannelAttention

from .onnx_helper import (
    CustomOnnxCropToMatchSizeOp
)

class RecurrentDecoderSim(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        # self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3*4**0, decoder_channels[0], 0, True, ratio=3)
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3*4**0, decoder_channels[1], 0, True, ratio=4)
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3*4**0, decoder_channels[2], 0, False, ratio=3)
        self.decode0 = OutputBlockNew( decoder_channels[2], 3, decoder_channels[3])

    def forward(self, s0, f1, f2, f3, f4, r1, r2, r3):
        s1, s2, s3 = self.avgpool(s0)
        # x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(f4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3
    

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        # self.avgpool = nn.PixelUnshuffle(2)
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)
        
    def forward(self, x, r):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        return x, r

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels, us_type:int, use_src: bool,  ks=3, ratio=2):
        super().__init__()
        self.out_channels = out_channels
        self.use_src = use_src
        us_out = in_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if us_type == 1:    # shuffle
            assert(in_channels % 4 == 0)
            self.use_src = True
            self.upsample = nn.PixelShuffle(2)
            us_out = in_channels // 4
        elif us_type == 2:   # transpose conv
            self.use_src = True
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        if self.use_src:
            total_in = us_out + skip_channels + 3
        else:
            total_in = us_out + skip_channels + 0
        self.chatt = ChannelAttention(skip_channels, in_channels)
        padding = 1 if ks == 3 else 2
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, total_in, ks, 1, padding=padding, groups=total_in, bias=False),
            nn.BatchNorm2d(total_in),
            nn.Conv2d(total_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.bypass_channels = out_channels // ratio
        self.gru_channels = out_channels - self.bypass_channels
        self.gru = ConvGRU(self.gru_channels)

    def forward_single_frame(self, x, f, s, r):
        x = self.upsample(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :s.size(2), :s.size(3)]
        else:
            x = x
            # x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        f = self.chatt(f, x)
        if self.use_src:
            x = torch.cat([x, f, s], dim=1)
        else:
            x = torch.cat([x, f], dim=1)
        x = self.conv(x)
        a, b = x.split([self.bypass_channels, self.gru_channels], dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r
    
    def forward_time_series(self, x, f, s, r):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :H, :W]
        else:
            x = x
            # x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        f = self.chatt(f, x)
        if self.use_src:
            x = torch.cat([x, f, s], dim=1)
        else:
            x = torch.cat([x, f], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split([self.bypass_channels, self.gru_channels], dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r
    
    def forward(self, x, f, s, r):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)

class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size, padding=padding, groups=channels*2, bias=False),
            nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels*2, channels*2, kernel_size=1),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size, padding=padding, groups=channels*2, bias=False),
            nn.BatchNorm2d(channels*2),
            nn.Conv2d(channels*2, channels, kernel_size=1),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * c + z * h
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h):
        h = h.expand_as(x)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class OutputBlockNew(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        rin_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(rin_channels, rin_channels, 3, 1, 1, groups=rin_channels, bias=False),
            nn.BatchNorm2d(rin_channels),
            nn.Conv2d(rin_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=2, groups=out_channels, bias=False, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :s.size(1), :s.size(2)]
        else:
            x = x
            # x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        # x = torch.cat([x, s], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        # x = x[:, :, :H, :W]
        # x = torch.cat([x, s], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)

class ProjectionSim(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=1):
        super().__init__()
        padding = 0
        if kernel == 3:
            padding = 1
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
        )
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    