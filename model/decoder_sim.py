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
        self.chatt3   = ChannelAttention(feature_channels[2], feature_channels[2])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3*4**0, decoder_channels[0])
        self.chatt2   = ChannelAttention(feature_channels[1], feature_channels[1])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3*4**0, decoder_channels[1])
        self.chatt1   = ChannelAttention(feature_channels[0], feature_channels[0])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3*4**0, decoder_channels[2])
        self.decode0 = OutputBlockNew( decoder_channels[2], 3, decoder_channels[3])

    def forward(self, s0, f1, f2, f3, f4, r1, r2, r3, r4):
        s1, s2, s3 = self.avgpool(s0)
        # x4, r4 = self.decode4(f4, r4)
        f3 = self.chatt3(f3, f3)
        x3, r3 = self.decode3(f4, f3, s3, r3)
        f2 = self.chatt2(f2, f2)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        f1 = self.chatt1(f1, f1)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4
    

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
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # total_in = in_channels + skip_channels + 4
        total_in = in_channels + skip_channels + 3
        self.conv = nn.Sequential(
            nn.Conv2d(total_in, total_in, 3, 1, 1, groups=total_in, bias=False),
            nn.BatchNorm2d(total_in),
            nn.Conv2d(total_in, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r):
        x = self.upsample(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :s.size(2), :s.size(3)]
        else:
            x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
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
            x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
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
        self.conv = nn.Sequential(
            nn.Conv2d(rin_channels, rin_channels, 3, 1, 1, groups=rin_channels, bias=False),
            nn.BatchNorm2d(rin_channels),
            nn.Conv2d(rin_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
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
            x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        # x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        # x = torch.cat([x, s], dim=1)
        x = self.conv(x)
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding)
    
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
    