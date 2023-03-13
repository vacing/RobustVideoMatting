import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .onnx_helper import (
    CustomOnnxCropToMatchSizeOp
)

class RecurrentDecoderSim(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        # self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlockNew( decoder_channels[2], 3, decoder_channels[3])

    def forward(self, s0, f1, f2, f3, f4, r1, r2, r3, r4):
        s1, s2, s3 = self.avgpool(s0)
        # x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(f4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4
    

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        
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
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
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
        f = f.unflatten(0, (B, T))
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
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r = self.ih(torch.add(x, h))
        c = self.hh(torch.add(x, r))
        h = (1 - r) * h + r * c
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
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :s.size(1), :s.size(2)]
        else:
            x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)

class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=[1, 1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.conv(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :s.size(2), :s.size(3)]
        else:
            x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        x = self.conv(x)
        if not torch.onnx.is_in_onnx_export():
            x = x[:, :, :H, :W]
        else:
            x = CustomOnnxCropToMatchSizeOp.apply(x, s)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)
