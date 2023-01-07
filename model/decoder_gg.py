import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode3 = UpsamplingBlock(feature_channels[3], decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], decoder_channels[2])
        self.decode0 = OutputBlockNew( decoder_channels[2], decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):
        s1, s2, s3 = self.avgpool(s0)
        x3 = self.decode3(f4, f3, s3)
        x2 = self.decode2(x3, f2, s2)
        x1 = self.decode1(x2, f1, s1)
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

class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True),
        )
        self.gru = ConvGRU(out_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, groups=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward_single_frame(self, x, f, s):
        _, _, H, W = f.shape
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = self.conv(x)
        x = self.gru(x, f)
        b = self.conv3(x)
        x = x + b
        return x
    
    def forward_time_series(self, x, f, s):
        B, T, _, H, W = f.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = self.conv(x)
        x = self.gru(b, f)
        b = self.conv3(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, f, s: Optional[Tensor]):
        print(x.shape, f.shape, s.shape)
        if x.ndim == 5:
            return self.forward_time_series(x, f, s)
        else:
            return self.forward_single_frame(x, f, s)

class ConvGRU(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, input_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(input_channels, input_channels, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward_single_frame(self, x, h):
        sum = torch.add(x, h)
        scale = F.adaptive_avg_pool2d(sum, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        # return scale.add(3).clamp(0, 6).div(6)
        scale = F.hardsigmoid(scale, inplace=True)
        return self.conv(scale * h + x)
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            h = self.forward_single_frame(xt, h)
            o.append(h)
        o = torch.stack(o, dim=1)
        return o
        
    def forward(self, x, h):
        print(x.shape, h.shape)
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)

class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=[1, 1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.conv(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = x[:, :, :H, :W]
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)

class OutputBlockNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            # 开销最大的conv
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
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

class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
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
    