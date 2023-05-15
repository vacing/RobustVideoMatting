from torch import nn

class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward_single_frame(self, x):
        return self.aspp1(x) * self.aspp2(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ref_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ref_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward_single_frame(self, x, ref):
        return self.att(x) * self.aspp1(x)
    
    def forward_time_series(self, x, ref):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1), ref.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x, ref):
        if x.ndim == 5:
            return self.forward_time_series(x, ref)
        else:
            return self.forward_single_frame(x, ref)