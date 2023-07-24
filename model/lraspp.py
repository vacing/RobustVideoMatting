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

# https://github.com/LeeJunHyun/Image_Segmentation/blob/db34de21767859e035aee143c59954fa0d94bbcd/network.py#L108
class ChannelAttention(nn.Module):
    def __init__(self, x_channels, g_channels):
        super().__init__()
        self.wx = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, 1, bias=False),
            nn.BatchNorm2d(x_channels),
        )
        self.wg = nn.Sequential(
            nn.Conv2d(g_channels, x_channels, 1, bias=False),
            nn.BatchNorm2d(x_channels),
        )
        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(x_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        # self.lraspp = LRASPP(x_channels, x_channels)
        
    def forward_single_frame(self, x, ref):
        x1 = self.wx(x)
        g1 = self.wg(ref)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        # x = self.lraspp(x)
        return x * psi
    
    def forward_time_series(self, x, ref):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1), ref.flatten(0, 1)).unflatten(0, (B, T))
        return x
    
    def forward(self, x, ref):
        if x.ndim == 5:
            return self.forward_time_series(x, ref)
        else:
            return self.forward_single_frame(x, ref)