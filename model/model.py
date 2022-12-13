import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .mobilenetv3 import MobileNetV3SmallEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner

from .onnx_helper import CustomOnnxResizeByFactorOp

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'mobilenetv3_small', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        sim_ratio = 0.5
        lraspp_out = int(sim_ratio * 128)
        dec_out = [int(sim_ratio * v) for v in [80, 40, 32, 16]]
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, lraspp_out)
            self.decoder = RecurrentDecoder([16, 24, 40, lraspp_out], dec_out)
        elif variant == 'mobilenetv3_small':
            self.backbone = MobileNetV3SmallEncoder(pretrained_backbone)
            self.aspp = LRASPP(576, lraspp_out)
            self.decoder = RecurrentDecoder([16, 16, 24, lraspp_out], dec_out)
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(dec_out[3], 4)
        self.project_seg = Projection(dec_out[3], 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self, src, r1, r2, r3, r4,
                downsample_ratio: float = 0.25,
                segmentation_pass: bool = True):
        
        if torch.onnx.is_in_onnx_export():
            # 导出静态模型，不需要自定义该算子
            # src_sm = CustomOnnxResizeByFactorOp.apply(src, downsample_ratio)
            # src_sm = self._interpolate(src, scale_factor=downsample_ratio)
            src_sm = src
        elif downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if torch.onnx.is_in_onnx_export() or downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [pha, *rec]
        else:
            seg = self.project_seg(hid)
            if not torch.onnx.is_in_onnx_export():
                seg = F.interpolate(seg, src.shape[2:], mode="bilinear", align_corners=False)
            seg = seg.sigmoid()
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.reshape(B, T, x.size(1), x.size(2), x.size(3))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
