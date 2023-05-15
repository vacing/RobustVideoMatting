"""
python export_onnx.py \
    --model-variant mobilenetv3 \
    --checkpoint rvm_mobilenetv3.pth \
    --precision float16 \
    --opset 12 \
    --device cuda \
    --output model.onnx
    
Note:
    The device is only used for exporting. It has nothing to do with the final model.
    Float16 must be exported through cuda. Float32 can be exported through cpu.
"""

import argparse
import torch
import math

from model import MattingNetwork


class Exporter:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.export()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True,
                            choices=['mobilenetv3', 'mobilenetv3_small', 'mobilenetv3_smaller', 'mobilenetv3_sim', 'resnet50'])
        parser.add_argument('--model-decoder', type=str, required=True,
                            choices=['rvm', 'rvm_small', 'rvm_sim', 'rvm_sim_small', 'gg'])
        parser.add_argument('--refiner', type=str, default='deep_guided_filter', choices=['deep_guided_filter', 'fast_guided_filter'])
        parser.add_argument('--seg', type=bool, required=False, default=False)
        parser.add_argument('--precision', type=str, required=True, choices=['float16', 'float32'])
        parser.add_argument('--opset', type=int, required=True)
        parser.add_argument('--device', type=str, required=True)
        parser.add_argument('--checkpoint', type=str, required=True)
        parser.add_argument('--output', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        print(self.args.checkpoint)
        self.precision = torch.float32 if self.args.precision == 'float32' else torch.float16
        self.model = MattingNetwork(self.args.model_variant, decoder=self.args.model_decoder, refiner=self.args.refiner).\
                                    eval().to(self.args.device, self.precision)
        self.model.load_state_dict(torch.load(self.args.checkpoint, map_location=self.args.device)["model_state"], strict=True)
        
    def export(self):
        # downsample_ratio = torch.tensor([0.25]).to(self.args.device)
        # src = torch.randn(1, 3, 640, 720).to(self.args.device, self.precision)
        downsample_ratio = torch.tensor([1]).to(self.args.device)
        sim_ratio = 1
        iw = 192
        ih = 192
        src_size = [1, 3, math.ceil(iw), math.ceil(ih)]
        dec_in = [0, 16, 20, 40, 64]
        if self.args.model_decoder in ["rvm_small", "rvm_sim_small"]:
            dec_in = [0, 8, 12, 16, 64]
        elif self.args.model_decoder in ["gg"]:
            dec_in = [0, 8, 8, 12, 64]
        r1_size = [1, math.ceil(dec_in[1] * sim_ratio), math.ceil(src_size[2] / 2), math.ceil(src_size[3] / 2)]
        r2_size = [1, math.ceil(dec_in[2] * sim_ratio), math.ceil(r1_size[2] / 2), math.ceil(r1_size[3] / 2)]
        r3_size = [1, math.ceil(dec_in[3] * sim_ratio), math.ceil(r2_size[2] / 2), math.ceil(r2_size[3] / 2)]
        r4_size = [1, math.ceil(dec_in[4] * sim_ratio), math.ceil(r3_size[2] / 2), math.ceil(r3_size[3] / 2)]
        src = torch.randn(*src_size).to(self.args.device, self.precision)
        r1i = torch.randn(*r1_size).to(self.args.device, self.precision)
        r2i = torch.randn(*r2_size).to(self.args.device, self.precision)
        r3i = torch.randn(*r3_size).to(self.args.device, self.precision)
        r4i = torch.randn(*r4_size).to(self.args.device, self.precision)
        
        dynamic_spatial =       {0: 'batch_size', 2: 'height', 3: 'width'}
        dynamic_everything =    {0: 'batch_size', 1: 'channels', 2: 'height', 3: 'width'}
        dynamic_axes={
            'src': dynamic_spatial,
            'fgr': dynamic_spatial,
            'res': dynamic_spatial,
            'r1i': dynamic_everything,
            'r2i': dynamic_everything,
            'r3i': dynamic_everything,
            'r4i': dynamic_everything,
            'r1o': dynamic_spatial,
            'r2o': dynamic_spatial,
            'r3o': dynamic_spatial,
            'r4o': dynamic_spatial,
            }
        # fix size
        dynamic_axes={}
        
        print("seg", self.args.seg)
        torch.onnx.export(
            model=self.model,
            args=(src, r1i, r2i ,r3i, r4i, downsample_ratio, self.args.seg),
            f=self.args.output,
            export_params=True,
            verbose=False,
            opset_version=self.args.opset,
            do_constant_folding=True,
            input_names=['src', 'r1i', 'r2i', 'r3i', 'r4i', 'downsample_ratio'],
            output_names=['res', 'r1o', 'r2o', 'r3o', 'r4o'],
            dynamic_axes=dynamic_axes,
            )

if __name__ == '__main__':
    Exporter()
