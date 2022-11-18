python export_onnx.py \
    --model-variant mobilenetv3 \
    --seg 1 \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output model_seg.onnx \
    --checkpoint $1 && \
onnxsim model_seg.onnx model_seg_sim.onnx && \
echo "export success"
