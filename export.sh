python export_onnx.py \
    --model-variant mobilenetv3 \
    --seg 1 \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output model_seg.onnx \
    --checkpoint $1 && \
python export_onnx.py \
    --model-variant mobilenetv3 \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output model_mat.onnx \
    --checkpoint $1 && \
echo "export success"
