
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1118/checkpoint/stage3/epoch-27.pth
python export_onnx.py \
    --model-variant mobilenetv3 \
    --seg 1 \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output model_seg.onnx \
    --checkpoint $cp && \
onnxsim model_seg.onnx model_seg_sim.onnx && \
echo "export success"
