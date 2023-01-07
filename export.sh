
# /apdcephfs_cq2/share_1630463/portrait_matting_cache_1129/checkpoint/stage3/epoch-27.pth
cp=$1
python export_onnx.py \
    --model-variant mobilenetv3_small \
    --seg 1 \
    --precision float32 \
    --opset 12 \
    --device cuda \
    --output model_seg.onnx \
    --refiner "fast_guided_filter"  \
    --checkpoint $cp && \
onnxsim model_seg.onnx model_seg_sim1.onnx && \
onnxsim model_seg_sim1.onnx model_seg_sim2.onnx && \
onnxsim model_seg_sim2.onnx model_seg_sim.onnx && \
echo "export success"
