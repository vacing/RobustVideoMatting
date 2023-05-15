
cp=$1
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1207/checkpoint/stage3/epoch-27.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1129/checkpoint/stage3/epoch-27.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1118/checkpoint/stage3/epoch-27.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1202/checkpoint/stage3/epoch-27.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1129/checkpoint/stage3/epoch-27.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1228v2/checkpoint/stage3/epoch-40.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_1229/checkpoint/stage3/epoch-50.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_test/checkpoint/stage1/epoch-0.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache/checkpoint/stage1/epoch-15.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230115_smallPrvm/checkpoint/stage3/epoch-42.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230115_smallPgg/checkpoint/stage4/epoch-43.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230201_smallPgg/checkpoint/stage4/epoch-43.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230209_smallerPgg/checkpoint/stage3/epoch-42.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230220_smallerPgg/checkpoint/stage3/epoch-42.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230305_smallerPsim/checkpoint/stage4/epoch-38.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230309_smallerPTsim/checkpoint/stage4/epoch-45.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230312_smallerPTgg/checkpoint/stage4/epoch-45.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230316_smallerPTsim/checkpoint/stage3/epoch-67.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230317_smallerSTsim/checkpoint/stage4/epoch-73.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230324_smallerPTsim/checkpoint/stage4/epoch-123.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230327_simPTsim/checkpoint/stage4/epoch-151.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230407_smallerNPTsimN192/checkpoint/stage4/epoch-127.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230403_smallerPTsimN/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230411_smallerPTsimN192/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230411_smallerNPTsimN192/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230412_smallerPTsimN192/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230418_smallerPTsimN192/checkpoint/stage3/epoch-117.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230417_smallerPTsimN192/checkpoint/stage3/epoch-117.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230419_smallerPTsimN192/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230421_smallerPTsimN192/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230423_smallerPTsimN192/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230423_smallerPTsimN192_norcrop/checkpoint/stage4/epoch-123.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230425_smallerPTsimN192/checkpoint/stage4/epoch-123.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230427_bigPrvm192_norcrop/checkpoint/stage4/epoch-63.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230426_smallerPTsimN192_norcrop/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_test/checkpoint/stage1/epoch-0.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230428_smallerPTsimG192_norcrop/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230430_smallerPTsimBG192_norcrop/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230502_smallerPTsimG192_norcrop/checkpoint/stage4/epoch-128.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230504_smallerPTrvmsmall192_norcrop/checkpoint/stage4/epoch-127.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230504_smallerPTsimBG192_norcrop/checkpoint/stage4/epoch-127.pth
cp=/apdcephfs_cq2/share_1630463/portrait_matting_cache_230506_smallerPTsimG192_norcrop/checkpoint/stage4/epoch-127.pth
backbone=mobilenetv3_sim
backbone=mobilenetv3
backbone=mobilenetv3_smaller
decoder=gg
decoder=rvm
decoder=rvm_small
decoder=rvm_sim_small
python export_onnx.py \
    --model-variant ${backbone}\
    --model-decoder ${decoder} \
    --seg 1 \
    --precision float32 \
    --opset 12 \
    --device cpu \
    --output model_seg.onnx \
    --refiner "fast_guided_filter"  \
    --checkpoint $cp && \
onnxsim model_seg.onnx model_seg_sim1.onnx && \
onnxsim model_seg_sim1.onnx model_seg_sim2.onnx && \
onnxsim model_seg_sim2.onnx model_seg_sim.onnx && \
echo "export success"
