python3 inference.py \
    --variant mobilenetv3 \
    --checkpoint "/apdcephfs_cq2/share_1630463/portrait_matting/model_offical/rvm_mobilenetv3.pth" \
    --device cuda \
    --input-source "$1" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
    # --input-source "/apdcephfs_cq2/share_1630463/portrait_matting/test/f1.mp4" \
    # --checkpoint "/apdcephfs_cq2/share_1630463/portrait_matting_cache/checkpoint/stage4/epoch-27.pth" \
