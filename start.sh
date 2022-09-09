#! /bin/bash
set -x
ChekPointPath="/apdcephfs_cq2/share_1630463/portrait_matting_cache/"
pip3 install -r requirements_training.txt
echo "start.sh" >> ${ChekPointPath}/__shell.log
python3 find_epoch.py
epoch=$?

check_point=""
# 0 1-19 20-21 22
if [ $epoch -eq 0 ]; then
    check_point=""
elif [ $epoch -lt 20 ]; then
    check_point=${ChekPointPath}"checkpoint/stage1/epoch-${epoch}.pth"
elif [ $epoch -lt 22 ]; then
    check_point=${ChekPointPath}"checkpoint/stage2/epoch-${epoch}.pth"
else
    check_point=${ChekPointPath}"checkpoint/stage3/epoch-${epoch}.pth"
fi
echo "checkpoint:" ${check_point}

# 0-19
if [ $epoch -lt 20 ]; then
python3 train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint ${check_point} \
    --checkpoint-dir ${ChekPointPath}/checkpoint/stage1 \
    --log-dir ${ChekPointPath}/log/stage1 \
    --epoch-start 0 \
    --epoch-end 20

    check_point=${ChekPointPath}"checkpoint/stage1/epoch-19.pth"
fi

# 20-21
if [ $epoch -lt 22 ]; then
python3 train.py \
    --model-variant mobilenetv3 \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 50 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint ${check_point} \
    --checkpoint-dir ${ChekPointPath}checkpoint/stage2 \
    --log-dir ${ChekPointPath}/log/stage2 \
    --epoch-start 20 \
    --epoch-end 22

    check_point=${ChekPointPath}"checkpoint/stage2/epoch-21.pth"
fi

# 22
if [ $epoch -lt 23 ]; then
    end_log=${ChekPointPath}/__shell_end.log
    if [ ! -d $end_log ]; then
    python3 train.py \
        --model-variant mobilenetv3 \
        --dataset videomatte \
        --train-hr \
        --resolution-lr 512 \
        --resolution-hr 2048 \
        --seq-length-lr 40 \
        --seq-length-hr 6 \
        --learning-rate-backbone 0.00001 \
        --learning-rate-aspp 0.00001 \
        --learning-rate-decoder 0.00001 \
        --learning-rate-refiner 0.0002 \
        --checkpoint ${check_point} \
        --checkpoint-dir ${ChekPointPath}checkpoint/stage3 \
        --log-dir ${ChekPointPath}/log/stage3 \
        --epoch-start 22 \
        --epoch-end 23 && \
        echo "$?" >> ${end_log}
    fi
    check_point=${ChekPointPath}"checkpoint/stage3/epoch-22.pth"
fi
