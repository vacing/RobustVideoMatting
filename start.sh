#! /bin/bash
set -x
ChekPointPath="/apdcephfs_cq2/share_1630463/portrait_matting_cache/"
pip3 install -r requirements_training.txt
echo "start.sh" >> ${ChekPointPath}/__shell_start.log
end_log=${ChekPointPath}/__shell_end.log

# find epoch
python3 find_next_epoch.py
next_epoch=$?
last_epoch=$((next_epoch-1))

check_point=""
# 0 1-19 20-21 22
if [ $last_epoch -le 0 ]; then
    check_point=""
elif [ $last_epoch -lt 20 ]; then
    check_point=${ChekPointPath}"checkpoint/stage1/epoch-${last_epoch}.pth"
elif [ $last_epoch -lt 22 ]; then
    check_point=${ChekPointPath}"checkpoint/stage2/epoch-${last_epoch}.pth"
elif [ $last_epoch -lt 23 ]; then
    check_point=${ChekPointPath}"checkpoint/stage3/epoch-${last_epoch}.pth"
else
    check_point=${ChekPointPath}"checkpoint/stage4/epoch-${last_epoch}.pth"
fi
echo "checkpoint:" ${check_point}

# 0-19
if [ $next_epoch -lt 20 ]; then
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
    --epoch-start $next_epoch \
    --epoch-end 20
    if [ $? -ne 0 ]; then
        echo "stage 1 train error"
        exit -1
    fi

    next_epoch=20
    check_point=${ChekPointPath}"checkpoint/stage1/epoch-19.pth"
fi

# 20-21
if [ $next_epoch -lt 22 ]; then
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
    --epoch-start $next_epoch \
    --epoch-end 22
    if [ $? -ne 0 ]; then
        echo "stage 2 train error"
        exit -1
    fi

    next_epoch=22
    check_point=${ChekPointPath}"checkpoint/stage2/epoch-21.pth"
fi

# 22
if [ $next_epoch -lt 23 ]; then
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
        --epoch-start $next_epoch \
        --epoch-end 23
    if [ $? -ne 0 ]; then
        echo "stage 3 train error"
        exit -1
    fi
    next_epoch=23
    check_point=${ChekPointPath}"checkpoint/stage3/epoch-22.pth"
fi

# 23-27
if [ $next_epoch -lt 24 ]; then
    if [ ! -d $end_log ]; then
        python3 train.py \
            --model-variant mobilenetv3 \
            --dataset imagematte \
            --train-hr \
            --resolution-lr 512 \
            --resolution-hr 2048 \
            --seq-length-lr 40 \
            --seq-length-hr 6 \
            --learning-rate-backbone 0.00001 \
            --learning-rate-aspp 0.00001 \
            --learning-rate-decoder 0.00005 \
            --learning-rate-refiner 0.0002 \
            --checkpoint ${check_point} \
            --checkpoint-dir ${ChekPointPath}checkpoint/stage4 \
            --log-dir ${ChekPointPath}log/stage4 \
            --epoch-start $next_epoch \
            --epoch-end 28
        if [ $? -ne 0 ]; then
            echo "stage 4 train error"
            exit -1
        fi
    fi
    echo "$?" >> ${end_log}
fi