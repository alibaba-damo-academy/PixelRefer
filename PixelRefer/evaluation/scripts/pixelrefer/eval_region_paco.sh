#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="PixelRefer-2B"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/pixelrefer/infer_paco_lvis.py \
    --model-path work_dirs/${MODEL_NAME} \
    --image-folder data/COCO/ \
    --question-file /mnt/damovl/yuanyq/benchmarks/osprey/paco_val_1k_category.json \
    --output-file ./eval/paco/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python evaluation/eval_paco_lvis.py --pred-path ./eval/paco/${MODEL_NAME}.json
