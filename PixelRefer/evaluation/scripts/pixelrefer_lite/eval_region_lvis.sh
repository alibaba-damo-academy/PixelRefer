#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-3}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="PixelRefer-Lite-2B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/pixelrefer_lite/infer_paco_lvis.py \
    --model-path work_dirs/${MODEL_NAME} \
    --image-folder data/COCO/ \
    --output-file ./eval/lvis/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python evaluation/eval_paco_lvis.py --pred-path ./eval/lvis/${MODEL_NAME}.json