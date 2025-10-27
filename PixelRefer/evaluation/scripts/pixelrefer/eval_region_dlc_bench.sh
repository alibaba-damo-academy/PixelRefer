#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="PixelRefer-2B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/pixelrefer/dam/infer_dam.py \
    --model-path work_dirs/$MODEL_NAME \
    --image-folder benchmarks/DLC-Bench/images \
    --output-file ./eval/dam/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python evaluation/dam/change_format.py --pred-path ./eval/dam/${MODEL_NAME}.json

python evaluation/dam/eval.py --pred ./eval/dam/${MODEL_NAME}_pred.json \
 --qa benchmarks/DLC-Bench/qa.json \
 --class-names benchmarks/DLC-Bench/class_names.json \
 --suffix gpt \
 --model $MODEL_NAME 