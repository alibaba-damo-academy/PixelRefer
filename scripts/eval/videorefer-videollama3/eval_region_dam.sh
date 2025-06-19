#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="VideoRefer-VideoLLaMA3-7B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videorefer_videollama3/evaluation/dam/infer_dam.py \
    --model-path DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B \
    --image-folder DLC-Bench/images \
    --output-file ./eval_output/dam/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python videorefer_videollama3/evaluation/dam/change_format.py --pred-path ./eval_output/dam/${MODEL_NAME}.json

python videorefer_videollama3/evaluation/dam/eval.py --pred ./eval_output/dam/${MODEL_NAME}_pred.json \
 --qa DLC-Bench/qa.json \
 --class-names DLC-Bench/class_names.json \
 --suffix gpt \
 --model $MODEL_NAME 