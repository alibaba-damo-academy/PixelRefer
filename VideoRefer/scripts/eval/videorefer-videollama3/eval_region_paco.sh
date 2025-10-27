#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="VideoRefer-VideoLLaMA3-7B"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videorefer_videollama3/evaluation/infer_paco_lvis.py \
    --model-path DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B \
    --image-folder COCO/train2017 \
    --question-file /mnt/damovl/yuanyq/benchmarks/osprey/paco_val_1k_category.json \
    --output-file ./eval_output/paco/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python videorefer_videollama3/evaluation/eval_paco_lvis.py --pred-path ./eval_output/paco/${MODEL_NAME}.json
