#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="VideoRefer-VideoLLaMA3-7B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videorefer_videollama3/evaluation/infer_refl4.py \
    --model-path DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B \
    --output-file ./eval_output/refl4_detail/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait




# CLAIR
python videorefer_videollama3/evaluation/clair/clair.py --pred ./eval_output/refl4_detail/${MODEL_NAME}.json
python videorefer_videollama3/evaluation/clair/merge_score.py --pred ./eval_output/refl4_detail/${MODEL_NAME}_gpt.json

