#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="VideoRefer-VideoLLaMA3-7B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videorefer_videollama3/evaluation/infer_videorefer_q.py \
        --model-path DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B \
        --output-file ./eval/videorefer-bench-q/${MODEL_NAME}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX  &
done

wait



python videorefer_videollama3/evaluation/eval_videorefer_q_bench.py --pred-path ./eval/videorefer-bench-q/${MODEL_NAME}.json