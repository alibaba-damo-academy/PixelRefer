#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="VideoRefer-VideoLLaMA3-7B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videorefer_videollama3/evaluation/infer_hc_stvg.py \
    --model-path DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B \
    --video-folder /mnt/workspace/workgroup/yuanyq/datasets/HC-STVG/video_parts \
    --output-file ./eval/hc_stvg/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python videorefer_videollama3/evaluation/captioning/change2eval_format.py --pred-path ./eval/hc_stvg/${MODEL_NAME}.json
python videorefer_videollama3/evaluation/captioning/eval_cococap.py --pred-path ./eval/hc_stvg/${MODEL_NAME}.json