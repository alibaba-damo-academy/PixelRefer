#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="PixelRefer-2B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/pixelrefer/infer_hc_stvg.py \
    --model-path work_dirs/$MODEL_NAME \
    --video-folder data/HC-STVG/video_parts \
    --output-file ./eval/hc_stvg/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python evaluation/captioning/change2eval_format.py --pred-path ./eval/hc_stvg/${MODEL_NAME}.json
python evaluation/captioning/eval_cococap.py --pred-path ./eval/hc_stvg/${MODEL_NAME}.json