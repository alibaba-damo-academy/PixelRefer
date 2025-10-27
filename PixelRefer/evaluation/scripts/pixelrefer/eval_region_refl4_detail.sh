#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="PixelRefer-2B"



for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/pixelrefer/infer_refl4_detail.py \
    --model-path work_dirs/$MODEL_NAME \
    --output-file ./eval/refl4_detail/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait

cat ./eval/refl4_detail/${MODEL_NAME}_*.json > ./eval/refl4_detail/${MODEL_NAME}_merge.json


# CLAIR
python evaluation/clair/clair.py --pred ./eval/refl4_detail/${MODEL_NAME}.json
python evaluation/clair/merge_score.py --pred ./eval/refl4_detail/${MODEL_NAME}_gpt.json

