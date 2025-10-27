#!/bin/bash
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"

IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_NAME="PixelRefer-2B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python evaluation/pixelrefer/infer_refl4.py \
    --model-path work_dirs/$MODEL_NAME \
    --output-file ./eval/refl4/${MODEL_NAME}_${IDX}.json \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX  &
done

wait


python evaluation/refl4/gpt.py --pred ./eval/refl4/${MODEL_NAME}.json

python evaluation/captioning/change2eval_format.py --pred-path ./eval/refl4/${MODEL_NAME}_reformat.json --pred-key pred_gpt --answer-key Answer
python evaluation/captioning/eval_cococap.py --pred-path ./eval/refl4/${MODEL_NAME}_reformat.json --ann /mnt/workspace/workgroup/yuanyq/code/videollama3/ProjectX_region/eval/refl4/refl4_gt.json
