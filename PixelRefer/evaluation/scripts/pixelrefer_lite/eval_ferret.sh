
MODEL_NAME="PixelRefer-Lite-2B"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CUDA_VISIBLE_DEVICES=${GPULIST[0]} python evaluation/pixelrefer_lite/infer_ferret.py \
    --question-file /mnt/workspace/workgroup/yuanyq/code/Osprey/osprey/eval/ferret-bench/box_refer_reason.json \
    --model-path work_dirs/${MODEL_NAME} \
    --output-file ./eval/ferret_bench/${MODEL_NAME}_reason.jsonl


python evaluation/ferret_bench/2.eval_ferret.py \
    --question /mnt/workspace/workgroup/yuanyq/code/ml-ferret/ferret/eval/ferret_gpt4_data/refer_reason/question.jsonl \
    --context /mnt/workspace/workgroup/yuanyq/code/ml-ferret/ferret/eval/ferret_gpt4_data/refer_reason/context.jsonl \
    --answer-list ./eval/ferret_bench/${MODEL_NAME}_reason.jsonl /mnt/workspace/workgroup/yuanyq/code/ml-ferret/ferret/eval/ferret_gpt4_data/refer_reason/answer.jsonl \
    --output ./eval/ferret_bench/${MODEL_NAME}_reason_gpt.jsonl  \
    --rule /mnt/workspace/workgroup/yuanyq/code/ml-ferret/ferret/eval/ferret_gpt4_data/rule.json


python evaluation/ferret_bench/3.summarize.py --files ./eval/ferret_bench/${MODEL_NAME}_reason_gpt.jsonl
