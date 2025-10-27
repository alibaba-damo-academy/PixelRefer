MODEL_NAME="VideoRefer-VideoLLaMA3-7B"
gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python videorefer_videollama3/evaluation/infer_videorefer_d.py \
        --model-path DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B \
        --output-file ./eval_output/videorefer-bench-d/${MODEL_NAME}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX  &

done
wait


cd videorefer_videollama3/evaluation/videorefer_bench_d
python 1.eval_gpt.py --input-file ./eval_output/videorefer-bench-d/${MODEL_NAME}_gpt.json
python 2.extract_re.py --input-file ./eval_output/videorefer-bench-d/${MODEL_NAME}_gpt.json
python 3.analyze_score.py --input-file ./eval_output/videorefer-bench-d/${MODEL_NAME}_gpt.json


