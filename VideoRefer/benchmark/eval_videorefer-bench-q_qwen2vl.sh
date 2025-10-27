output_file=eval_output/videorefer-bench-d/qwen/qwen-videorefer-d.json

python3 benchmark/infer_videorefer_bench_q_qwen2vl.py \
--video-folder eval/VideoRefer-Bench-Q/masked-first-frame \
--question-file eval/VideoRefer-Bench-Q/VideoRefer-Bench-Q.json \
--output-file $output_file 

python videorefer/eval/eval_videorefer_bench_q.py \
    --pred-path $output_file

