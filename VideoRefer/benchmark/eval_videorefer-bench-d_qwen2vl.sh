output_file=eval_output/videorefer-bench-d/qwen/qwen-videorefer-d.json

python3 benchmark/infer_videorefer_bench_d_qwen2vl.py \
--video-folder eval/VideoRefer-Bench-D/masked-first-frame \
--question-file eval/VideoRefer-Bench-D/VideoRefer-Bench-D.json \
--output-file $output_file \
--mode single \

gpt_output_file=eval_output/videorefer-bench-d/qwen/qwen-videorefer-d-gpt.json

python3 videorefer/eval/videorefer_bench_d/1.eval_gpt.py \
    --input-file $output_file \
    --output-file $gpt_output_file

python3 videorefer/eval/videorefer_bench_d/2.extract_re.py \
    --input-file $gpt_output_file

python3 videorefer/eval/videorefer_bench_d/3.analyze_score.py \
    --input-file $gpt_output_file