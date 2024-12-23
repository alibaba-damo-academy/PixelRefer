# Evaluation for VideoRefer 📊

This document provides instructions on evaluating VideoRefer on video referring tasks and general video understanding tasks.

## 1.VideoRefer-Bench
Please prepare the datasets and annotations used for evaluation, as outlined in [VideoRefer-Bench](../Benchmark.md).
1. VideoRefer-Bench-D
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_videorefer-bench-d.sh
```
Note: Adjust the `--mode` parameter to switch between annotation modes: use `single` for single-frame mode and `multi` for multi-frame mode.

2. VideoRefer-Bench-Q
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_videorefer-bench-q.sh
```
Note: 
- Fill in the `AZURE_API_KEY`, `AZURE_API_ENDPOINT` and `AZURE_API_DEPLOYNAME` in the `eval_videorefer-bench-q.sh` first.
- Adjust the `--mode` parameter to switch between annotation modes: use `single` for single-frame mode and `multi` for multi-frame mode.


## 2.General Video Understanding
We test three benchmarks, MVBench, videomme and perception test.

The evaluation data structure is derived from [VideoLLaMA2](https://github.com/DAMO-NLP-SG/VideoLLaMA2).

```
VideoLLaMA2
├── eval
│   ├── mvbench # Official website: https://huggingface.co/datasets/OpenGVLab/MVBench
|   |   ├── video/
|   |   |   ├── clever/
|   |   |   └── ...
|   |   └── json/
|   |   |   ├── action_antonym.json
|   |   |   └── ...
│   ├── perception_test_mcqa # Official website: https://huggingface.co/datasets/OpenGVLab/MVBench
|   |   ├── videos/ # Available at: https://storage.googleapis.com/dm-perception-test/zip_data/test_videos.zip
|   |   └── mc_question_test.json # Download from https://storage.googleapis.com/dm-perception-test/zip_data/mc_question_test_annotations.zip
│   ├── videomme # Official website: https://video-mme.github.io/home_page.html#leaderboard
|   |   ├── test-00000-of-00001.parquet
|   |   ├── videos/
|   |   └── subtitles/
```

Running command:

```bash
# mvbench evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_video_qa_mvbench.sh
# videomme evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_video_mcqa_videomme.sh
# perception test evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/eval/eval_video_mcqa_perception_test_mcqa.sh
```