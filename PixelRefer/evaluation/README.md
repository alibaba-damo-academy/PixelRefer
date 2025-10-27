# Evaluation Guidelines

This section provides instructions on evaluating PixelRefer/PixelRefer-Lite on image&video referring tasks.


## 1. Image-level Benchmarks

### Category-level
This task requires the model to output the category or part-level category corresponding to a given region. 

#### LVIS

- Download the [lvis_val_1k_category.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-ValData/resolve/main/lvis_val_1k_category.json?download=true) from Osprey to `benchmark` folder.

```bash
# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_lvis.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_lvis.sh
```

#### PACO
- Download the [paco_val_1k_category.json](https://huggingface.co/datasets/sunshine-lwt/Osprey-ValData/resolve/main/paco_val_1k_category.json?download=true) from Osprey to `benchmarks` folder.
```bash
# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_paco.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_paco.sh
```


### Phrase-level
This task requires the model to generate a short phrase or brief description for each given region.

#### VG
- Download our processed [vg_val_5000.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/vg_val_5000.json?download=true) to `benchmarks` folder.
```bash
# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_vg.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_vg.sh
```

#### Ref-L4
Following DAM, both model predictions and ground-truth captions are first summarized by GPT-4o, and then evaluated using short captioning metrics.
- Download the [Ref-L4](https://huggingface.co/datasets/JierunChen/Ref-L4/) benchmark to the `benchmarks` folder.
- Download our processed question file [ref-l4-val.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/ref-l4-val.json?download=true) and processed GT [refl4_gt.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/refl4_gt.json?download=true) with GPT-4o to `benchmarks/Ref-L4` folder.

```bash
# use GPT-4o to process model predictions 
export API_KEY=xxx
export BASE_URL=xxx

# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_refl4.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_refl4.sh
```

### Detailed Caption
In this setting, the model is expected to generate comprehensive and fine-grained descriptions of each region, going beyond short phrases to capture nuanced attributes and contextual information. 

#### DLC-Bench
- Download [DLC-Bench](https://huggingface.co/datasets/nvidia/DLC-Bench) to the `benchmarks` folder.
```bash
# use GPT-4 to evaluate
export API_KEY=xxx
export BASE_URL=xxx

# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_dlc_bench.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_dlc_bench.sh
```

#### Ref-L4-CLAIR
- Download the [Ref-L4](https://huggingface.co/datasets/JierunChen/Ref-L4/) benchmark to the `benchmarks` folder.
- Download our processed question file [ref-l4-val.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/ref-l4-val.json?download=true) to `benchmarks/Ref-L4` folder.
```bash
# use GPT-4 to evaluate
export API_KEY=xxx
export BASE_URL=xxx

# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_refl4_detail.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_refl4_detail.sh
```

### Question-Answering

#### Ferret-Reasoning
- Download the processed json file [box_refer_reason.json](https://github.com/CircleRadon/Osprey/blob/main/osprey/eval/ferret-bench/box_refer_reason.json) to `benchmarks/ferret-bench` folder.

```bash
# use GPT-4 to evaluate
export API_KEY=xxx
export BASE_URL=xxx

# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_region_refl4_detail.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_region_refl4_detail.sh
```


## 2. Video-level Benchmarks

### VideoRefer-Bench
- Download [VideoRefer-Bench](https://huggingface.co/datasets/DAMO-NLP-SG/VideoRefer-Bench) to `benchmarks` folder.

#### VideoRefer-Bench-D
```bash
# use GPT-4o to evaluate
export API_KEY=xxx
export BASE_URL=xxx

# PixelRefer:
bash evaluation/scripts/pixelrefer/eval_videorefer_d.sh

#PixelRefer-Lite:
bash evaluation/scripts/pixelrefer_lite/eval_videorefer_d.sh
```

#### VideoRefer-Bench-Q
- Download video sources:
    - [MeViS](https://codalab.lisn.upsaclay.fr/competitions/15094)
    - [A2D](https://web.eecs.umich.edu/~jjcorso/r/a2d/index.html#downloads)
    - [Youtube-VOS](https://competitions.codalab.org/competitions/29139#participate-get_data)

1. PixelRefer (We only use one masked frame in the inference):

```bash
bash evaluation/scripts/pixelrefer/eval_videorefer_q.sh
```

2. PixelRefer-Lite
- Download the processed json file [VideoRefer-Bench-Q-allframe.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/VideoRefer-Bench-Q-allframe.json?download=true) to `benchmarks/` folder.
```bash
bash evaluation/scripts/pixelrefer_lite/eval_videorefer_q.sh
```

### HC-STVG

1. PixelRefer (We only use one masked frame in the inference):
- Download the the processed json file [hc-stvg-val.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/hc_stvg_val.json?download=true) where the mask is randomly chosen are extracted from SAM.

```bash
bash evaluation/scripts/pixelrefer/eval_stvg.sh
```

2. PixelRefer-Lite
- Download the the processed json file [hc-stvg-val-allframe.json](https://huggingface.co/datasets/CircleRadon/referring_evaluation/resolve/main/hc-stvg-val-allframe.json?download=true) where all the masks are extracted from SAM.

```bash
bash evaluation/scripts/pixelrefer_lite/eval_stvg.sh
```

