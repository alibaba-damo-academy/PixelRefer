# VideoRefer-Bench
VideoRefer-Bench enables an in-depth evaluation of video-based referring conversational models through two types of assessments:

1. Video-based object-level Description Generation
2. Zero-shot object-level question-answer

---

## VideoRefer-Bench-D

The benchmark is designed to evaluate the description generation performance of video-based referring models. The benchmark comprises a total of 400 curated data entries. We curated the test set based on Panda-70M, employing the automatic pipeline, followed by a meticulous human check.

This benchmark covers four key aspects:

1. **Subject Correspondence (SC)**: This dimension evaluates whether the subject of the generated description accurately corresponds to that specified in the ground truth.
2. **Appearance Description (AD)**: This criterion assesses the accuracy of appearance-related details, including color, shape, texture, and other relevant visual attributes.
3. **Temporal Description (TD)**: This aspect analyzes whether the representation of the object’s motion is consistent with the actual movements.
4. **Hallucination Detection (HD)**: This facet identifies discrepancies by determining if the generated description includes any facts, actions, or elements absent from reality, like imaginative interpretations or incorrect inferences.

| Type                   | GPT-4o        | InternVL2-26B | Qwen2-VL-7B | Elysium    | Artemis | VideoRefer-7B        | VideoRefer-VideoLLaMA3-7B  |
| ---------------------- | ------------- | ------------- | ----------- | ---------- | ------- | ----------------- | ----------------- |
| Subject Correspondence | 3.34/4.15     | 3.55/4.08     | 2.97/3.30   | 2.35/-     | -/3.42  | 4.41/**4.44** | **4.63**/- |
| Appearance Description | 2.96/**3.31** | 2.99/3.35     | 2.24/2.54   | 0.30/-     | -/1.34  | 3.27/3.27     | **3.59**/- |
| Temporal Description   | 3.01/**3.11** | 2.57/3.08     | 2.03/2.22   | 0.02/-     | -/1.39  | 3.03/3.10     | **3.38**/- |
| Hallucinaton Detection | 2.50/2.43     | 2.25/2.28     | 2.31/2.12   | **3.59**/- | -/2.90  | 2.97/**3.04** | 3.29/- |
| Average                | 2.95/3.25     | 2.84/3.20     | 2.39/2.55   | 1.57/-     | -/2.26  | 3.42/**3.46** | **3.72**/- |

### Data download
The annotation of VideoRefer-Bench-D can be downloaded [here](https://huggingface.co/datasets/DAMO-NLP-SG/VideoRefer-Bench/blob/main/VideoRefer-Bench-D.json).

Given the vast size of the Panda-70M dataset, downloading it can be quite costly. Therefore, we have provided the video used in the benchmark [here](https://huggingface.co/datasets/DAMO-NLP-SG/VideoRefer-Bench/blob/main/Panda-70M-part.zip).

Data structure:
```bash
VideoRefer
└── eval
    └── VideoRefer-Bench-D
        ├── VideoRefer-Bench-D.json
        └── Panda-70M-part 
```

### Data Format
For each object, we uniformly sampled 32 frames to generate the corresponding mask.

The data format organized in the benchmark json file is as below:

```json
[
    {
        "id": 0,
        "video": "rLlzmcp3J6s_0:01:09.633_0:01:14.333.mp4",
        "caption": "The cub is a smaller, light colored lion. It is lying down and resting its head against the other lion. The cub looks calm and relaxed. It is the lion on the far left side of the frame.",
        "frame_idx": "36",
        "annotation":[
            {
                "2":{
                    "segmentation": {
                    }
                },
                "6":{
                    "segmentation": {
                    }
                },
                ...
            }
        ]
    }
]
```

- `frame_idx`: When using single-frame mask mode, we only use the single mask with the frame_idx.
- All the segmentations are in `RLE` format.

### Evaluation
We use GPT-4o to evaluate this benchmark by assigning scores to the generated predictions on a scale from 0 to 5 across four dimensions.

The evaluation code can be found in [videorefer/eval/videorefer_bench_d](../videorefer/eval/videorefer_bench_d).

> To evaluate other general MLLMs on the VideoRefer-Bench, please refer to [evaluation.md](evaluation_general_mllms.md)

## VideoRefer-Bench-Q
The benchmark is designed to evaluate the proficiency of MLLMs in interpreting video objects, including 1,000 high-quality multiple-choice questions.

The benchmark covers five types of questions:

1. Basic Questions
2. Sequential Questions
3. Relationship Questions
4. Reasoning Questions
5. Future Predictions

| Type                   | GPT-4o   | GPT-4o-mini | InternVL2-26B | Qwen2-VL-7B | VideoRefer-7B | VideoRefer-VideoLLaMA3-7B |
| ---------------------- | -------- | ----------- | ------------- | ----------- | ---------- | ---------- |
| Basic Questions        | 62.3     | 57.6        | 58.5          | 62.0        | 75.4   | **88.1** |
| Sequential Questions   | 74.5     | 67.1        | 63.5          | 69.6        | 68.6   | **77.7** |
| Relationship Questions | 66.0     | 56.5        | 53.4          | 54.9        | 59.3   | **70.24** |
| Reasoning Questions    | 88.0     | 85.9        | 88.0          | 87.3        | **89.4**   | 88.8 |
| Reasoning Questions    | 73.7     | 75.4        | 78.9          | 74.6        | 78.1       | **79.8** |
| Average                | 71.3     | 65.8        | 65.0          | 66.0        | 71.9   | **80.1** |

### Data download
The annotation of VideoRefer-Bench-Q can be downloaded [here]().

The source video in VideoRefer-Bench includes MeViS and Davis.
- MeViS
    - Available at: https://codalab.lisn.upsaclay.fr/competitions/15094
- DAVIS
    - Available at: https://davischallenge.org/davis2017/code.html
    - Please download and unzip `TrainVal`, `Test-Dev` and `Test-Challenge` to the JPEGImages directory.

Data structure:
```bash
VideoRefer
└── eval
    └── VideoRefer-Bench-Q
        ├── VideoRefer-Bench-Q.json
        ├── MeViS 
        |   ├── valid_u/ 
        |   |   └── JPEGImages/      
        └── DAVIS 
            └── JPEGImages/  
                └── 480p/      

```

### Data Format
For each object, we uniformly sampled 32 frames to generate the corresponding mask.

The data format organized in the benchmark json file is as below:

```json
[
    {
        "id": 0,
        "video": "DAVIS/JPEGImages/480p/aerobatics",
        "Question": "What is <object3><region> not wearing?",
        "type": "Basic Questions",
        "options": [
            "(A) A helmet",
            "(B) A hat",
            "(C) Sunglasses",
            "(D) A watch"
        ],
        "Answer": "(A) A helmet",
        "frame_idx": "57",
        "annotation":[
            {
                "0":{
                    "segmentation": {
                    }
                },
                "3":{
                    "segmentation": {
                    }
                },
                ...
            }
        ]
    }
]
```

- `frame_idx`: When using single-frame mask mode, we only use the single mask with the frame_idx.
- All the segmentations are in `RLE` format.

> To evaluate other general MLLMs on the VideoRefer-Bench, please refer to [evluation.md](evaluation_general_mllms.md)
