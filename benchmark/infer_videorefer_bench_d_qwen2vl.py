import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

class VideoRefer_Bench_D_general(Dataset):
    def __init__(self, video_folder, data_list, mode):
        self.video_folder = video_folder
        self.data_list = data_list
        self.mode = mode
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        line = self.data_list[idx]
        video_path = os.path.join(self.video_folder, str(idx))
        video_frames = os.listdir(video_path)
        video_frames = [os.path.join(video_path, vf) for vf in video_frames]
        
        if self.mode=='single':
            question = 'I have outlined an object with a red contour in the first frame in the video. Please describe the object in the entire video in detail.'
        else:
            question = 'I have outlined an object with a red contour in the video. Please describe the object in detail.'
            
        return {
            'video': line['video'],
            'video_frames': video_frames,
            'question': question,
            'caption': line['caption'],
        }

def collate_fn(batch):
    video = [x['video'] for x in batch]
    vf = [x['video_frames'] for x in batch]
    qs = [x['question'] for x in batch]
    cap = [x['caption'] for x in batch]
    return video, vf, qs, cap

def build_videorefer_bench_d_eval(args):
    questions = json.load(open(args.question_file))
    dataset = VideoRefer_Bench_D_general(args.video_folder, questions, args.mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return dataloader

def run_inference(args):
    # load model
    # Qwen2VL as example
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    # load processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    val_loader = build_videorefer_bench_d_eval(args)
    
    final_data = []
    ans_file = open(args.output_file, "w")

    
    for i, (videos, video_frames, questions, captions) in enumerate(tqdm(val_loader)):
        video = videos[0]
        video_frame = video_frames[0]
        question = questions[0]
        caption = captions[0]
        
        # generate messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_frame,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
            
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        record = {
            'video': video,
            'caption': caption,
            'pred': output_text[0],
        }
        ans_file.write(json.dumps(record) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--video-folder', help='Directory containing video files.', required=True)
    parser.add_argument('--question-file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output-file', help='Directory to save the model results JSON.', required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mode", type=str, default='single')
    args = parser.parse_args()

    run_inference(args)
