import json
from sentence_transformers import SentenceTransformer, util
import argparse
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate videorefer-bench-q.")
    parser.add_argument("--pred-path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--pred-key", default='pred', help="The path to file containing prediction.")
    parser.add_argument("--answer-key", default='Answer', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_paths = glob.glob(args.pred_path.replace('.json', '*.json'))
    data_paths = [d for d in data_paths if 'gt' not in d and 'res' not in d and 'gpt' not in d]
    data = []
    for data_path in data_paths:
        try:
            for line in open(data_path):
                data.append(json.loads(line))
        except:
            data+=json.load(open(data_path))
    # data = json.load(open(args.pred_path))

    annfile = []
    resfile = []
    images = []
    for i,d in enumerate(tqdm(data)):
        images.append({'id': str(i)})
        answer = d[args.answer_key]
        pred = d[args.pred_key]
        annfile.append({'image_id': str(i), 'id': str(i), 'caption': answer})
        resfile.append({'image_id': str(i), 'caption': pred})
        
    ann_file = {}
    ann_file['images'] = images
    ann_file['annotations'] = annfile
    with open(args.pred_path.replace('.json', '_gt.json'), 'w') as f:
        f.write(json.dumps(ann_file, indent=4))
    with open(args.pred_path.replace('.json', '_res.json'), 'w') as f:
        f.write(json.dumps(resfile, indent=4))

if __name__ == '__main__':
    main()
