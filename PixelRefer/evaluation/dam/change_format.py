import json
from sentence_transformers import SentenceTransformer, util
import argparse
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DLC-Bench.")
    parser.add_argument("--pred-path", default=r'', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    data_paths = glob.glob(args.pred_path.replace('.json', '_*.json'))
    data_paths = [d for d in data_paths if 'pred' not in d]
    data = []
    for data_path in data_paths:
        for line in open(data_path):
            data.append(json.loads(line))
    # data = json.load(open(args.pred_path))

    outfile = {}
    for i,d in enumerate(tqdm(data)):
        outfile[d['id']] = d['pred']
    outfile['query'] = 'Please describe <region> in detail.'


    with open(args.pred_path.replace('.json', '_pred.json'), 'w') as f:
        f.write(json.dumps(outfile, indent=4))

if __name__ == '__main__':
    main()
