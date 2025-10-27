import json
from sentence_transformers import SentenceTransformer, util
import argparse
from tqdm import tqdm
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate videorefer-bench-q.")
    parser.add_argument("--pred-path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--bert-model", default='/mnt/workspace/workgroup/yuanyq/checkpoints/all-MiniLM-L6-v2', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args

def process(string):
    string = string.replace('_', ' ').replace(':', ' ').replace(',', ' ').replace('.', ' ').replace('(', ' ').replace(')', ' ')
    string = string.strip().lower()
    return string

def SemanticIOU(value: list[str], target: list[str]) -> None:

    intersection = len(set(value.split()) & set(target.split()))
    union = len(set(value.split()) | set(target.split()))

    return intersection / union


def main():
    args = parse_args()

    bert_model = SentenceTransformer(args.bert_model)
    
    all_sim = 0
    all_num = 0
    all_iou = 0

    data_paths = glob.glob(args.pred_path.replace('.json', '_*.json'))
    data = []
    for data_path in data_paths:
        for line in open(data_path):
            data.append(json.loads(line))
    # data = json.load(open(args.pred_path))

    for d in tqdm(data):
        pred = process(d['pred'])
        answer = process(d['Answer'])
        
        outputs_embeddings = bert_model.encode(pred, convert_to_tensor=True)
        class_sentence_embeddings = bert_model.encode(answer, convert_to_tensor=True)
        cosine_scores = util.cos_sim(outputs_embeddings, class_sentence_embeddings)

        semantic_iou = SemanticIOU(pred, answer)
        d['sim'] = float(cosine_scores[0][0])
        d['siou'] = semantic_iou
        all_sim += cosine_scores[0][0]
        all_iou += semantic_iou
        all_num += 1
        
    print("final sim:{}, semantic iou:{}".format(all_sim/(all_num+1e-10), all_iou/(all_num+1e-10)))
    
    with open(args.pred_path, 'w') as f:
        f.write(json.dumps(data, indent=4))

if __name__ == '__main__':
    main()
