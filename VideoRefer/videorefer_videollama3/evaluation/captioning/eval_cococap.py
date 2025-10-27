from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import pylab
import argparse
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate videorefer-bench-q.")
    parser.add_argument("--pred-path", default=r'', help="The path to file containing prediction.")
    parser.add_argument("--ann", default=None, help="The path to file containing prediction.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    annFile = args.pred_path.replace('.json', '_gt.json')
    resFile = args.pred_path.replace('.json', '_res.json')
    
    if args.ann is not None:
        annFile = args.ann

    # with open(annFile, "r") as f:
    #     gts = json.load(f)
    # with open(resFile, "r") as f:
    #     res = json.load(f)


    coco = COCO(annFile)
    coco_result = coco.loadRes(resFile)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()


    # cocoEval = COCOEvalCap()

    # cocoEval.evaluate(gts,res)

    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')


if __name__ == '__main__':
    main()