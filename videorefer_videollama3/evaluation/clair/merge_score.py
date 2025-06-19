import argparse
import json
parser = argparse.ArgumentParser(description='Evaluate model outputs')
parser.add_argument('--pred', type=str, help='Path to the prediction JSON file', required=True)
args = parser.parse_args()


data = json.load(open(args.pred))

scores = 0
nums = 0
for d in data:
    scores+=d['score']
    nums+=1

print('score: ', scores/nums)