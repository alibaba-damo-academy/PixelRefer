import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input-file', required=True)
args = parser.parse_args()

data = json.load(open(args.input_file))
score_all = 0
sum_all = 0

type_list = ['Subject Correspondence', 'Appearance Description', 'Temporal Description', 'Hallucination Detection']
for tp in type_list:
    sum = 0
    score = 0
    for i,d in enumerate(data):
        if tp not in d:
            continue
        sum+=1
        score += d[tp]
    

    print(tp, ': ', score/sum)
    print(sum)
    score_all+=score/sum

print('all....')
print(score_all/4)