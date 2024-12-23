import json
import re
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate videorefer-bench-q.")
    parser.add_argument("--pred-path", default=r'', help="The path to file containing prediction.")
    args = parser.parse_args()
    return args


def main():
    all_sum = {}
    right_num = {}

    args = parse_args()
    data = []
    for line in open(args.pred_path):
        d = json.loads(line)
        data.append(d)

    for i,d in enumerate(data):
        gt = d['Answer']
        match = re.search(r'\(([A-Z])\)', gt)
        if match:
            gt = match.group(1) 

        match = re.search(r'\(([A-Z])\)', d['pred'])
        if match:
            answer = match.group(1) 
        else:
            match = re.search(r'([A-Z])\)', d['pred'])
            if match:
                answer = match.group(1) 
            else:
                answer = d['pred'].replace('.','')[0]

        if d['type'] not in all_sum:
            all_sum[d['type']] = 0
            right_num[d['type']] = 0
        if answer.lower()==gt.lower():
            right_num[d['type']]+=1
        # else:
        #     print(gt, ' ', answer)
        all_sum[d['type']]+=1

    all_type_sum = 0
    all_type_right = 0
    for tp in all_sum.keys():
        print('####### ',tp, ' #######')
        print('all num: ',all_sum[tp])
        print('right num: ',right_num[tp])
        print('accuracy: ', right_num[tp]/all_sum[tp])
        all_type_sum+=all_sum[tp]
        all_type_right+=right_num[tp]

    print('####### average #######')
    print('all num: ',all_type_sum)
    print('right num: ',all_type_right)
    print('accuracy: ', all_type_right/all_type_sum)


if __name__ == '__main__':
    main()
