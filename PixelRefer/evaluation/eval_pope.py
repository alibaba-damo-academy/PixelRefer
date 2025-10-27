
import os
import json
import argparse

def eval_pope(answers):
    label_list = [label['Answer'] for label in answers]

    for answer in answers:
        text = answer['pred']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['pred'] = 'no'
        else:
            answer['pred'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['pred'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    #把acc写到json
    result = {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Yes_ratio': yes_ratio
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    args = parser.parse_args()
    results = {}
    results['overall'] = 0
    data = []
    for line in open(args.question_file):
        data_ = json.loads(line)
        data.append(data_)
    # data = json.load(open(args.question_file))
    for category in ['adversarial', 'popular', 'random']:
        cur_data = [x for x in data if x['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_data)))
        res = eval_pope(cur_data)
        print("====================================")
        results[category] = res
        results['overall']+=res['Accuracy']
    results['overall'] /= 3    
    with open(args.result_file, 'w') as f:
        json.dump(results, f, indent=4)
