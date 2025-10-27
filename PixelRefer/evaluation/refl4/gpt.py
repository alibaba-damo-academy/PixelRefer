import argparse
import json
import os
import time
import glob
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Pool

NUM_SECONDS_TO_SLEEP = 0.5

client = OpenAI(
    api_key = os.getenv("API_KEY"),
    base_url = os.getenv("BASE_URL"),
)

prompt="""
Give you a description of an object, please condense the sentence into a brief phrase that highlights the object\'s category and key characteristics. 
For example:
INPUT: A freestanding, top-loading washing machine with a white exterior. It features a front-loading door with a recessed handle at the top center for opening and closing. The control panel is located on the upper right side of the machine, consisting of several buttons and possibly a display screen.
OUTPUT: a white washing machine
"""

def get_eval(content: str):
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4-0409",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": content
                            }
                        ]
                    }
                ],
            )
            message = completion.choices[0].message.content
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)
    return message


def evaluate_item(d):
    if len(d['pred'].split(' '))>5:
        pred_new = get_eval(d['pred'])
        d['pred_gpt'] = pred_new
    else:
        d['pred_gpt'] = d['pred']
    # answer_new = get_eval(d['Answer'])
    # d['answer_gpt'] = answer_new
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model outputs')
    parser.add_argument('--pred', type=str, help='Path to the prediction JSON file', required=True)
    args = parser.parse_args()

    pred_files = glob.glob(args.pred.replace('.json', '_*.json'))
    pred_files = [pf for pf in pred_files if 'gt' not in pf and 'res' not in pf]
    data = []
    for pf in pred_files:
        if 'gpt' not in pf and 'reformat' not in pf:
            for line in open(pf):
                data.append(json.loads(line))

    with Pool(10) as pool:
        processed_data = list(tqdm(pool.imap(evaluate_item, data), total=len(data)))

    with open(args.pred.replace('.json', '_reformat.json'), 'w') as f:
        f.write(json.dumps(processed_data, indent=4))
