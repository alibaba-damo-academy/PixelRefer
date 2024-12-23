# !export API_KEY=api_key
# !pip install google-generativeai
import os
from tqdm import tqdm
import time
import requests
import PIL.Image
import json
import base64
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input-file', required=True)
parser.add_argument('--output-file', required=True)

args = parser.parse_args()

GPT4V_KEY = "FOOLWVyTxtww5kzZ9Zqa9ABqaPWk2iWYdnKs6sqy48KMO43I3xl2JQQJ99AKACYeBjFXJ3w3AAABACOGeUuC"
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}

GPT4V_ENDPOINT = "https://nlpla-m3ed7fwy-eastus.openai.azure.com/openai/deployments/gpt-4o-0513/chat/completions?api-version=2024-02-15-preview"


data = []
for line in open(args.input_file):
    d = json.loads(line)
    data.append(d)

with open('videorefer/eval/videorefer_bench_d/system.txt', 'r') as f:
    system_message = f.read()

for d in tqdm(data):
    if 'pred' not in d:
        continue

    gt = '##Correct answer: '+d['caption'] + '\n'
    pred = '##Predicted answer: '+d['pred'] +'\n'
    payload = {
        "messages": [
            {"role": "system", "content":[{"type": "text", "text": system_message}]},
            {"role": "user", "content":[{"type": "text", "text": gt+pred}]}
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        # "max_tokens": 800
    }
    
    for i in range(5):
        try:
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            generate_content = response.json()['choices'][0]['message']['content']
            d['gpt'] = generate_content
            break

        except Exception as e:
            print("error. model generation failed.")

b = json.dumps(data)
f2 = open(args.output_file, 'w')
f2.write(b)
f2.close()