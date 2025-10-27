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
from openai import AzureOpenAI

def init():
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2024-02-15-preview"
    )

    return client

def interaction(client, message_text):
    completion = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYNAME"),
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion


def main(args):
    client = init()

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
    
        messages = [
            {"role": "system", "content":[{"type": "text", "text": system_message}]},
            {"role": "user", "content":[{"type": "text", "text": gt+pred}]}
        ],

        
        for i in range(5):
            try:
                completion = interaction(client, message)
                generate_content = completion.choices[0].message.content
                d['gpt'] = generate_content
                break

            except Exception as e:
                print("error. model generation failed.")

        b = json.dumps(data)
        f2 = open(args.output_file, 'w')
        f2.write(b)
        f2.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-4o")
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument("--api-key", required=True, type=str, help="Azure Openai API key.")
    parser.add_argument("--api-endpoint", required=True, type=str, help="Azure Openai API endpoint.")
    parser.add_argument("--api-deployname", required=True, type=str, help="Azure Openai API deployname.")
    args = parser.parse_args()

    # Set the OpenAI API key.
    os.environ["AZURE_OPENAI_KEY"] = args.api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = args.api_endpoint
    os.environ["AZURE_OPENAI_DEPLOYNAME"] = args.api_deployname

    client = init()

    main(args)
