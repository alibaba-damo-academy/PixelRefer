import os
import json
import argparse
from openai import OpenAI
from tqdm.contrib.concurrent import process_map
import glob
def process_data(args):
    d, system_message, api_key, base_url = args
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    if 'gpt' in d or 'pred' not in d:
        return d  # Skip already processed or incomplete pieces

    gt = '##Correct answer: ' + d['Answer'] + '\n'
    pred = '##Predicted answer: ' + d['pred'] + '\n'

    for i in range(5):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-0806",
                messages=[
                    {
                        "role": "system", 
                        "content": system_message
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": gt + pred
                            }
                        ]
                    }
                ],
            )

            json_str = completion.choices[0].message.content
            d['gpt'] = json_str
            break

        except Exception as e:
            print(f"Error: {e}")
    
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)

    args = parser.parse_args()

    api_key = ""
    base_url = ""

    data_list = glob.glob(args.input_file.replace('.json', '_*.json'))
    data_list = [dl for dl in data_list if 'gpt' not in dl]
    data = []
    try:
        for d in data_list:
            data += json.load(open(d))
    except:
        for d in data_list:
            for line in open(d):
                d_ = json.loads(line)
                data.append(d_)
        
    with open('system.txt', 'r') as f:
        system_message = f.read()

    # Preparing arguments for each function call in the process
    map_args = [(d, system_message, api_key, base_url) for d in data]

    # Using process_map to handle parallel processing with a progress bar
    results = process_map(
        process_data,
        map_args,
        max_workers=10
    )

    # Write the processed data back to the output file
    with open(args.output_file, 'w') as f2:
        json.dump(results, f2)


if __name__ == "__main__":
    main()
