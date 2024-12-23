import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input-file', required=True)
args = parser.parse_args()

data = json.load(open(args.input_file))
final_data = []
err = 0
for d in data:
    try:
        input_string = d['gpt']
        pattern = r'\d+\.\s+(.*?):\s+([\d.]+)'
        matches = re.findall(pattern, input_string)

        result_dict = {description: float(score) for description, score in matches}
        final_data.append(dict(d, **result_dict))
    except:
        err+=1
    
if err>0:
    print('####error num: ',err)
    
with open(args.input_file, 'w') as f:
    f.write(json.dumps(final_data))
