import json
import logging
import re
import sys
import glob
from typing import List, Optional, Tuple
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url=YOUR_URL,
)

_CLAIR_PROMPT = """\
You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.
Candidate set:
{candidate_statements}
Reference set:
{target_statements}
On a precise scale from 0 to 100, how likely is it that the candidate set is \
describing the same image as the reference set? (JSON format, with a key "score", \
value between 0 and 100, and a key "reason" with a string value.)
"""

def clair(
    candidates: List[str],
    targets: List[str],
    model: str = "chat-gpt",
    max_tokens: int = 1024,
) -> Tuple[float, Optional[str]]:
    # Compute the CLAIR score for a list of candidates and targets.

    # Format the candidates and targets
    candidate_statements = [f"- {c}\n" for c in candidates]
    target_statements = [f"- {t}\n" for t in targets]
    formatted_prompt = _CLAIR_PROMPT.format(
        candidate_statements="".join(candidate_statements),
        target_statements="".join(target_statements),
    )

    temperature, score, reason = 0.0, None, None
    response = ""  # Initialize response to empty string
    for _ in range(5):
        try:
            # Run the model
            completion = client.chat.completions.create(
                model="gpt-4-0409",
                messages=[{"role": "user", "content": formatted_prompt}],
            )
            response = completion.choices[0].message.content
            
            # Parse the first JSON object in the response
            parsed = response.split("{")[1]
            parsed = "{" + parsed.split("}")[0] + "}"
            data = json.loads(parsed)
            score = float(data["score"])
            reason = data.get("reason", 'Unknown')
            break
        except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
            # Attempt regex parsing if JSON parsing fails
            logging.error(f"Error parsing response: {str(e)}, response: {response}")
            parsed = re.findall(r"\d*\.?\d+", response)
            if len(parsed) > 0:
                score = float(parsed[0])
                if score < 1:
                    score *= 100
                reason_matches = re.findall(r"(?i)reason.*", response)
                reason = reason_matches[0].strip()[len('reason'):].replace(':', '').strip() if len(reason_matches) > 0 else 'Unknown'
                break
            else:
                continue
    else:
        logging.error("Could not parse response from CLAIR after 5 tries. Setting score to 0.")
        score = 0.0
        reason = 'Parsing failure'

    return score / 100, reason

def process_sample(sample):
    try:
        # logging.info(f"Processing sample: {sample}")
        score, reason = clair([sample['pred']], sample['Answer'], model='chat-gpt', max_tokens=128)
        sample['score'] = score
        sample['reason'] = reason
        # logging.info(f"Completed sample processing: {sample}")
        return sample
    except Exception as e:
        logging.error(f"Error processing sample: {sample}, Exception: {e}")
        sample['score'] = 0.0
        sample['reason'] = f"Error: {str(e)}"
        return sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model outputs')
    parser.add_argument('--pred', type=str, help='Path to the prediction JSON file', required=True)
    args = parser.parse_args()

    # logging.basicConfig(level=logging.INFO)

    pred_files = glob.glob(args.pred.replace('.json', '_*.json'))
    data = []
    for pf in pred_files:
        if 'gpt' not in pf:
            for line in open(pf):
                data.append(json.loads(line))

    # Using ThreadPoolExecutor for multithreading
    processed_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_sample, sample): sample for sample in data}
        for future in tqdm(as_completed(futures), total=len(data)):
            processed_data.append(future.result())

    with open(args.pred.replace('.json', '_gpt.json'), 'w') as f:
        f.write(json.dumps(processed_data, indent=4))
