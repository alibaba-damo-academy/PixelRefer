# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
from tqdm import tqdm
import argparse
import inflect
import requests

prompt_eval = """Answer the multiple-choice question based on the text description of an object in an image. You need to follow these rules:
1. Do not output any reasoning. Do not perform correction. Please output exactly one answer from the choices for each question. Do not repeat the question.
2. There is no need for exact matching. Please choose the closest option based on the description.

The description is:
{pred_caption}

From the description above, please answer the following question with one of the choices:
{question_text_str}
"""

from openai import OpenAI

client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url=YOUR_URL,
)

#ideatalk
def query(prompt, temperature, max_tokens, model):
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4-0409",
                messages=[{"role": "user", "content": prompt}],
            )

            message = completion.choices[0].message.content
            break
        except Exception as exp:
            print(exp)
    return message

def parse_pred(pred, choices, key):
    pred = pred.strip().lower()
    substr_indices = []
    for index, choice in enumerate(choices):
        choice = choice.strip().lower()
        prefix = "abcde"[index]
        if choice == pred or pred == f"{prefix}. {choice}" or pred == prefix:
            return index
        if choice in pred:
            substr_indices.append((index, pred.index(choice), len(choice)))

    # Only one match (choice in prediction)
    if len(substr_indices) == 1:
        return substr_indices[0][0]

    # Prefix match
    choices_label = "abcde"
    if pred[0] in choices_label and pred[1] == ".":
        ret = choices_label.index(pred[0])
        # print(f"{key}: Chosen {ret} for pred: {pred}, choices: {choices}")
        # print(f"{key}: More than one occurrence found or no substr of choice in pred: pred {pred}, choices {choices}, substr indices: {substr_indices}, returning {ret} (choice {choices_label})")
        return ret

    # More than one match
    if substr_indices:
        # Return the last occurrence if there are multiple matches (referenced from MMMU): https://github.com/MMMU-Benchmark/MMMU/blob/b119c944a15c145c10d52a58e841c5b9cb6a535e/eval/utils/eval_utils.py#L57
        if len(substr_indices) > 1:
            ret, ret_pos, _ = max(substr_indices, key=lambda x: x[1])
            max_items = [item for item in substr_indices if item[1] == ret_pos]
            if len(max_items) > 1:
                # select the item with the longest match if there are multiple occurrence at the same place
                ret = max(max_items, key=lambda x: x[2])[0]
            print(f"{key}: More than one occurrence found: pred {pred}, choices {choices}, {substr_indices}, returning {ret} (choice {choices_label})")
        else:
            ret = substr_indices[0][0]
        return ret

    # Parse the case where pred is a substr of choice
    match_lengths = []
    for index, choice in enumerate(choices):
        choice = choice.strip().lower()
        if pred in choice:
            match_lengths.append((index, len(choice)))
    if match_lengths:
        # Return the longest matched substring if there are multiple matches
        if len(match_lengths) > 1:
            ret = max(match_lengths, key=lambda x: x[1])[0]
            print(f"{key}: More than one occurrence found: pred {pred}, choices {choices}, {match_lengths}, returning {ret}")
        else:
            ret = match_lengths[0][0]
        return ret
    
    if pred[0] in "abcde" and (len(pred.strip()) == 1 or pred[1] == "\n"):
        ret = "abcde".index(pred[0])
        print(f"{key}: Chosen {ret} for pred: {pred}, choices: {choices}")
        return ret

    print(f"*WARNING*: {key}: No match found. Pred: {pred}, choices: {choices}")

    # If no matching choice is found, raise an error.
    # raise ValueError(f"No match found. Pred: {pred}, Choices: {choices}")
    # If no matching choice is found, return None (treat as no mention, score 0).
    return None

def evaluate(question_dicts, pred_caption, temperature, max_tokens, model, *, response_override=None, key, verbose=False) -> dict:
    pred_answers = []
    prompt = []
    response = []
    for index, question_dict in enumerate(question_dicts):
        question_text_str = f"{question_dict['question']}\n"
        choices_text = ""
        for choice_index, (choice, score) in enumerate(question_dict['choices']):
            choice_index = 'ABCDE'[choice_index]
            choices_text += f"{choice_index}. {choice}\n"
        question_text_str += choices_text
        prompt_item = prompt_eval.format(pred_caption=pred_caption, question_text_str=question_text_str.strip())
        
        if response_override is None or len(response_override) < index or response_override[index] is None:
            response_item = query(prompt_item, temperature, max_tokens, model)
            # print(f"Prompt:\n{prompt_item}")
            # print(f"Output: {response_item}")
        else:
            response_item = response_override[index]

        pred_answer = response_item.strip()
        pred_answers.append(pred_answer)
        prompt.append(prompt_item)
        response.append(response_item)
    
    assert len(pred_answers) == len(question_dicts), f"Length mismatch for key {key} question {index}: pred: {len(pred_answers)} vs question: {len(question_dicts)}"
    pred_indices = [parse_pred(pred_answer, [choice for choice, score in question_dict['choices']], key) for pred_answer, question_dict in zip(pred_answers, question_dicts)]

    assert len(pred_indices) == len(question_dicts), f"Length mismatch for key {key} question {index}: pred: {len(pred_indices)} vs question: {len(question_dicts)}"
    
    # If no matching, treat as no mention.
    try:
        parsed_eval_results = [question_dict['choices'][pred_index][1] if pred_index is not None else 0 for pred_index, question_dict in zip(pred_indices, question_dicts)]
    except IndexError as e:
        print(f"Error: {e}, key: {key}, pred_indices: {pred_indices}, question_dicts: {question_dicts}")
        raise e

    parsed_eval_results_positives = []
    parsed_eval_results_negatives = []

    details_positives = []
    details_negatives = []
    details_recognition = []
    recognition_result = None
    for question_index, (parsed_eval_result, question_dict) in enumerate(zip(parsed_eval_results, question_dicts)):
        if question_dict['type'] == 'recognition':
            # If the type is recognition, it's the recognition question.
            if parsed_eval_result == "correct":
                recognition_result = True
            elif parsed_eval_result == "incorrect":
                recognition_result = False
                print(f"Recognition is incorrect for key {key}, setting score to at most 0 for all questions")
            else:
                raise ValueError(f"Invalid recognition result: {parsed_eval_result}")
            details_recognition.append({
                **question_dict,
                'pred_answer': pred_answers[question_index],
                'pred_index': pred_indices[question_index],
                'eval_result': parsed_eval_result,
            })
        elif question_dict['type'] == 'negative':
            assert recognition_result is not None, f"Negative questions come before recognition question in {key}, {question_dicts}"
            if recognition_result is False:
                if verbose:
                    print(f"Processing negative question {question_index} for key {key}, setting score to at most 0 since recognition is incorrect")
                parsed_eval_result = min(0, parsed_eval_result)
            # If the type is negative, it's one of the negatives.
            parsed_eval_results_negatives.append(parsed_eval_result)
            details_negatives.append({
               **question_dict,
               'pred_answer': pred_answers[question_index],
               'pred_index': pred_indices[question_index],
               # Subtract 1 to get the index in the original question list (excluding the recognition question)
               'question_index': question_index - 1,
               'eval_result': parsed_eval_result,
            })
        elif question_dict['type'] == 'positive':
            assert recognition_result is not None, f"Positive questions come before recognition question in {key}, {question_dicts}"
            if recognition_result is False:
                if verbose:
                    print(f"Processing positive question {question_index} for key {key}, setting score to at most 0 since recognition is incorrect")
                parsed_eval_result = min(0, parsed_eval_result)
            parsed_eval_results_positives.append(parsed_eval_result)
            details_positives.append({
               **question_dict,
               'pred_answer': pred_answers[question_index],
               'pred_index': pred_indices[question_index],
               # Subtract 1 to get the index in the original question list (excluding the recognition question)
               'question_index': question_index - 1,
               'eval_result': parsed_eval_result,
            })
        else:
            raise ValueError(f"Invalid question type: {question_dict['type']}")

    score_pos = sum(parsed_eval_results_positives) / len(parsed_eval_results_positives)
    # It's possible that we don't have negatives for an instance. For this case, we skip over the instance for negative score calculation.
    if len(parsed_eval_results_negatives):
        score_neg = sum(parsed_eval_results_negatives) / len(parsed_eval_results_negatives)
    else:
        score_neg = None

    # Overall score is the average of the positive and negative scores
    info = dict(
        details_positives=details_positives,
        details_negatives=details_negatives,
        details_recognition=details_recognition,
        prompt=prompt,
        response=response,
        score=(sum(parsed_eval_results_positives) + sum(parsed_eval_results_negatives)) / (len(parsed_eval_results_positives) + len(parsed_eval_results_negatives)),
        score_pos=score_pos,
        score_neg=score_neg,
        neg_valid_num=len(parsed_eval_results_negatives),
        recognition_result=recognition_result,
    )

    return info

def is_plural(string):
    # A case that the inflect library does not handle
    if string == "bus":
        return False
    # singular_noun returns False if the word is already singular (otherwise it returns the singular form)
    return p.singular_noun(string) is not False

if __name__ == '__main__':
    # Example:
    # python eval_model_outputs.py --pred model_outputs_cache/dam_3b_v1.json --base-url "http://localhost:9100/v1"

    parser = argparse.ArgumentParser(description='Evaluate model outputs')
    parser.add_argument('--pred', type=str, help='Path to the prediction JSON file', required=True)
    parser.add_argument('--qa', type=str, help='Path to the reference QA file', default="DLC-bench/qa.json")
    parser.add_argument('--class-names', type=str, help='Path to the class names JSON file', default="DLC-bench/class_names.json")
    parser.add_argument('--default-prediction', type=str, default=None, help='Default prediction if key is not present in the prediction file')
    parser.add_argument('--api-call-limit', type=int, default=1000, help='API call limit')
    parser.add_argument('--api-key', type=str, default=None, help='Path to the OpenAI API key file')
    parser.add_argument('--suffix', type=str, default="", help='Suffix for the evaluation file')
    parser.add_argument('--model', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help='Model name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--quiet', action='store_true', help='Enable quiet mode (result only)')
    parser.add_argument('--csv', action='store_true', help='Output results as CSV only')
    args = parser.parse_args()

    always_print = print
    if args.quiet:
        print = lambda *args, **kwargs: None

    # v3 is from v2.1
    eval_file = os.path.splitext(args.pred)[0] + f"_eval{args.suffix}.json"
    eval_results = {}

    if False:
        assert not os.path.exists(eval_file), f"Evaluation file exists at {eval_file}"
    else:
        if os.path.exists(eval_file):
            print(f"Loading existing evaluation data from {eval_file}")
            try:
                with open(eval_file) as f:
                    eval_results = json.load(f)
            except Exception as e:
                always_print(f"Error loading evaluation data {eval_file}: {e}")
                raise e

    with open(args.pred) as f:
        data_pred = json.load(f)

    with open(args.qa) as f:
        data_qa = json.load(f)

    with open(args.class_names) as f:
        data_class_names = json.load(f)

    scores = {}
    scores_pos = {}
    scores_neg = {}

    keys = list(data_qa.keys())
   
    p = inflect.engine() 
    
    print(f"Using model {args.model}")

    missing_key_count = 0
    for key in tqdm(keys, disable=args.quiet):
        key = str(key)
        if key in eval_results:
            if args.verbose:
                print(f"Skipping {key}")
            response_override = eval_results[key]['response']
        else:
            response_override = None

        if key not in data_pred:
            print(key)
            if args.default_prediction is None:
                raise ValueError(f"Key {key} not found in prediction data, and no default prediction provided")
            else:
                print(f"Key {key} not found in prediction data, using default prediction {args.default_prediction}")
                pred_value = args.default_prediction
                missing_key_count += 1
        elif data_pred[key].startswith("Error:"):
            if args.default_prediction is None:
                raise ValueError(f"Key {key} has an error in prediction data, and no default prediction provided: {data_pred[key]}")
            else:
                print(f"Key {key} has an error in prediction: {data_pred[key]}, using default prediction {args.default_prediction}")
                pred_value = args.default_prediction
                missing_key_count += 1
        else:
            pred_value = data_pred[key]
        
        # print(f"Evaluating {key}")
        class_name = data_class_names[key]
        
        if is_plural(class_name):
            recognition_question = f"Is it likely that the objects in the description are {class_name} or objects of a similar type? Again, It does not have to be an exact match."
        else:
            recognition_question = f"Is it likely that the object in the description is {p.a(class_name)} or an object of a similar type? Again, It does not have to be an exact match."
        recognition_question_dict = {
            "question": recognition_question,
            "choices": [("Yes", "correct"), ("No", "incorrect")],
            "type": "recognition",
        }
        
        # Add the recognition question to the beginning of the list
        question_dicts = [recognition_question_dict, *data_qa[key]]
        info = evaluate(question_dicts=question_dicts, pred_caption=pred_value, model=args.model, temperature=0., max_tokens=300, response_override=response_override, key=key, verbose=args.verbose)
        score = info["score"]
        scores[key] = score
        scores_pos[key] = info["score_pos"]
        scores_neg[key] = info["score_neg"]
        eval_results[key] = {
            "pred": pred_value,
            **info
        }

        if args.verbose:
            print(f"Score: {score}")

        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=4)

    avg_score_pos = sum(scores_pos.values()) / len(scores_pos)
    scores_neg_valid_only = [item for item in scores_neg.values() if item is not None]
    avg_score_neg = sum(scores_neg_valid_only) / len(scores_neg_valid_only)

    if args.csv:
        # Print comma-separated values directly to stdout
        always_print(f"{avg_score_pos:.3f},{avg_score_neg:.3f},{(avg_score_pos + avg_score_neg) / 2:.3f}")
    else:
        always_print(f"Result for {args.pred}")
        always_print(f"Average Positive Score: {avg_score_pos:.3f}")
        always_print(f"Average Negative Score: {avg_score_neg:.3f}")
        always_print(f"Average of Positive and Negative Scores: {(avg_score_pos + avg_score_neg) / 2:.3f}")
        always_print(f"Summary (Pos\tNeg\tAvg(Pos, Neg)):\t{avg_score_pos:.3f},\t{avg_score_neg:.3f},\t{(avg_score_pos + avg_score_neg) / 2:.3f}")
        print(f"QA Scores: {scores}")

        if missing_key_count:
            print(f"Note: Missing {missing_key_count} keys, using default prediction {args.default_prediction}")

    eval_results["avg_pos"] = avg_score_pos
    eval_results["avg_neg"] = avg_score_neg
    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Evaluation data saved to {eval_file}")