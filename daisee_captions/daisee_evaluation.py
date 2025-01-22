import json
import os
import re
import sys
import logging
import argparse
import openai
from openai import OpenAI

import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score



def init_logger(
    program:str    
)->logging.Logger:
    # Set up logging
    logger = logging.getLogger(program)
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(f'../logs/{os.path.splitext(program)[0]}.log')
    fh.setLevel(logging.INFO)

    # Create console handler with a higher log level
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(name)s|%(levelname)s|%(asctime)s] %(message)s',
        datefmt="%Y-%m-%d"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("Initializing ok.")

    return logger

def parse_args():
    '''
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/daisee_inference.json
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/daisee_base.json
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/engagenet_base.json
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/engagenet_finetune.json
    
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_daisee_base_config_eval.json
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_daisee_test_config_eval.json
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_engagenet_base_config_eval.json
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_engagenet_finetune_config_eval.json
    '''
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        '-result-path',
        "--result-path", 
        required=True, 
        help="path to result file."
    )
  
    args = parser.parse_args()
    return args

def sanitize_sentence(sentence: str) -> str:
    """
    Escapes special characters in the sentence to prevent JSON formatting issues.
    """
    return sentence.replace('\\', '\\\\').replace('"', '\\"')

def extract_json(content: str) -> str:
    """
    Extracts JSON string from the response, especially if it's enclosed within code blocks.
    """
    # Pattern to match JSON within code blocks
    json_pattern = re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL)
    match = json_pattern.search(content)
    if match:
        return match.group(1)
    
    # If not in code block, attempt to extract the first JSON object
    json_start = content.find('{')
    json_end = content.rfind('}')
    if json_start != -1 and json_end != -1:
        return content[json_start:json_end+1]
    
    # Return the entire content if no braces found (may cause JSONDecodeError)
    return content



def openai_parser(
    sentence:str,
    retries:int=3
)->str:
    sentence = sanitize_sentence(sentence)
    prompt_template = f"""
You are an AI assistant that parses English sentences into JSON format.

Given the following sentence:
"{sentence}"

Parse it into a well-structured JSON object with appropriate keys and values.

Ensure the JSON is valid and follows proper syntax.

Example 1:
Sentence: "The student is Not-Engaged."
JSON:
{{
    "pred": "Not-Engaged"
}}

Example 2:
Sentence: "The student is Barely-Engaged."
JSON:
{{
    "pred": "Barely-Engaged"
}}

Example 3:
Sentence: "The student is Engaged."
JSON:
{{
    "pred": "Engaged"
}}

Example 4:
Sentence: "The student is Highly-Engaged."
JSON:
{{
    "pred": "Highly-Engaged"
}}
"""
    message_content = ''
    for att in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":prompt_template},
                    {"role": "user", "content":f"Parse the following sentence:{sentence}"}
                ],
                response_format={"type": "json_object"},
                max_tokens=128,
                temperature=0.0,
                n=1
            )        
            message_content = response.choices[0].message.content.strip()
            message_content = extract_json(message_content)
            message_json = json.loads(message_content)

            if 'pred' not in message_json:
                message_json['pred'] = ''
                return message_json
            return message_json
            
        except json.decoder.JSONDecodeError as je:
            logger.info(f"JSONDecodeError - {je}")
            continue

    logger.info(f"FAILED TO PARSE {retries} TIMES")
    return {'pred':message_content}

def check_string_in_output(
    output:str, 
    search_string:str
)->bool:
    # Escape special characters in search_string if necessary
    output,search = re.sub(r'\W', '', output).lower(),re.sub(r'\W', '', search_string).lower()
    pattern = re.escape(search)
    match = re.search(pattern, output)
    return bool(match)


def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="micro"),
        MulticlassPrecision(num_classes=num_classes, average="macro"),
        MulticlassRecall(num_classes=num_classes, average="macro"),
        MulticlassF1Score(num_classes=num_classes, average="macro"),
    ])
    return metrics


def get_acc(
    results:dict,
    classes:int=4,
    video:bool=False
)->dict:
    # mapping = {
    #     'The student is Not-Engaged':0,
    #     'The student is Barely-Engaged':1,
    #     'The student is Engaged':2,
    #     'The student is Highly-Engaged':3
    # }
    mapping = {
        'The student is not-engaged':0,
        'The student is barely-engaged':1,
        'The student is engaged':2,
        'The student is highly-engaged':3
    }
    metrics = load_metrics(classes)

    inference_samples = len(results)
    logger.info(f"INFERENCE SAMPLES - {inference_samples}")
    pred_table,target_table = torch.zeros(inference_samples),torch.zeros(inference_samples)
    count, json_results = 0,{}
    id_key = 'video_name' if video else 'video_id'
    for i,sample in enumerate(results):
        answer,pred = sample['A'],sample['pred']
        pred = pred[0] if video else pred
        count += 1
        target_table[i] = mapping[answer]
        pred_table[i] = target_table[i]

        if not check_string_in_output(pred,answer.split(' ')[-1]):
            pred_table[i] = (target_table[i] - 1) % classes   
            logger.info(f"WRONG pred {sample[id_key]}")

        parsed_json = openai_parser(pred)
        logger.info("Parsed JSON:")
        logger.info(json.dumps(parsed_json, indent=4))
        logger.info(f'\nA:{answer.split(" ")[-1]}\nP:{parsed_json["pred"]}')
        
        if not isinstance(parsed_json['pred'],str) or\
             parsed_json['pred'].lower() not in answer.split(' ')[-1].lower() :
            count -= 1
        performance = metrics.forward(pred_table[:i + 1],target_table[:i + 1])
        json_results[sample[id_key]] = (parsed_json['pred'],answer.split(' ')[-1].lower())
    
    # /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/daisee_inference.json
    outpath = args.resut_path.split('/')
    if not os.path.exists(os.path.join(outpath[:-2],'structured')):
        os.mkdir(os.path.join(outpath[:-2],'structured'))
    with open(os.path.join(outpath[:-2],'structured',outpath[-1]),'w') as f:
        json.dump(json_results,f,indent=4)

    performance = metrics.compute()
    logger.info(f"Key word ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"Key word PR - {performance['MulticlassPrecision']}")
    logger.info(f"Key word RE - {performance['MulticlassRecall']}")
    logger.info(f"Key word F1 - {performance['MulticlassF1Score']}")
    logger.info(f"Parsed ACC - {count/inference_samples}")
    metrics.reset()   
    return


def threshold_gpt_eval(
    result_path:str,
)->None:
    '''
    FINAL COUNT ACC - 0.8609865470852018
    '''
    with open(result_path,'r') as f:
        results = json.load(f)
    total_samples = len(results)
    score_above_thres = 0
    for subject in results:
        subject = results[subject]
        score_above_thres += int(subject[0]['score'] >= 4)

    logger.info(f"FINAL COUNT ACC - {score_above_thres/total_samples}")


def main()->None:
    args = parse_args()
    result_path = args.result_path

    with open(result_path,'r') as f:
        results = json.load(f)
    get_acc(results,video='video' in args.result_path)
    


if __name__ == "__main__":
    """
    daisee_inference.json
    [daisee_evaluation.py|INFO|2025-01-22] Key word ACC - 0.5278856158256531
    [daisee_evaluation.py|INFO|2025-01-22] Key word PR - 0.37611472606658936
    [daisee_evaluation.py|INFO|2025-01-22] Key word RE - 0.24560368061065674
    [daisee_evaluation.py|INFO|2025-01-22] Key word F1 - 0.2070852369070053
    [daisee_evaluation.py|INFO|2025-01-22] Parsed ACC - 0.8772421524663677

    """
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    client = OpenAI()
    main()

