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
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/daisee_base.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/daisee_inference.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_daisee_base_config_eval.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_daisee_test_config_eval.json --retries 5  
    
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/engagenet_base.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/engagenet_finetune.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_engagenet_base_config_eval.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_engagenet_finetune_config_eval.json --retries 5
    '''
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument(
        '-result-path',
        "--result-path", 
        required=True, 
        help="path to result file."
    )
    parser.add_argument(
        '-retries',
        "--retries", 
        required=False,
        type=int,
        default=3, 
        help="number of retries for openai api."
    )    
    parser.add_argument(
        '-classes',
        "--classes", 
        required=False,
        type=int,
        default=4, 
        help="number of classes."
    )
    args = parser.parse_args()
    return args

def sanitize_sentence(sentence: str) -> str:
    """
    Escapes special characters in the sentence to prevent JSON formatting issues.
    """
    return sentence.replace('\n', '').replace('\\r', '').replace('\\', '\\\\')
    # return sentence.replace('\\', '\\\\').replace('"', '\\"')

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

The only output that matters are the values for the key "pred".
The only possible values for the key "pred" are "not-engaged", "barely-engaged", "engaged", and "highly-engaged".
However, if none of these values are contained within the sentence, then the value for the key "pred" should be the sentence itself.

Example 1:
User: "The student is Not-Engaged."
OUTPUT:
{{
    "pred": "Not-Engaged"
}}

Example 2:
User: "The student is Barely-Engaged."
OUTPUT:
{{
    "pred": "Barely-Engaged"
}}

Example 3:
User: "The student is Engaged."
OUTPUT:
{{
    "pred": "Engaged"
}}

Example 4:
User: "The student is Highly-Engaged."
OUTPUT:
{{
    "pred": "Highly-Engaged"
}}
"""
    responses = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":prompt_template},
            {"role": "user", "content":f"Parse the following sentence:{sentence}"}
        ],
        response_format={"type": "json_object"},
        max_tokens=128,
        temperature=0.0,
        n=retries
    )      
    message_content = ' '
    for res in range(retries):
        try:
            message_content = sanitize_sentence(responses.choices[res].message.content.strip())
            message_content = extract_json(message_content)
            message_json = json.loads(message_content)

            if 'pred' not in message_json:
                message_json['pred'] = message_content
            if message_json['pred'].strip() == '':
                message_json['pred'] = "*"

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
    retries:int,
    classes:int=4,
    video:bool=False
)->dict:
    mapping = {
        'The student is Not-Engaged':0,
        'The student is Barely-Engaged':1,
        'The student is Engaged':2,
        'The student is Highly-Engaged':3
    }
    # mapping = {
    #     'The student is not-engaged':0,
    #     'The student is barely-engaged':1,
    #     'The student is engaged':2,
    #     'The student is highly-engaged':3
    # }
    metrics_keyword = load_metrics(classes)
    metrics_struc = load_metrics(classes)

    inference_samples = len(results)
    logger.info(f"INFERENCE SAMPLES - {inference_samples}")

    pred_keyword,target_keyword = torch.zeros(inference_samples),torch.zeros(inference_samples)
    pred_struc,target_struc = torch.zeros(inference_samples),torch.zeros(inference_samples)

    json_results = {}
    id_key = 'video_name' if video else 'video_id'

    for i,sample in enumerate(results):
        answer,pred = sample['A'],sample['pred']
        pred = pred[0] if video else pred
        target_keyword[i] = mapping[answer]
        pred_keyword[i] = target_keyword[i]

        if not check_string_in_output(pred,answer.split(' ')[-1]):
            pred_keyword[i] = (target_keyword[i] - 1) % classes   
            logger.info(f"WRONG pred {sample[id_key]}")
        metrics_keyword.forward(pred_keyword[:i + 1],target_keyword[:i + 1])

        parsed_json = openai_parser(pred,retries=retries)
        logger.info("Parsed JSON:")
        logger.info(json.dumps(parsed_json, indent=4))
        logger.info(f'\nA:{answer.split(" ")[-1]}\nP:{parsed_json["pred"]}')
        
        target_struc[i] = mapping[answer]
        pred_struc[i] = target_struc[i]
        
        if isinstance(parsed_json['pred'],list) or\
            isinstance(parsed_json['pred'],dict) or\
             parsed_json['pred'].lower() not in answer.split(' ')[-1].lower():
            pred_struc[i] = (target_struc[i] - 1) % classes
        metrics_struc.forward(pred_struc[:i + 1],target_struc[:i + 1])   

        json_results[sample[id_key]] = (parsed_json['pred'],answer.split(' ')[-1])
        

    performance = metrics_keyword.compute()
    logger.info(f"Key word ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"Key word PR - {performance['MulticlassPrecision']}")
    logger.info(f"Key word RE - {performance['MulticlassRecall']}")
    logger.info(f"Key word F1 - {performance['MulticlassF1Score']}")
    metrics_keyword.reset()   

    performance = metrics_struc.compute()
    logger.info(f"Structure ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"Structure PR - {performance['MulticlassPrecision']}")
    logger.info(f"Structure RE - {performance['MulticlassRecall']}")
    logger.info(f"Structure F1 - {performance['MulticlassF1Score']}")
    metrics_struc.reset()   
    return json_results


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
    json_result = get_acc(
        results=results,
        retries=args.retries,
        classes=args.classes,
        video='video' in args.result_path
    )

    # /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/daisee_inference.json
    outpath = args.result_path.split('/')
    outpath[0] = '/'
    if not os.path.exists('structured'):
        os.mkdir('structured')
    with open(os.path.join('structured',outpath[-1]),'w') as f:
        json.dump(json_result,f,indent=4)


if __name__ == "__main__":
    """
    minigpt4_eval_outputs/daisee_base.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.03365866467356682
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.22113727033138275
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.018307646736502647
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.023643970489501953
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.031656429171562195
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.23444971442222595
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.028618473559617996
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.029216088354587555

    minigpt4_eval_outputs/daisee_inference.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.5278856158256531
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.37611472606658936
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.24560368061065674
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.2070852369070053
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.9130264520645142
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.5903357863426208
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.696989119052887
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.6137951016426086

    minigpt4video_eval_outputs/mistral_daisee_base_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.5688818097114563
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.4028744101524353
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.28821173310279846
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.3005830943584442
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.4408881962299347
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.3947935104370117
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.39212799072265625
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.31713855266571045

    minigpt4video_eval_outputs/mistral_daisee_test_config_eval.json


    minigpt4_eval_outputs/engagenet_base.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.03188373148441315
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.04356253519654274
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.026909854263067245
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.02992699295282364
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.023858124390244484
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.04703712463378906
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.02129654958844185
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.029087310656905174

    minigpt4_eval_outputs/engagenet_finetune.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.38917461037635803
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.303382009267807
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.33527135848999023
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.2853012681007385
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.27623289823532104
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.24811212718486786
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.21550515294075012
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.21532279253005981

    minigpt4video_eval_outputs/mistral_engagenet_base_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.35479679703712463
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.24729931354522705
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.2997397184371948
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.2471315562725067
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.5129430294036865
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.4749894142150879
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.4634208679199219
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.44980400800704956

    minigpt4video_eval_outputs/mistral_engagenet_finetune_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.6407010555267334
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.723282516002655
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.6108910441398621
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.5911238789558411
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.6754897832870483
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.6039474606513977
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.6162317991256714
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.5889008045196533

    """
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    client = OpenAI()
    main()

