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

from sentence_transformers import SentenceTransformer, util


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
    python3 daisee_evaluation.py --result-path /home/tony/MiniGPT-4/results/daisee_inference.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_daisee_base_config_eval.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_daisee_test_config_eval.json --retries 5  
    
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/engagenet_base.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4_eval_outputs/engagenet_finetune.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_engagenet_base_config_eval.json --retries 5
    python3 daisee_evaluation.py --result-path /home/tony/nvme2tb/ieee_fer_dpo/minigpt4video_eval_outputs/mistral_engagenet_finetune_config_eval.json --retries 5
    
    python3 daisee_evaluation.py --result-path /data/inference_daisee.json --retries 5
    python3 daisee_evaluation.py --result-path /data/engagenet_dpo_infer.json --retries 5

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

def sentence_eval(
    classes:tuple, 
    model:torch.nn.Module, 
    ref:str, 
    answer:str
)->bool:
    query_embedding = model.encode(answer)
    passage_embedding = model.encode([c for c in classes])
    score = util.cos_sim(query_embedding, passage_embedding)[0]
    return max(score)==score[int(ref)]

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
        MulticlassPrecision(num_classes=num_classes, average="weighted"),
        MulticlassRecall(num_classes=num_classes, average="weighted"),
        MulticlassF1Score(num_classes=num_classes, average="weighted"),
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
    # metrics_dist = load_metrics(classes)

    inference_samples = len(results)
    logger.info(f"INFERENCE SAMPLES - {inference_samples}")

    pred_keyword,target_keyword = torch.zeros(inference_samples),torch.zeros(inference_samples)
    pred_struc,target_struc = torch.zeros(inference_samples),torch.zeros(inference_samples)
    # pred_dist,target_dist = torch.zeros(inference_samples),torch.zeros(inference_samples)
    best_of_two = {}
    json_results = {}
    id_key = 'video_name' if video else 'video_id'

    # eval_model = SentenceTransformer("BAAI/bge-m3")
    # eval_model.eval()
    '''
    if not check_string_in_output(pred[0],subject['caption'].split(' ')[-1]):
            if not check_string_in_output(pred1[0],subject['caption'].split(' ')[-1]):
    '''
    for i,sample in enumerate(results):
        answer,pred = sample['A'],sample['pred'] if isinstance(sample['pred'], str) else sample['pred'][0]
        pred1 = sample['pred1'] if isinstance(sample['pred1'], str) else sample['pred1'][0]
        target_keyword[i] = target_struc[i] = mapping[answer] # target_dist[i] target_struc[i] =

        pred_keyword[i] = target_keyword[i]
        if not check_string_in_output(pred,answer.split(' ')[-1]) and not check_string_in_output(pred1,answer.split(' ')[-1]) :
            pred_keyword[i] = (target_keyword[i] - 1) % classes   
            logger.info(f"WRONG pred {sample[id_key]}")
        metrics_keyword.forward(pred_keyword[:i + 1],target_keyword[:i + 1])
        best_of_two[sample[id_key]] = int(check_string_in_output(pred1,answer.split(' ')[-1])) 

        # pred_dist[i] = target_dist[i]
        # if not sentence_eval(mapping.keys(), eval_model,target_dist[i],pred):
        #     pred_dist[i] = (target_dist[i] - 1) % classes
        # metrics_dist.forward(pred_dist[:i + 1],target_dist[:i + 1]) 

        parsed_0 = openai_parser(pred,retries=retries)
        parsed_1 = openai_parser(pred1,retries=retries)

        logger.info("Parsed JSON:")
        logger.info(json.dumps(parsed_0, indent=4))
        logger.info(json.dumps(parsed_1, indent=4))

        logger.info(f'\nA:{answer.split(" ")[-1]}\nP:{parsed_0["pred"]}\nP1:{parsed_1["pred"]}')

        pred_struc[i] = target_struc[i]
        if isinstance(parsed_0['pred'],list) and isinstance(parsed_1['pred'],list)  or\
             isinstance(parsed_0['pred'],dict) and isinstance(parsed_1['pred'],dict) or\
             parsed_0['pred'].lower() not in answer.split(' ')[-1].lower() and\
             parsed_1['pred'].lower() not in answer.split(' ')[-1].lower():
            pred_struc[i] = (target_struc[i] - 1) % classes

        metrics_struc.forward(pred_struc[:i + 1],target_struc[:i + 1])   

        json_results[sample[id_key]] = (
            parsed_0['pred'],
            parsed_1['pred'],
            answer.split(' ')[-1],
        )
        

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

    # performance = metrics_dist.compute()
    # logger.info(f"Dist ACC - {performance['MulticlassAccuracy']}")
    # logger.info(f"Dist PR - {performance['MulticlassPrecision']}")
    # logger.info(f"Dist RE - {performance['MulticlassRecall']}")
    # logger.info(f"Dist F1 - {performance['MulticlassF1Score']}")
    return json_results,best_of_two


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
    json_result,best_of_two = get_acc(
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

    if not os.path.exists('best_of_two'):
        os.mkdir('best_of_two')
    with open(os.path.join('best_of_two',outpath[-1]),'w') as f:
        json.dump(best_of_two,f,indent=4)

    logger.info(f"STRUCTURE DUMPED TO - {os.path.join('structured',outpath[-1])}")

if __name__ == "__main__":
    """
    minigpt4_eval_outputs/daisee_base.json #TODO REDO
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.03365866467356682
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.22113727033138275
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.018307646736502647
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.023643970489501953
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.031656429171562195
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.23444971442222595
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.028618473559617996
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.029216088354587555

    minigpt4_eval_outputs/daisee_inference.json # TODO REDO
    [inference.py|INFO|2025-01-28] FINAL ACC - 0.6791259050369263
    [inference.py|INFO|2025-01-28] FINAL PR - 0.5235922932624817
    [inference.py|INFO|2025-01-28] FINAL RE - 0.5938832759857178
    [inference.py|INFO|2025-01-28] FINAL F1 - 0.5322630405426025


    minigpt4video_eval_outputs/mistral_daisee_base_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-31] Key word ACC - 0.5688818097114563
    [daisee_evaluation.py|INFO|2025-01-31] Key word PR - 0.7374650835990906
    [daisee_evaluation.py|INFO|2025-01-31] Key word RE - 0.5688818097114563
    [daisee_evaluation.py|INFO|2025-01-31] Key word F1 - 0.5799382328987122
    [daisee_evaluation.py|INFO|2025-01-31] Structure ACC - 0.44269511103630066
    [daisee_evaluation.py|INFO|2025-01-31] Structure PR - 0.6930316686630249
    [daisee_evaluation.py|INFO|2025-01-31] Structure RE - 0.44269514083862305
    [daisee_evaluation.py|INFO|2025-01-31] Structure F1 - 0.5275216102600098
    [daisee_evaluation.py|INFO|2025-01-31] Dist ACC - 0.31278279423713684
    [daisee_evaluation.py|INFO|2025-01-31] Dist PR - 0.44077253341674805
    [daisee_evaluation.py|INFO|2025-01-31] Dist RE - 0.31278279423713684
    [daisee_evaluation.py|INFO|2025-01-31] Dist F1 - 0.3580528795719147
    [daisee_evaluation.py|INFO|2025-01-31] STRUCTURE DUMPED TO - structured/mistral_daisee_base_config_eval.json

    minigpt4video_eval_outputs/mistral_daisee_test_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-31] Key word ACC - 0.6602604985237122                                                                      
    [daisee_evaluation.py|INFO|2025-01-31] Key word PR - 0.7764840126037598                                                                       
    [daisee_evaluation.py|INFO|2025-01-31] Key word RE - 0.6602604985237122                                                                       
    [daisee_evaluation.py|INFO|2025-01-31] Key word F1 - 0.6716954708099365                                                                       
    [daisee_evaluation.py|INFO|2025-01-31] Structure ACC - 0.6105217933654785                                                                     
    [daisee_evaluation.py|INFO|2025-01-31] Structure PR - 0.8676671981811523                                                                      
    [daisee_evaluation.py|INFO|2025-01-31] Structure RE - 0.6105217933654785                                                                      
    [daisee_evaluation.py|INFO|2025-01-31] Structure F1 - 0.6874428987503052                                                                      
    [daisee_evaluation.py|INFO|2025-01-31] Dist ACC - 0.4328434467315674                                                                          
    [daisee_evaluation.py|INFO|2025-01-31] Dist PR - 0.6979579925537109                                                                           
    [daisee_evaluation.py|INFO|2025-01-31] Dist RE - 0.43284347653388977                                                                          
    [daisee_evaluation.py|INFO|2025-01-31] Dist F1 - 0.532105028629303                                                                            
    [daisee_evaluation.py|INFO|2025-01-31] STRUCTURE DUMPED TO - structured/mistral_daisee_test_config_eval.json       

    minigpt4_eval_outputs/engagenet_base.json #TODO REDO
    [daisee_evaluation.py|INFO|2025-01-24] Key word ACC - 0.03188373148441315
    [daisee_evaluation.py|INFO|2025-01-24] Key word PR - 0.04356253519654274
    [daisee_evaluation.py|INFO|2025-01-24] Key word RE - 0.026909854263067245
    [daisee_evaluation.py|INFO|2025-01-24] Key word F1 - 0.02992699295282364
    [daisee_evaluation.py|INFO|2025-01-24] Structure ACC - 0.023858124390244484
    [daisee_evaluation.py|INFO|2025-01-24] Structure PR - 0.04703712463378906
    [daisee_evaluation.py|INFO|2025-01-24] Structure RE - 0.02129654958844185
    [daisee_evaluation.py|INFO|2025-01-24] Structure F1 - 0.029087310656905174

    minigpt4_eval_outputs/engagenet_finetune.json 
    [daisee_evaluation.py|INFO|2025-01-31] Key word ACC - 0.6791259050369263
    [daisee_evaluation.py|INFO|2025-01-31] Key word PR - 0.6882764101028442
    [daisee_evaluation.py|INFO|2025-01-31] Key word RE - 0.6791259050369263
    [daisee_evaluation.py|INFO|2025-01-31] Key word F1 - 0.6495401859283447
    [daisee_evaluation.py|INFO|2025-01-31] Structure ACC - 0.6053004860877991
    [daisee_evaluation.py|INFO|2025-01-31] Structure PR - 0.6674372553825378
    [daisee_evaluation.py|INFO|2025-01-31] Structure RE - 0.6053004860877991
    [daisee_evaluation.py|INFO|2025-01-31] Structure F1 - 0.6242648363113403
    [daisee_evaluation.py|INFO|2025-01-31] Dist ACC - 0.4633844196796417
    [daisee_evaluation.py|INFO|2025-01-31] Dist PR - 0.5882208943367004
    [daisee_evaluation.py|INFO|2025-01-31] Dist RE - 0.4633844196796417
    [daisee_evaluation.py|INFO|2025-01-31] Dist F1 - 0.5081114172935486
    [daisee_evaluation.py|INFO|2025-01-31] STRUCTURE DUMPED TO - structured/engagenet_finetune.json                                     

    minigpt4video_eval_outputs/mistral_engagenet_base_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-31] Key word ACC - 0.35479679703712463
    [daisee_evaluation.py|INFO|2025-01-31] Key word PR - 0.3258094787597656
    [daisee_evaluation.py|INFO|2025-01-31] Key word RE - 0.35479676723480225
    [daisee_evaluation.py|INFO|2025-01-31] Key word F1 - 0.3110787272453308
    [daisee_evaluation.py|INFO|2025-01-31] Structure ACC - 0.4983034133911133
    [daisee_evaluation.py|INFO|2025-01-31] Structure PR - 0.5771093368530273
    [daisee_evaluation.py|INFO|2025-01-31] Structure RE - 0.4983034133911133
    [daisee_evaluation.py|INFO|2025-01-31] Structure F1 - 0.5261300802230835
    [daisee_evaluation.py|INFO|2025-01-31] Dist ACC - 0.3845134377479553
    [daisee_evaluation.py|INFO|2025-01-31] Dist PR - 0.3979026675224304
    [daisee_evaluation.py|INFO|2025-01-31] Dist RE - 0.38451340794563293
    [daisee_evaluation.py|INFO|2025-01-31] Dist F1 - 0.3839612603187561
    [daisee_evaluation.py|INFO|2025-01-31] STRUCTURE DUMPED TO - structured/mistral_engagenet_base_config_eval.json
    
    minigpt4video_eval_outputs/mistral_engagenet_finetune_config_eval.json
    [daisee_evaluation.py|INFO|2025-01-31] Key word ACC - 0.6407010555267334
    [daisee_evaluation.py|INFO|2025-01-31] Key word PR - 0.722614049911499
    [daisee_evaluation.py|INFO|2025-01-31] Key word RE - 0.6407010555267334
    [daisee_evaluation.py|INFO|2025-01-31] Key word F1 - 0.6263036727905273
    [daisee_evaluation.py|INFO|2025-01-31] Structure ACC - 0.6740406155586243
    [daisee_evaluation.py|INFO|2025-01-31] Structure PR - 0.7318193912506104
    [daisee_evaluation.py|INFO|2025-01-31] Structure RE - 0.6740406155586243
    [daisee_evaluation.py|INFO|2025-01-31] Structure F1 - 0.6929082870483398
    [daisee_evaluation.py|INFO|2025-01-31] Dist ACC - 0.479320764541626
    [daisee_evaluation.py|INFO|2025-01-31] Dist PR - 0.6000171899795532
    [daisee_evaluation.py|INFO|2025-01-31] Dist RE - 0.4793207347393036
    [daisee_evaluation.py|INFO|2025-01-31] Dist F1 - 0.5176502466201782
    [daisee_evaluation.py|INFO|2025-01-31] STRUCTURE DUMPED TO - structured/mistral_engagenet_finetune_config_eval.json

    """
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    client = OpenAI()
    main()

