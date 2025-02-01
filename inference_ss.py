import argparse
import os
import re
import sys
import glob
import json
import random
import logging 
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import torchmetrics
import minigpt4.tasks as tasks
from PIL import Image
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score
from sentence_transformers import SentenceTransformer, util

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from minigpt4.conversation.conversation import StoppingCriteriaList, StoppingCriteriaSub

def init_logger(
    program:str    
)->logging.Logger:
    # Set up logging
    logger = logging.getLogger(program)
    logger.setLevel(logging.INFO)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(f'logs/{os.path.splitext(program)[0]}.log')
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

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def init_inference(args:argparse.ArgumentParser)->tuple:
    logger.info('Initializing Chat')
    cfg = Config(args)
    setup_seeds(cfg)
    logger.info(f"CONFIG:\n{cfg}")

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device='cuda:{}'.format(args.gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.engagenet.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    logger.info('Initialization Finished')
    return model,vis_processor

def generate_kwargs(
    embs:torch.tensor,
    stopping_criteria:StoppingCriteriaList,
    max_new_tokens:int=20, 
    num_beams:int=1, 
    min_length:int=1, 
    top_p:float=0.9,
    repetition_penalty:int=1.50, 
    length_penalty:int=1, 
    temperature:int=0.01, 
)->dict:
    return dict(
        inputs_embeds=embs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=num_beams,
        do_sample=True,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=float(temperature),
    )

def embedding_prepare(
    model:torch.nn.Module,
    prompt:str, 
    img_list:list,
    max_length:int=2000,
    max_new_tokens:int=300, 
)->torch.tensor:
    embs = model.get_context_emb(prompt, img_list)
    current_max_len = embs.shape[1] + max_new_tokens
    if current_max_len - max_length > 0:
        logger.info('Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.')
    begin_idx = max(0, current_max_len - max_length)
    embs = embs[:, begin_idx:]
    return embs,max_new_tokens

def model_answer(
    model:torch.nn.Module,
    inputs:dict,
)->str:
    with model.maybe_autocast():
        output_token = model.llama_model.generate(**inputs)[0]
    output_text = model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    return output_text

def parse_args():
    """
    python3 inference.py \
        --gpu-id 1 \
        --cfg-path eval_configs/minigpt4_eval.yaml 

    python3 inference.py \
        --gpu-id 0 \
        --cfg-path eval_configs/minigpt4_eval.yaml \
        --test-dir /mnt/nvme2tb/EngageNetFrames/test \
        --test-labels /mnt/nvme2tb/ieee_fer_dpo/engagenet_captions/test_filter_cap.json \
        --out-json engagenet_base.json
            
    """
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('-cfg-path',"--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument('-classes',"--classes",type=int, default=4,help="The number of classes")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--test-dir", 
        type=str, 
        default='/mnt/nvme2tb/DAiSEE_Frames/Test_frames', 
        help="directory of images to evaluate"
    )
    parser.add_argument(
        "--test-labels", 
        type=str, 
        default='/mnt/nvme2tb/ieee_fer_dpo/daisee_captions/test_filter_cap.json', 
        help="directory of images to evaluate"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--eval_prompts", 
        type=str, 
        default='prompts/instruction_align.txt', 
        help="text file of instruction prompts"
    )
    parser.add_argument(
        "--eval_questions", 
        type=str, 
        default='prompts/daisee_questions.txt', 
        help="text file of question prompts"
    )
    parser.add_argument(
        "--out-json", 
        type=str, 
        default='inference_daisee.json', 
        help="out json file name"
    )
    parser.add_argument(
        "--saved-dir", 
        type=str, 
        default='results/engagenet_dpo_infer.json', 
        help="directory of saved evaluation answers"
    )
    args = parser.parse_args()
    return args

def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="micro"),
        MulticlassPrecision(num_classes=num_classes, average="weighted"),
        MulticlassRecall(num_classes=num_classes, average="weighted"),
        MulticlassF1Score(num_classes=num_classes, average="weighted"),
    ])
    return metrics


def get_test_labels(
    label_path:str
)->dict:
    # mapping = {
    #     'The student is not-engaged':0,
    #     'The student is barely-engaged':1,
    #     'The student is engaged':2,
    #     'The student is highly-engaged':3
    # }
    mapping = {
        'The student is Not-Engaged':0,
        'The student is Barely-Engaged':1,
        'The student is Engaged':2,
        'The student is Highly-Engaged':3
    }
    with open(label_path,'r') as f:
        labels = json.load(f)

    with open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w') as f:
        json.dump(labels,f,indent=4)
    
    return labels,mapping

def get_saved_labels(file_path:str)->dict:
    classes = np.array([
        [0,"The student is not-engaged"],
        [1,"The student is barely-engaged"],
        [2,"The student is engaged"],
        [3,"The student is highly-engaged"],
    ])
    # mapping = {
    #     'The student is not-engaged':0,
    #     'The student is barely-engaged':1,
    #     'The student is engaged':2,
    #     'The student is highly-engaged':3
    # }
    mapping = {
        'The student is Not-Engaged':0,
        'The student is Barely-Engaged':1,
        'The student is Engaged':2,
        'The student is Highly-Engaged':3
    }
    with open(file_path,'r') as f:
        labels = json.load(f)

    # with open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w') as f:
    #     json.dump(labels,f,indent=4)
    
    return labels,classes,mapping

def check_string_in_output(
    output:str, 
    search_string:str
)->bool:
    # Escape special characters in search_string if necessary
    output,search = re.sub(r'\W', '', output).lower(),re.sub(r'\W', '', search_string).lower()
    pattern = re.escape(search)
    match = re.search(pattern, output)
    return bool(match)

def sentence_eval(classes, model, ref, answer):
    query_embedding = model.encode(answer)
    passage_embedding = model.encode([c[1] for c in classes])
    score=util.cos_sim(query_embedding, passage_embedding)[0]
    return max(score)==score[int(ref)]

def main()->None:
    args = parse_args()
    model,vis_processor = init_inference(args)
    
    #test_label_path,test_dir = args.test_labels,args.test_dir
    
    # with open(args.eval_prompts, 'r', encoding='utf-8') as file:
    #     prompt = file.read()
    # instruction_pool = prompt.split('\n\n')

    # question = "\nQuestion: What is the students engagement level?\n"
    # with open(args.eval_questions,'r') as f:
    #     questions = [q for q in f.read().split('\n') if q != question]
    file_path=args.saved_dir
    labels,classes,mapping = get_saved_labels(file_path)

    # stop_words_ids = [torch.tensor([2]).to("cuda:{}".format(args.gpu_id))]
    # stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    metrics = load_metrics(args.classes).to(device='cuda:{}'.format(args.gpu_id))
    inference_samples = len(labels)
    logger.info(f"INFERENCE SAMPLES - {inference_samples}")
    pred_table,target_table = torch.zeros(inference_samples).to(device='cuda:{}'.format(args.gpu_id)),torch.zeros(inference_samples).to(device='cuda:{}'.format(args.gpu_id))
    #model.eval()

    answers = []
    for i,subject in enumerate(labels):
        subject_sample = subject['video_id']
        #instruct_sample = random.choice(instruction_pool) + f"\n\n### Input:\n"
        #q2 = random.sample(questions,2)
        # question1 = "\n"+q2[0]+"\n"
        # question2 = "\n"+q2[-1]+"\n"
        target_table[i] = mapping[subject['A']]
        pred_table[i] = target_table[i]
        
        pred = subject['pred']
        pred1 = subject['pred1']
        logger.info(f"SUBJECT: {subject_sample}")
        logger.info(f"CAPTION - {subject['A'].split(' ')[-1].lower()}")
        logger.info(f"OUTPUT - {pred[0].lower()}")
        logger.info(f"OUTPUT1 - {pred1[0].lower()}")

        eval_model = SentenceTransformer("BAAI/bge-m3")
        if sentence_eval(classes, eval_model, mapping[subject['A']],pred[0])==False and sentence_eval(classes, eval_model, mapping[subject['A']],pred1[0])==False:
            pred_table[i] = (target_table[i] - 1) % args.classes
        performance = metrics.forward(pred_table[:i + 1],target_table[:i + 1])
        logger.info(f"ACC - {performance['MulticlassAccuracy']}")
        logger.info(f"PR - {performance['MulticlassPrecision']}")
        logger.info(f"RE - {performance['MulticlassRecall']}")
        logger.info(f"F1 - {performance['MulticlassF1Score']}")
        


    performance = metrics.compute()
    logger.info(f"FINAL ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"FINAL PR - {performance['MulticlassPrecision']}")
    logger.info(f"FINAL RE - {performance['MulticlassRecall']}")
    logger.info(f"FINAL F1 - {performance['MulticlassF1Score']}")
    metrics.reset()
    # with open(f"results/{args.out_json}", 'w') as f:
    #     json.dump(answers, f, indent=4)

if __name__ == "__main__":
    """
    /home/tony/MiniGPT-4/results/inference.json 1784
    Average score for correctness: 4.240470852017937
    Average score for detailed orientation: 3.975896860986547
    Average score for contextual understanding: 4.230381165919282
    Average score temporal understanding: 4.176008968609866
    Average score for consistency: 3.2045964125560538

    FINAL ACC - 0.5375036001205444
    FINAL PR - 0.3783494830131531
    FINAL RE - 0.25030452013015747
    FINAL F1 - 0.21113687753677368
    FINAL COUNT ACC - 0.547645739910314

    MiniGPT-4 base engagenet
    [inference.py|INFO|2025-01-17] FINAL ACC - 0.03188373148441315
    [inference.py|INFO|2025-01-17] FINAL PR - 0.04356253519654274
    [inference.py|INFO|2025-01-17] FINAL RE - 0.026909854263067245
    [inference.py|INFO|2025-01-17] FINAL F1 - 0.02992699295282364

    MiniGPT-4 finetune engagenet
    [inference.py|INFO|2025-01-17] FINAL ACC - 0.38917461037635803                                                                                                                                             
    [inference.py|INFO|2025-01-17] FINAL PR - 0.3033820390701294                                                                                                                                               
    [inference.py|INFO|2025-01-17] FINAL RE - 0.33527135848999023                                                                                                                                              
    [inference.py|INFO|2025-01-17] FINAL F1 - 0.2853012681007385                                                                                                                                               
    """
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()