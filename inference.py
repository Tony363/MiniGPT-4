import argparse
import os
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
    logger.info(vars(args))
    cfg = Config(args)
    setup_seeds(cfg)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device='cuda:{}'.format(args.gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    # chat = ChatInference(model, vis_processor, device='cuda:{}'.format(args.gpu_id),annotate=annotate,conv_rec=3)
    # logger.info(f"USING DEVICE - {chat.model.device}")
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
    python3 inference.py --gpu-id 1 --cfg-path eval_configs/minigpt4_eval.yaml --consistency-qa gpt_evaluation/consistency_qa_raw.json
    """
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('-cfg-path',"--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument('-classes',"--classes",type=int, default=3,help="The number of classes")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--test-dir", 
        type=str, 
        default='/home/tony/nvme2tb/DAiSEE_Frames/Test_frames', 
        help="directory of images to evaluate"
    )
    parser.add_argument(
        "--test-labels", 
        type=str, 
        default='daisee_captions/test_filter_cap.json', 
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
        '-consistency-qa','--consistency-qa', 
        type=str,
        default='/home/tony/MiniGPT-4/gpt_evaluation/consistency_qa_raw.json',
        help='json of qa pairs', 
        required=True
    )
    args = parser.parse_args()
    return args


def main()->None:
    args = parse_args()
    model,vis_processor = init_inference(args)
    
    test_label_path,test_dir = args.test_labels,args.test_dir
    with open(test_label_path,'r') as f:
        labels = json.load(f)
    
    with open(args.eval_prompts, 'r', encoding='utf-8') as file:
        prompt = file.read()
    instruction_pool = prompt.split('\n\n')

    question = "\nQuestion: What is the students engagement level?\n"
    with open(args.consistency_qa,'r') as f:
        qa_pairs = json.load(f)

    stop_words_ids = [torch.tensor([2]).to("cuda:{}".format(args.gpu_id))]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    answers = []
    model.eval()
    for subject in labels['annotations']:
        subject_sample = subject['video_id']
        instruct_prompt = random.choice(instruction_pool) 
        instruct_prompt += f"\n\n### Input:\n"
        
        questions = qa_pairs[subject_sample]['Q1'],qa_pairs[subject_sample]['Q2']
        img_list = []
        for image_path in sorted(glob.glob(os.path.join(test_dir,subject_sample[:6],f"{subject_sample}-*.jpg"))):
            image = vis_processor(Image.open(image_path).convert("RGB")).to(device='cuda:{}'.format(args.gpu_id))
            image,_ = model.encode_img(image.unsqueeze(0))
            img_list.append(image)
            instruct_prompt +="<img><ImageHere><\img>"# TODO add index number of frame?

        instruct_prompt += question + "\n### Response:\n"
        
        embs,max_new_tokens = embedding_prepare(model, instruct_prompt, img_list)
        inputs = generate_kwargs(embs=embs, stopping_criteria=stopping_criteria,max_new_tokens=max_new_tokens)
        pred = model_answer(model, inputs)
        
        embs,max_new_tokens = embedding_prepare(model, instruct_prompt, img_list)
        inputs = generate_kwargs(embs=embs, stopping_criteria=stopping_criteria,max_new_tokens=max_new_tokens)
        pred_q1 = model_answer(model, inputs)



        logger.info(f"subject: {subject_sample}\noutput: {pred}\n")
        answers.append({
            "video_id": subject_sample,
            'Q': question.split('Question:')[-1],
            'Q1':question.split('Question:')[-1],
            "pred": pred,
            'pred1':pred,
            'pred2':pred_q1,
            'A': subject['caption'],
        })
        
    with open(f"results/{os.path.splitext(program)[0]}.json", 'w') as f:
        json.dump(answers, f, indent=4)

if __name__ == "__main__":
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()