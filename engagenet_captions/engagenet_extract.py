'''
	-The given code extracts all the frames for the entire dataset and saves these frames in the folder of the video clips.
	-Kindly have ffmpeg (https://www.ffmpeg.org/) (all credits) in order to successfully execute this script.
	-The script must in the a same directory as the Dataset Folder.

   ffmpeg -i "/home/tony/nvme2tb/DAiSEE/DataSet/Test/videos/500044/5000441001/5000441001.avi" 5000441001%d.jpg -hide_banner

'''
import glob
import os
import re
import json
import subprocess
import pandas as pd

# import torch
# import torchmetrics
# from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score


def get_daisee_filtercap(
    label_path:str,
    outfile:str
)->dict:
    df = pd.read_csv(label_path,index_col=False)
    labels = {
        'annotations':[]
    }
    engaged_map = {
        0: 'Not-Engaged',
        1: 'Barely-Engaged',
        2: 'Engaged',
        3: 'Highly-Engaged'
    }

    for idx, row in df.iterrows():
        labels['annotations'].append({
            'video_id':row[0].replace('.avi','').replace('.mp4',''),
            'caption': f"The student is {row[1]}"
        })


    with open(os.path.join("/home/tony/MiniGPT-4/engagenet_captions",outfile), 'w') as f:
        json.dump(labels,f)

    return labels

def prepare_hf_dataset(
    label:str,
    outpath:str,
    train_test_split:str
)->None:
    labels = pd.read_csv(label,index_col=False)
    if not os.path.exists(os.path.join(outpath,'hf_dataset')):
        os.mkdir(os.path.join(outpath,'hf_dataset'))
    if not os.path.exists(os.path.join(outpath,'hf_dataset',train_test_split)):
        os.mkdir(os.path.join(outpath,'hf_dataset',train_test_split))
        os.mkdir(os.path.join(outpath,'hf_dataset',train_test_split,'not_engaged'))
        os.mkdir(os.path.join(outpath,'hf_dataset',train_test_split,'barely_engaged'))
        os.mkdir(os.path.join(outpath,'hf_dataset',train_test_split,'engaged'))
        os.mkdir(os.path.join(outpath,'hf_dataset',train_test_split,'highly_engaged'))

    for idx, row in labels.iterrows():
        sample = row[0].replace('.avi','').replace('.mp4','')
        sample_paths = sorted(glob.glob(os.path.join(outpath,train_test_split,f"{sample}-*.jpg")))
        for path in sample_paths:
            if row[1] == 'not-engaged':
                subprocess.call(['cp',path,os.path.join(outpath,'hf_dataset',train_test_split,'not_engaged')])
            elif row[1] == 'barely-engaged':
                subprocess.call(['cp',path,os.path.join(outpath,'hf_dataset',train_test_split,'barely_engaged')])
            elif row[1] == 'engaged':
                subprocess.call(['cp',path,os.path.join(outpath,'hf_dataset',train_test_split,'engaged')])
            elif row[1] == 'highly-engaged':
                subprocess.call(['cp',path,os.path.join(outpath,'hf_dataset',train_test_split,'highly_engaged')])
    
    return

def check_string_in_output(
    output:str, 
    search_string:str
)->bool:
    # Escape special characters in search_string if necessary
    output,search = re.sub(r'\W', '', output).lower(),re.sub(r'\W', '', search_string).lower()
    pattern = re.escape(search)
    match = re.search(pattern, output)
    return bool(match)
'''
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
    classes:int=4
)->dict:
    mapping = {
        'The student is Not-Engaged':0,
        'The student is Barely-Engaged':1,
        'The student is Engaged':2,
        'The student is Highly-Engaged':3
    }
    metrics = load_metrics(classes)

    inference_samples = len(results)
    print(f"INFERENCE SAMPLES - {inference_samples}")
    pred_table,target_table = torch.zeros(inference_samples),torch.zeros(inference_samples)
    count = 0
    for i,sample in enumerate(results):
        answer,pred = sample['A'],sample['pred']
        count += 1
        target_table[i] = mapping[answer]
        pred_table[i] = target_table[i]
        if not check_string_in_output(pred,answer.split(' ')[-1]):
            pred_table[i] = (target_table[i] - 1) % classes   
            count -= 1
            print(f"WRONG pred {sample['video_id']}")
            print("pred - ",re.sub(r'\W', '', pred).lower())
            print("answer - ",re.sub(r'\W', '', answer.split(' ')[-1]).lower())
            print(re.sub(r'\W', '', answer.split(' ')[-1]).lower() in re.sub(r'\W', '', pred).lower())
        performance = metrics.forward(pred_table[:i + 1],target_table[:i + 1])


    performance = metrics.compute()
    print(f"FINAL ACC - {performance['MulticlassAccuracy']}")
    print(f"FINAL PR - {performance['MulticlassPrecision']}")
    print(f"FINAL RE - {performance['MulticlassRecall']}")
    print(f"FINAL F1 - {performance['MulticlassF1Score']}")
    print(f"FINAL COUNT ACC - {count/inference_samples}")
    metrics.reset()   
    return
'''
def main()->None:
    # ann_path = '/home/tony/nvme2tb/DAiSEE/Labels/AllLabels.csv'
    test_samples = '/home/tony/nvme2tb/EngageNetFrames/en_test_labels.csv'
    val_samples = '/home/tony/nvme2tb/EngageNetFrames/en_val_labels.csv'
    train_samples = '/home/tony/nvme2tb/EngageNetFrames/en_train_labels.csv'
    train_val_samples = '/home/tony/nvme2tb/EngageNetFrames/en_trainval_labels.csv'

    get_daisee_filtercap(test_samples,"test_filter_cap.json")
    get_daisee_filtercap(val_samples,"val_filter_cap.json")
    get_daisee_filtercap(train_samples,"train_filter_cap.json")

    train = pd.read_csv(train_samples,index_col=False)
    val = pd.read_csv(val_samples,index_col=False)
    train_val = pd.concat([train,val],axis=0)
    train_val.to_csv(train_val_samples,index=False)
    get_daisee_filtercap(train_val_samples,"train_val_filter_cap.json")

    # prepare_hf_dataset(train_val_samples,'/home/tony/nvme2tb/EngageNetFrames','train')
    # prepare_hf_dataset(val_samples,'/home/tony/nvme2tb/EngageNetFrames','val')
    # prepare_hf_dataset(test_samples,'/home/tony/nvme2tb/EngageNetFrames','test')
    prepare_hf_dataset(train_val_samples,'/home/tony/nvme2tb/EngageNetFrames','train_val')


    # results_path = '/home/tony/MiniGPT-4/results/daisee_inference.json'
    # with open(results_path,'r') as f:
    #     results = json.load(f)
    # get_acc(results)

if __name__ == "__main__":
    main()

