'''
	-The given code extracts all the frames for the entire dataset and saves these frames in the folder of the video clips.
	-Kindly have ffmpeg (https://www.ffmpeg.org/) (all credits) in order to successfully execute this script.
	-The script must in the a same directory as the Dataset Folder.

   ffmpeg -i "/home/tony/nvme2tb/DAiSEE/DataSet/Test/videos/500044/5000441001/5000441001.avi" 5000441001%d.jpg -hide_banner

'''

import os
import json
import subprocess
import pandas as pd

def get_daisee_filtercap(
    label_path:str,
    outfile:str
)->dict:
    df = pd.read_csv(label_path,index_col=False)
    labels = {
        'annotations':[]
    }
    mapping = {
        0: 'not',
        1: 'not very',
        2: 'very',
        3: 'very very'
    }
    for idx, row in df.iterrows():
        labels['annotations'].append({
            'video_id':row[0].replace('.avi',''),
            'caption': f"The student is {mapping[row[1]]} bored, {mapping[row[2]]} engaged, {mapping[row[3]]} confused and {mapping[row[-1]]} frustrated."
        })

    with open(os.path.join("daisee_captions",outfile), 'w') as f:
        json.dump(labels,f)

    return labels

def get_weird_samples(
    label_path:str
)->None:
    df = pd.read_csv(label_path,index_col=False)
    print(df.shape)
    df.drop(columns=df.columns[0],inplace=True)
    m = df.to_numpy()
    print("weird samples",m[((m > 0).sum(axis=1) > 1)])
    print("# weird samples",((m > 0).sum(axis=1) > 1).sum())



def split_video(video_file, image_name_prefix, destination_path):
    return subprocess.check_output('ffmpeg -i "' + destination_path+video_file + '" ' + image_name_prefix + '%d.jpg -hide_banner', shell=True, cwd=destination_path)

if __name__ == "__main__":
    ann_path = '/home/tony/nvme2tb/daisee_annotations'
    test_labels = '/home/tony/nvme2tb/DAiSEE/Labels/TestLabels.csv'
    val_labels = '/home/tony/nvme2tb/DAiSEE/Labels/TrainLabels.csv'
    train_labels = '/home/tony/nvme2tb/DAiSEE/Labels/ValidationLabels.csv'
    
    original_labels = '/home/tony/nvme2tb/DAiSEE/Labels'
    get_weird_samples(os.path.join(original_labels,"AllLabels.csv"))
    get_daisee_filtercap(test_labels,"test_filter_cap.json")
    get_daisee_filtercap(val_labels,"val_filter_cap.json")
    get_daisee_filtercap(train_labels,"train_filter_cap.json")

    # dataset = os.listdir('/home/tony/nvme2tb/DAiSEE/DataSet')
    # split_video(
    #     video_file='/home/tony/nvme2tb/DAiSEE/DataSet/Test/videos/500044/5000441001/5000441001.avi',
    #     image_name_prefix='5000441001',
    #     destination_path=''
    # )

    # print ("================================================================================\n")
    # print ("Frame Extraction Successful")
