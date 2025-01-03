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
            'video_id':row[0].replace('.avi','').replace('.mp4',''),
            'caption': f"Boredom: {mapping[row[1]]}\nEngagement: {mapping[row[2]]}\nConfusion: {mapping[row[3]]}\nFrustration: {mapping[row[-1]]}\n"
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
    ann_path = '/home/tony/nvme2tb/DAiSEE/Labels/AllLabels.csv'
    test_samples = '/home/tony/nvme2tb/DAiSEE/Labels/TestLabels.csv'
    val_samples = '/home/tony/nvme2tb/DAiSEE/Labels/ValidationLabels.csv'
    train_samples = '/home/tony/nvme2tb/DAiSEE/Labels/TrainLabels.csv'
    train_val_samples = '/home/tony/nvme2tb/DAiSEE/Labels/TrainValidation.csv'
    train = pd.read_csv(train_samples,index_col=False)
    val = pd.read_csv(val_samples,index_col=False)
    train_val = pd.concat([train,val],axis=0)
    train_val.to_csv(train_val_samples,index=False)
    
    get_weird_samples(ann_path)
    get_daisee_filtercap(test_samples,"test_filter_cap.json")
    get_daisee_filtercap(val_samples,"val_filter_cap.json")
    get_daisee_filtercap(train_samples,"train_filter_cap.json")

    train = pd.read_csv(train_samples,index_col=False)
    val = pd.read_csv(val_samples,index_col=False)
    train_val = pd.concat([train,val],axis=0)
    train_val.to_csv(train_val_samples,index=False)
    get_daisee_filtercap(train_val_samples,"train_val_filter_cap.json")


    # dataset = os.listdir('/home/tony/nvme2tb/DAiSEE/DataSet')
    # split_video(
    #     video_file='/home/tony/nvme2tb/DAiSEE/DataSet/Test/videos/500044/5000441001/5000441001.avi',
    #     image_name_prefix='5000441001',
    #     destination_path=''
    # )

    # print ("================================================================================\n")
    # print ("Frame Extraction Successful")
