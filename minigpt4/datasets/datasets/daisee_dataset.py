import os
import random
import glob
import torch
from PIL import Image
import webdataset as wds
from collections import OrderedDict
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class DaiseeDataset(BaseDataset,__DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths,instruct_prompts=None):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool = ""
        self.question = "Question: What is the students level of emotional state?\n"
        if instruct_prompts is not None:
            with open(instruct_prompts, 'r', encoding='utf-8') as file:
                prompt = file.read()
            self.instruction_pool = prompt.split('\n\n')

        exist_annotation = []
        for ann in self.annotation:
            subject,exists = ann['video_id'][:6],True
            for image_path in sorted(glob.glob(os.path.join(self.vis_root,subject,f"{ann['video_id']}-*.jpg"))):
                if not os.path.exists(image_path):
                    exists = False
            if exists:
                exist_annotation.append(ann)
        self.annotation = exist_annotation

    def __getitem__(
        self,
        index:int
    )->dict:
        ann = self.annotation[index]
        subject = ann['video_id'][:6]
        instruction,images = random.choice(self.instruction_pool),[]
        instruction += f"\n### Input:\n"#<img><ImageHere>" 

        # print("WTF")
        # print(os.path.join(self.vis_root,subject,f"{ann['video_id']}-*.jpg"))
        for image_path in sorted(glob.glob(os.path.join(self.vis_root,subject,f"{ann['video_id']}-*.jpg"))):
            image = self.vis_processor(Image.open(image_path).convert("RGB"))
            images.append(image)
            instruction += "<img><ImageHere>"
        # print(len(images))

        instruction += self.question + "### Response:\n"
        images = torch.stack(images)
        return{
            "image": images,
            "answer": ann['caption'],
            "image_id": ann['video_id'],
            "instruction_input": instruction,
        } 