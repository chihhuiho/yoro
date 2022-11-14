import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
import torchvision
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import numpy as np
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
import torch
#import util.dist as dist
import vilt.modules.dist_utils as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

def read_flickr_entity(flickr_img_path, flickr_dataset_path, max_len=None, tokenizer=None, max_bbox=None, split="train"):
    db = []
    imgset_file = "{0}/flickr_{1}.pth".format(flickr_dataset_path, split)
    data = torch.load(imgset_file)

    for d in tqdm(data):
        img_id = d[0]
        image_path = os.path.join(flickr_img_path, img_id)
        
        with open(image_path, "rb") as fp:
            binary = fp.read()
 
        img = plt.imread(image_path)
        w = float(img.shape[0])
        h = float(img.shape[1])
        
        # compute token positive
        token_positive_lst = []
        caption = d[2]
        token_positive_lst.append([[0, len(caption)]])
        #print(split)        
        #print(caption)        

        box = d[1]
        all_bboxes_lst = []
        bbox_lst = []
        # compute all bboxes that the phrase referred to
        gt_x, gt_y, gt_w, gt_h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        gt_x /= w
        gt_y /= h
        gt_w /= w
        gt_h /= h
        all_bboxes_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
        bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))

        ''' 
        fig, ax = plt.subplots()
        img = plt.imread(image_path)
        ax.imshow(img)
        plt.axis("off")
        rect = patches.Rectangle((gt_x*w, gt_y*h), gt_w*w, gt_h*h, linewidth=3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.savefig(f"RRR_{split}.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        ''' 

        category_id_lst = []
        category_id_lst.append(0)
        db.append([img_id[:-4], 0, binary, image_path, w, h, [caption],  category_id_lst, bbox_lst, token_positive_lst, all_bboxes_lst])
 
    return db


def make_arrow(flickr_img_path, flickr_dataset_path, dataset_root, max_len=None, max_box=None):

    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
             
    # train
    for split in ["train", "val", "test"]:
        print(f"Processing flickr {split}")
        
        db = read_flickr_entity(flickr_img_path, flickr_dataset_path,  max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split)
        
        #print(len(db)) 
        dataframe = pd.DataFrame(
                db, columns=["img_id", "phrase_id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "all_gt_bbox"],
          )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
                f"{dataset_root}/{split}.arrow", "wb"
            ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
