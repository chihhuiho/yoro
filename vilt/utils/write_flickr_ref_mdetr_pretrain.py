import json
import os
import pandas as pd
import pyarrow as pa
import random
import torch
from tqdm import tqdm
from glob import glob
from collections import defaultdict
import torchvision
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import numpy as np
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer, get_pretrained_roberta_tokenizer
from vilt.utils.write_refcoco_refcocop_refcocog_mdetr_all import make_arrow as read_ref_all_train_val
import matplotlib.pyplot as plt

def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

def read_flickr_entity(flickr_img_path, flickr_dataset_path, max_len=None, tokenizer=None, max_bbox=None, split="train"):
    db = []
    imgset_file = "{0}/flickr_{1}.pth".format(flickr_dataset_path, split)
    data = torch.load(imgset_file)
    cnt = 0
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

        category_id_lst = []
        category_id_lst.append(0)
        db.append([cnt , binary, image_path, w, h, [caption],  category_id_lst, bbox_lst, token_positive_lst, []])
        cnt += 1
    return db


def make_arrow(coco_path, flickr_img_path,flickr_dataset_path, mdetr_anno_path, ref_path, dataset_root, max_len=None, max_bbox=None, lang="bert"):

    if max_len is not None:
        if lang == "bert":
            tokenizer = get_pretrained_tokenizer("bert-base-uncased")
        elif lang == "roberta":
            tokenizer = get_pretrained_roberta_tokenizer("roberta-base")
    print("Processing ref all")
    ref_all_train, ref_all_val = read_ref_all_train_val(ref_path, mdetr_anno_path, dataset_root)
    # train
    print("Processing flickr_train_db")
    flickr_train_db = read_flickr_entity(flickr_img_path, flickr_dataset_path,  max_len=max_len, max_bbox=max_bbox, tokenizer=tokenizer, split="train")

    train_db = flickr_train_db + ref_all_train
    print("Total " + str(len(train_db)) + " train pairs")

    
    train_dataframe = pd.DataFrame(
                train_db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative"],
          )

    train_table = pa.Table.from_pandas(train_dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
            f"{dataset_root}/train.arrow", "wb"
        ) as sink:
        with pa.RecordBatchFileWriter(sink, train_table.schema) as writer:
            writer.write_table(train_table)

    # val
    flickr_val_db = read_flickr_entity(flickr_img_path, flickr_dataset_path,  max_len=max_len, max_bbox=max_bbox, tokenizer=tokenizer, split="val")

    val_db = ref_all_val + flickr_val_db
    print("Total " + str(len(val_db)) + " val pairs")
    val_dataframe = pd.DataFrame(
        val_db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative"],
        )
    val_table = pa.Table.from_pandas(val_dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
            f"{dataset_root}/val.arrow", "wb"
        ) as sink:
        with pa.RecordBatchFileWriter(sink, val_table.schema) as writer:
            writer.write_table(val_table)
