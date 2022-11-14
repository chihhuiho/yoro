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
from pycocotools.coco import COCO

def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

# https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md
def read_flickr(flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None, max_bbox=None, type="merged", split="train"):

    ann_file = f"{mdetr_anno_path}/final_flickr_{type}GT_{split}.json"
    print("Loading " + ann_file)
    coco = torchvision.datasets.CocoDetection(flickr_img_path, ann_file)
    coco_tool = COCO(ann_file)

    with open(ann_file, "r") as fp:
        annotation = json.load(fp)

    img_id_dict = {}
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])

        _, coco_target = coco.__getitem__(idx)
        filename = annotation["images"][idx]["file_name"][:-4]
        if filename not in img_id_dict:
            img_id_dict[filename] = []

        category_id_lst = []
        bbox_lst = []
        token_positive_lst = []
        w = float(annotation['images'][idx]['width'])
        h = float(annotation['images'][idx]['height'])
        for b in coco_target:
            category_id_lst.append(b['category_id'])
            gt_x, gt_y, gt_w, gt_h = b['bbox']
            gt_x /= w
            gt_y /= h
            gt_w /= w
            gt_h /= h
            bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
            token_positive_lst.append(b['tokens_positive'])
 
        if max_bbox is not None:
            if len(bbox_lst) > max_bbox:
                skip_idx += 1
                continue

        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [[annotation['images'][idx]['caption']], category_id_lst, bbox_lst, token_positive_lst]
            img_id_dict[filename].append(anno)

    for key in img_id_dict:
        if len(img_id_dict[key]) == 0:
            print("QQQ") 



def make_arrow(flickr_img_path, flickr_dataset_path, mdetr_anno_path, dataset_root, max_len=None, max_box=None, type="merged"):

    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
             
    # train
    for split in ["train"]:
        print(f"Processing flickr {type} {split}")
        read_flickr(flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split, type=type)

        '''
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
                f"{dataset_root}/{split}_{type}.arrow", "wb"
            ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        '''
