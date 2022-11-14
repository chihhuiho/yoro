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


ALL_ATTRIBUTES = [
    "small",
    "large",
    "gray",
    "red",
    "blue",
    "green",
    "brown",
    "purple",
    "cyan",
    "yellow",
    "cube",
    "sphere",
    "cylinder",
    "rubber",
    "metal",
]


def _encode_answer(answer):
    target = {}
    if answer in ["yes", "no"]:
        target["answer_type"] = 0
        target["answer_binary"] = 0 if answer == "no" else 1
        target["answer_attr"] = -100
        target["answer_reg"] = -100
    elif answer in ALL_ATTRIBUTES:
        target["answer_type"] = 1
        target["answer_binary"] = 0.0
        target["answer_attr"] = ALL_ATTRIBUTES.index(answer)
        target["answer_reg"] = -100
    else:
        target["answer_type"] = 2
        target["answer_binary"] = 0
        target["answer_attr"] = -100
        target["answer_reg"] = int(answer)
    return target



def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

# https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md
def read_clever(clever_path, max_len=None, tokenizer=None, max_bbox=None, split=None, type=None, dataset_root=None, start_idx=0):
    db = []
    print(f"==========={split}==========")
    image_folder = f"{clever_path}/CLEVR_v1.0/images/{split}"
    ann_file = f"{clever_path}/clevr_annotations/{type}/{split}.json"
    print("Loading " + ann_file)

    coco = torchvision.datasets.CocoDetection(image_folder, ann_file)
    coco_tool = COCO(ann_file)

    with open(ann_file, "r") as fp:
        annotation = json.load(fp)

    incorrect_idx = []
    skip_idx = 0
    max_num_box = 0
    for idx in tqdm(range(start_idx, len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        _, coco_target = coco.__getitem__(idx)

        image_id = annotation["images"][idx]["id"]
        image_path = os.path.join(image_folder, annotation['images'][idx]['file_name'])
        
        
        with open(image_path, "rb") as fp:
            binary = fp.read()

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
        token_positive_lst.append(b['tokens'])

        max_num_box = max(max_num_box, len(bbox_lst))
        if max_bbox is not None:
            if len(bbox_lst) > max_bbox:
                skip_idx += 1
                continue

        target = _encode_answer(annotation["images"][idx]["answer"])
        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, image_id, target['answer_type'], target['answer_attr'], target['answer_binary'], target['answer_reg']]
            db.append(anno)
        else:
            incorrect_idx.append(idx)
            
    print("Max # of box is " + str(max_num_box))
    return db


def make_arrow(clever_path, dataset_root, max_len=None, max_box=None, type="full", start_idx=0):

    assert type in ["full", "medium", "clevrref", "cogent_full", "cogent_medium"]

    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
    dataset_root = dataset_root + "/" + type
    print(dataset_root)
    for split in ["train", "val"]: 
        print(f"Processing clever {split} {type}")
        
        db = read_clever(clever_path, max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split, type=type)
        dataframe = pd.DataFrame(
                db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "image_id", "answer_type", "answer_attr", "answer_binary", "answer_reg"])

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
               f"{dataset_root}/{split}.arrow", "wb"
            ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
