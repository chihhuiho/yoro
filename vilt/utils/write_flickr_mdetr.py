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
def read_flickr(flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None, max_bbox=None, type="separate", split="train"):
    db = []
    if split == "test":
        max_bbox=None

    ann_file = f"{mdetr_anno_path}/final_flickr_{type}GT_{split}.json"
    print("Loading " + ann_file)
    coco = torchvision.datasets.CocoDetection(flickr_img_path, ann_file)
    coco_tool = COCO(ann_file)

    with open(ann_file, "r") as fp:
        annotation = json.load(fp)

    incorrect_idx = []
    skip_idx = 0

    ids = set()
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        _, coco_target = coco.__getitem__(idx)
        image_id = annotation["images"][idx]["id"]
        image_path = os.path.join(flickr_img_path, annotation['images'][idx]['file_name'])
        
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
            token_positive_lst.append(b['tokens_positive'])
 
            #original_id = b["original_id"] if "original_id" in b else None    
            #task_id = b["task_id"] if "task_id" in b else None   
        sentence_id = annotation['images'][idx]["sentence_id"] if "sentence_id" in annotation['images'][idx] else None    
        original_img_id = annotation['images'][idx]["original_img_id"] if "original_img_id" in annotation['images'][idx] else None    
        tokens_positive_eval = annotation['images'][idx]["tokens_positive_eval"] if "tokens_positive_eval" in annotation['images'][idx] else None 

        cur_id = f"{original_img_id}_{sentence_id}"
        if cur_id in ids:
            print(cur_id)
        else:
            ids.add(cur_id)
        if max_bbox is not None:
            if len(bbox_lst) > max_bbox:
                skip_idx += 1
                continue

        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']], image_id, sentence_id, original_img_id, tokens_positive_eval]
            db.append(anno)
        else:
            incorrect_idx.append(idx)

        print([annotation['images'][idx]['caption']])
        break
    print("Total " + str(len(db)) + " flickr training pairs added")
    print("Total " + str(len(incorrect_idx)) + " flickr pairs filtered")
    print("Skip " +  str(skip_idx) + " pairs")
    return db


def make_arrow(flickr_img_path, flickr_dataset_path, mdetr_anno_path, dataset_root, max_len=None, max_box=None, type="separate"):

    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
             
    # train
    for split in ["train", "val", "test"]:
        print(f"Processing flickr {type} {split}")
        db = read_flickr(flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split, type=type)
        print(len(db)) 
        dataframe = pd.DataFrame(
                db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative", "image_id", "sentence_id", "original_img_id", "tokens_positive_eval"],
          )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
                f"{dataset_root}/{split}_{type}.arrow", "wb"
            ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
