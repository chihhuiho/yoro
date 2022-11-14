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
def read_gqa(gqa_img_path, mdetr_anno_path, max_len=None, tokenizer=None, max_bbox=None, type=None, split=None, separate_arrow=False, arrow_size=500000, dataset_root=None, start_idx=0):
    db = []
    ann_file = f"{mdetr_anno_path}/finetune_gqa_{split}_{type}.json"
    print("Loading " + ann_file)

    with open(f"{mdetr_anno_path}/gqa_answer2id.json", "r") as f:
        answer2id = json.load(f)
    with open(f"{mdetr_anno_path}/gqa_answer2id_by_type.json", "r") as f:
        answer2id_by_type = json.load(f)
    type2id = {"obj": 0, "attr": 1, "rel": 2, "global": 3, "cat": 4}

    coco = torchvision.datasets.CocoDetection(gqa_img_path, ann_file)
    coco_tool = COCO(ann_file)

    with open(ann_file, "r") as fp:
        annotation = json.load(fp)

    incorrect_idx = []
    skip_idx = 0
    cnt = 0
    for idx in tqdm(range(start_idx, len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        # 'file_name', 'height', 'width', 'id', 'original_id', 'caption', 'tokens_negative', 'dataset_name', 'question_type', 'answer', 'questionId'
        questionId = annotation['images'][idx]["questionId"]
        dataset_name = annotation['images'][idx]["dataset_name"]
        if annotation['images'][idx]["answer"] not in answer2id:
            answer = "unknown"
        else:
            answer = annotation['images'][idx]["answer"]
        answer_id = answer2id[answer]
        #print(answer_id)
        answer_type = type2id[annotation['images'][idx]["question_type"]]
        #print(answer_type)

        if annotation['images'][idx]["answer"] not in answer2id_by_type["answer_attr"]:
            answer = "unknown"
        else:
            answer = annotation['images'][idx]["answer"]
        answer_attr = answer2id_by_type["answer_attr"][answer] if annotation['images'][idx]["question_type"] == "attr" else -100
        #print(answer_attr)

        if annotation['images'][idx]["answer"] not in answer2id_by_type["answer_global"]:
            answer = "unknown"
        else:
            answer = annotation['images'][idx]["answer"]
        answer_global = answer2id_by_type["answer_global"][answer] if annotation['images'][idx]["question_type"] == "global" else -100
        #print(answer_global)

        if annotation['images'][idx]["answer"] not in answer2id_by_type["answer_rel"]:
            answer = "unknown"
        else:
            answer = annotation['images'][idx]["answer"]
        answer_rel = answer2id_by_type["answer_rel"][answer] if annotation['images'][idx]["question_type"] == "rel" else -100
        #print(answer_rel)

        if annotation['images'][idx]["answer"] not in answer2id_by_type["answer_cat"]:
            answer = "unknown"
        else:
            answer = annotation['images'][idx]["answer"]
        answer_cat = answer2id_by_type["answer_cat"][answer] if annotation['images'][idx]["question_type"] == "cat" else -100
        #print(answer_cat)

        if annotation['images'][idx]["answer"] not in answer2id_by_type["answer_obj"]:
            answer = "unknown"
        else:
            answer = annotation['images'][idx]["answer"]
        answer_obj = answer2id_by_type["answer_obj"][answer] if annotation['images'][idx]["question_type"] == "obj" else -100
        #print(answer_obj)


        _, coco_target = coco.__getitem__(idx)
        image_id = annotation["images"][idx]["id"]

        image_path = os.path.join(gqa_img_path, annotation['images'][idx]['file_name'])
        
        
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
        
        if max_bbox is not None:
            if len(bbox_lst) > max_bbox:
                skip_idx += 1
                continue

        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']], image_id, answer_id, answer_type, answer_attr, answer_global, answer_rel, answer_cat, answer_obj, questionId]
            db.append(anno)
        elif split != "train" and split != "val":
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']], image_id, answer_id, answer_type, answer_attr, answer_global, answer_rel, answer_cat, answer_obj, questionId]
            db.append(anno)
        else:
            incorrect_idx.append(idx)
        

        if separate_arrow and (idx+1)%arrow_size == 0:
            dataframe = pd.DataFrame(
                   db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative", "image_id", "answer_id", "answer_type", "answer_attr", "answer_global", "answer_rel", "answer_cat", "answer_obj", "questionId"])

            table = pa.Table.from_pandas(dataframe)
            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                   f"{dataset_root}/{split}_{type}_{cnt}.arrow", "wb"
                ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
            print("Saving " + f"{dataset_root}/{split}_{type}_{cnt}.arrow")
            db.clear() 
            cnt += 1 
    if not separate_arrow:
        print("Total " + str(len(db)) + " gqa training pairs added")
        print("Total " + str(len(incorrect_idx)) + " gqa pairs filtered")
        print("Skip " +  str(skip_idx) + " pairs")
    else:
        dataframe = pd.DataFrame(
            db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative", "image_id", "answer_id", "answer_type", "answer_attr", "answer_global", "answer_rel", "answer_cat", "answer_obj", "questionId"])

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
                   f"{dataset_root}/{split}_{type}_{cnt}.arrow", "wb"
                ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        print("Saving " + f"{dataset_root}/{split}_{type}_{cnt}.arrow")
        db.clear() 
        cnt += 1 


    return db


def make_arrow(gqa_img_path, mdetr_anno_path, dataset_root, max_len=None, max_box=None, type="balanced", start_idx=0):

    assert type in ["balanced", "all"]

    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
             
    # train
    for split in ["train"]:#, "val", "testdev"]: # "challenge", "submission"
        print(f"Processing gqa {split} {type}")
        
        if type=="all" and split in ["val", "train"]:
            db = read_gqa(gqa_img_path, mdetr_anno_path, max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split, type=type, separate_arrow=True, dataset_root=dataset_root, start_idx=start_idx)
        else:
            db = read_gqa(gqa_img_path, mdetr_anno_path, max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split, type=type, start_idx=start_idx)
            dataframe = pd.DataFrame(
                db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative", "image_id", "answer_id", "answer_type", "answer_attr", "answer_global", "answer_rel", "answer_cat", "answer_obj", "questionId"])

            table = pa.Table.from_pandas(dataframe)
            os.makedirs(dataset_root, exist_ok=True)
            with pa.OSFile(
                   f"{dataset_root}/{split}_{type}.arrow", "wb"
                ) as sink:
                with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                    writer.write_table(table)
