import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
import numpy as np

def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

def compute_train_test_overlap_img_id(refcocop_root, mdetr_anno_root):
    train_set, test_set = set(), set()
    for split in ["testA", "testB", "train", "val"]:
        with open(f"{mdetr_anno_root}/finetune_refcoco_{split}.json", "r") as fp:
            mdetr_annotation = json.load(fp)
        for idx in range(len(mdetr_annotation['images'])):
            if split == "train":
                train_set.add(mdetr_annotation['images'][idx]['file_name'])
            else:
                test_set.add(mdetr_annotation['images'][idx]['file_name'])

    for split in ["testA", "testB", "train", "val"]:
        with open(f"{mdetr_anno_root}/finetune_refcoco+_{split}.json", "r") as fp:
            mdetr_annotation = json.load(fp)
        for idx in range(len(mdetr_annotation['images'])):
            if split == "train":
                train_set.add(mdetr_annotation['images'][idx]['file_name'])
            else:
                test_set.add(mdetr_annotation['images'][idx]['file_name'])

    for split in ["test", "train", "val"]:
        with open(f"{mdetr_anno_root}/finetune_refcocog_{split}.json", "r") as fp:
            mdetr_annotation = json.load(fp)
        for idx in range(len(mdetr_annotation['images'])):
            if split == "train":
                train_set.add(mdetr_annotation['images'][idx]['file_name'])
            else:
                test_set.add(mdetr_annotation['images'][idx]['file_name'])
    intersect = train_set.intersection(test_set)
    print(len(train_set-intersect)) 
    return intersect

def make_arrow(refcocop_root, mdetr_anno_root, dataset_root ):
    intersect = compute_train_test_overlap_img_id(refcocop_root, mdetr_anno_root)
    
    train_db_all, val_db_all = [], []
    for dataset in ["refcoco", "refcoco+", "refcocog"]:
        train_db, val_db = get_db(refcocop_root, mdetr_anno_root, dataset_root, dataset, intersect)
        train_db_all = train_db_all + train_db
        val_db_all = val_db_all + val_db
    
    return train_db_all, val_db_all
    '''
    train_dataframe = pd.DataFrame(
           train_db_all, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative"],
        )
    val_dataframe = pd.DataFrame(
           val_db_all, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative"],
        )


    print("Total " + str(len(train_db_all)) + " training pairs")
    print("Total " + str(len(val_db_all)) + " validation pairs")

    train_table = pa.Table.from_pandas(train_dataframe)
    val_table = pa.Table.from_pandas(val_dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(
      f"{dataset_root}/train.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, train_table.schema) as writer:
                writer.write_table(train_table)
    with pa.OSFile(
      f"{dataset_root}/val.arrow", "wb"
    ) as sink:
        with pa.RecordBatchFileWriter(sink, val_table.schema) as writer:
                writer.write_table(val_table)
    '''

def get_db(refcocop_root, mdetr_anno_root, dataset_root, dataset, intersect):
    train_db, val_db = [], []
    for split in ["train", "val"]:
        json_file = f"{mdetr_anno_root}/finetune_{dataset}_{split}.json"
        with open(json_file, "r") as fp:
            mdetr_annotation = json.load(fp)
        for idx in tqdm(range(len(mdetr_annotation['images']))):
            if split == "train" and mdetr_annotation['images'][idx]['file_name'] in intersect:
                continue
            image_path = os.path.join(refcocop_root, "train2014", mdetr_annotation['images'][idx]['file_name'])
            caption = mdetr_annotation['images'][idx]['caption']
            if split == "train":
                train_db.append(image_path + mdetr_annotation['images'][idx]['caption'])
            elif split == "val":
                val_db.append(image_path + mdetr_annotation['images'][idx]['caption'])

            #break
    return train_db, val_db
