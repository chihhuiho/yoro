import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def make_arrow(refcocop_root, mdetr_anno_root, dataset_root ):
    for split in ["test", "train", "val"]:
        with open(f"{mdetr_anno_root}/finetune_refcocog_{split}.json", "r") as fp:
            mdetr_annotation = json.load(fp)
        #print(mdetr_annotation.keys()) #['info', 'licenses', 'images', 'annotations', 'categories']

        #print(mdetr_annotation['images'][0].keys())
        #dict_keys(['file_name', 'height', 'width', 'id', 'original_id', 'caption', 'dataset_name', 'tokens_negative'])
        #print(mdetr_annotation['annotations'][0].keys())
        #dict_keys(['area', 'iscrowd', 'image_id', 'category_id', 'id', 'bbox', 'original_id', 'tokens_positive'])

        db = []
        for idx in tqdm(range(len(mdetr_annotation['images']))):
            #print(mdetr_annotation['images'][idx]['file_name'])
            image_path = os.path.join(refcocop_root, "train2014", mdetr_annotation['images'][idx]['file_name'])
            with open(image_path, "rb") as fp:
                binary = fp.read()
            gt_x, gt_y, gt_w, gt_h = mdetr_annotation['annotations'][idx]['bbox']
            w = mdetr_annotation['images'][idx]['width']
            h = mdetr_annotation['images'][idx]['height']
            gt_x /= w
            gt_y /= h
            gt_w /= w
            gt_h /= h
            db.append([idx, binary, image_path, w, h, [mdetr_annotation['images'][idx]['caption']],  mdetr_annotation['annotations'][idx]['category_id'], [[gt_x, gt_y, gt_w, gt_h]], [mdetr_annotation['annotations'][idx]['tokens_positive']], [mdetr_annotation['images'][idx]['tokens_negative']]])


        print(f"{split} has " + str(len(db)))
        dataframe = pd.DataFrame(
            db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "tokens_negative"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/refcocog_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

