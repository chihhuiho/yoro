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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import spacy
import torch

torch.set_num_threads(1)

def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst


def make_arrow(gqa_img_path, refreasoning_path, dataset_root, max_len=None, max_box=None, use_spacy=False):
    tokenizer = get_pretrained_tokenizer("bert-base-uncased")

    for split in ["train", "val", "test"]:
        with open(f'{refreasoning_path}/{split}_expressions.json', 'r') as json_file:
            data = json.load(json_file)

        db = []
        idx = 0
        skip_idx = 0
        for key in data:
            image_id = data[key]["image_id"]
            image_path = os.path.join(gqa_img_path, image_id + ".jpg")
            caption = data[key]["expression"]
            if max_len is not None:
               tokenized = tokenizer(caption)
               text_length = len(tokenized['input_ids'])
               if text_length > max_len:
                   skip_idx += 1
                   continue
        
            category_id_lst = []
            bbox_lst = []
            token_positive_lst = []
 
            with open(image_path, "rb") as fp:
                binary = fp.read()


            fig, ax = plt.subplots()
            img = plt.imread(image_path)
            ax.imshow(img)
 
            w = img.shape[0]
            h = img.shape[1]
            category_id_lst.append(0)
            gt_x, gt_y, gt_w, gt_h = data[key]['bbox']
            rect = patches.Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.axis("off")
            plt.savefig(f'QQQ_{split}.png', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            print(data[key]['bbox'])
            print(w)
            print(h)
            print(caption)
 
            gt_x /= w
            gt_y /= h
            gt_w /= w
            gt_h /= h
            bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
            
            token_positive_lst.append([[0, len(caption)]])
            idx += 1
            db.append([idx, binary, image_path, w, h, [caption], category_id_lst, bbox_lst, token_positive_lst])
 
            break 

    '''
    idx = 0
    for ref in data["refs"]:
        if ref["valid_flag"] == -1:
            continue
        image_id = ref["image_id"]
        image_path = os.path.join(gqa_img_path, ref["image_id"] + ".jpg")
        caption = ref["sentences"][0]["sent"]
        if max_len is not None:
           tokenized = tokenizer(caption)
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue



        split = ref["split"]
        if split not in db:
            db[split] = []

        with open(image_path, "rb") as fp:
            binary = fp.read()

        category_id_lst = []
        bbox_lst = []
        token_positive_lst = []
        w = float(imgid2aspect[image_id]['width'])
        h = float(imgid2aspect[image_id]['height'])
        category_id_lst.append(ref["category_id"])
        gt_x, gt_y, gt_w, gt_h = ref['box']
        gt_x /= w
        gt_y /= h
        gt_w /= w
        gt_h /= h
        bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
        
        if use_spacy:
            try:
                doc = nlp(caption)
                root_phrase = None
                for np in doc.noun_chunks:
                    root_phrase = np.text
                    break
                word_pos = [] 
                if root_phrase is None:
                    word_pos.append([0, len(caption)])
                else:
                    word_len = 0
                    for word in root_phrase.split(" "):
                        word_pos.append([word_len, word_len+len(word)])
                        word_len = word_len+len(word)+1
                token_positive_lst.append(word_pos)
            except:
                token_positive_lst.append([[0, len(caption)]])
        else:
            token_positive_lst.append([[0, len(caption)]])
        db[split].append([idx, binary, image_path, w, h, [caption], category_id_lst, bbox_lst, token_positive_lst])
        idx += 1
    
    for split in db.keys():
        print(str(len(db[split])) + " samples in " + split)
      
        dataframe = pd.DataFrame(
        db[split], columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)

        with pa.OSFile(
            f"{dataset_root}/copsref_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
    '''
    
