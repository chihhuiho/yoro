import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from vilt.utils.refer.refer import REFER
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import spacy
import numpy as np

def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst


def make_arrow(refcoco_root, imageclef_root, dataset_root, use_spacy=False):
    #for split in ["train", "val", "test"]:
    referitgame = REFER(refcoco_root, dataset='refclef', splitBy='unc') # refclef is the same as refer it game

    for split in ["train", "val", "test"]:
        count_img_id = set()
        refer_ids = []
        refer_ids.extend(referitgame.getRefIds(split=split))
        refs = referitgame.loadRefs(ref_ids=refer_ids)
        db = []
        idx = 0 
        for ref_id, ref in tqdm(zip(refer_ids, refs)):
            #if split != "test":
            category_id_lst = []
            bbox_lst = []
            gt_x, gt_y, gt_w, gt_h = referitgame.getRefBox(ref_id=ref_id)
            w = referitgame.Imgs[ref['image_id']]["width"]
            h = referitgame.Imgs[ref['image_id']]["height"]
            gt_x /= w 
            gt_y /= h 
            gt_w /= w
            gt_h /= h
            bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
            category_id_lst.append(ref["category_id"])
            subfolder_name = str(ref['image_id']).zfill(5)[:2]
            image_path = os.path.join(imageclef_root, subfolder_name, '{}.jpg'.format(ref['image_id']))
            
            if ref['image_id'] not in count_img_id:
                count_img_id.add(ref['image_id'])

            with open(image_path, "rb") as fp:
                binary = fp.read()
  
            for sent in ref['sentences']:
                caption = sent["sent"]
                token_positive_lst = []
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
 
                db.append([idx, binary, image_path, w, h, [sent["sent"]],  category_id_lst, bbox_lst, token_positive_lst ])
                idx += 1
        
        print(f"{split} has " + str(len(db)))
        dataframe = pd.DataFrame(
            db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive"],
        )
        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/referitgame_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
        print(len(count_img_id))
