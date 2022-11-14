#!/usr/bin/env python

'''
SNLI-VE Generator

Authors: Ning Xie, Farley Lai(farleylai@nec-labs.com)

# Copyright (C) 2020 NEC Laboratories America, Inc. ("NECLA"). 
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
'''

import os
import jsonlines
from collections import defaultdict, OrderedDict
import pandas as pd
import pyarrow as pa
import json
import torchvision
from torchvision.datasets.vision import VisionDataset
from typing import Any, Callable, Optional, Tuple
import numpy as np
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image

def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst



def prepare_all_data(SNLI_root, SNLI_files):
    '''
    This function will prepare the recourse towards generating SNLI-VE dataset

    :param SNLI_root: root for SNLI dataset
    :param SNLI_files: original SNLI files, which can be downloaded via
                       https://nlp.stanford.edu/projects/snli/snli_1.0.zip

    :return:
        all_data: a set of data containing all split of SNLI dataset
        image_index_dict: a dict, key is a Flickr30k imageID, value is a list of data indices w.r.t. a Flickr30k imageID
    '''

    data_dict = {}
    for data_type, filename in SNLI_files.items():
        filepath = os.path.join(SNLI_root, filename)
        data_list = []
        with jsonlines.open(filepath) as jsonl_file:
            for line in jsonl_file:
                pairID = line['pairID']
                gold_label = line['gold_label']
                # only consider Flickr30k (pairID.find('vg_') == -1) items whose gold_label != '-'
                if gold_label != '-' and pairID.find('vg_') == -1:
                    imageId = pairID[:pairID.rfind('.jpg')] # XXX Removed suffix: '.jpg'
                    # Add Flikr30kID to the dataset
                    line['Flickr30K_ID'] = imageId
                    line = OrderedDict(sorted(line.items()))
                    data_list.append(line)
        data_dict[data_type] = data_list

    # all_data contains all lines in the original jsonl file
    all_data = data_dict['train'] + data_dict['dev'] + data_dict['test']

    # image_index_dict = {image:[corresponding line index in data_all]}
    image_index_dict = defaultdict(list)
    for idx, line in enumerate(all_data):
        pairID = line['pairID']
        imageID = pairID[:pairID.find('.jpg')]
        image_index_dict[imageID].append(idx)

    return all_data, image_index_dict



def _split_data_helper(image_list, image_index_dict):
    '''
    This will generate a dict for a data split (train/dev/test).
    key is a Flickr30k imageID, value is a list of data indices w.r.t. a Flickr30k imageID

    :param image_list: a list of Flickr30k imageID for a data split (train/dev/test)
    :param image_index_dict: a dict of format {ImageID: a list of data indices}, generated via prepare_all_data()

    :return: a dict of format {ImageID: a lost of data indices} for a data split (train/dev/test)
    '''
    ordered_dict = OrderedDict()
    for imageID in image_list:
        ordered_dict[imageID] = image_index_dict[imageID]
    return ordered_dict


def split_data(all_data, image_index_dict, split_root, split_files, SNLI_VE_root, SNLI_VE_files):
    '''
    This function is to generate SNLI-VE dataset based on SNLI dataset and Flickr30k split.
    The files are saved to paths defined by `SNLI_VE_root` and `SNLI_VE_files`

    :param all_data: a set of data containing all split of SNLI dataset, generated via prepare_all_data()
    :param image_index_dict: a dict of format {ImageID: a list of data indices}, generated via prepare_all_data()
    :param split_root: root for Flickr30k split
    :param split_files: Flickr30k split list files
    :param SNLI_VE_root: root to save generated SNLI-VE dataset
    :param SNLI_VE_files: filenames of generated SNLI-VE dataset for each split (train/dev/test)
    '''
    print('\n*** Generating data split using SNLI dataset and Flickr30k split files ***')
    with open(os.path.join(split_root, split_files['test'])) as f:
        content = f.readlines()
        test_list = [x.strip() for x in content]
    with open(os.path.join(split_root, split_files['train_val'])) as f:
        content = f.readlines()
        train_val_list = [x.strip() for x in content]
    train_list = train_val_list[:-1000]
    # select the last 1000 images for dev dataset
    dev_list = train_val_list[-1000:]

    train_index_dict = _split_data_helper(train_list, image_index_dict)
    dev_index_dict = _split_data_helper(dev_list, image_index_dict)
    test_index_dict = _split_data_helper(test_list, image_index_dict)

    all_index_dict = {'train': train_index_dict, 'dev': dev_index_dict, 'test': test_index_dict}
    # # Write jsonl files
    for data_type, data_index_dict in all_index_dict.items():
        print('Current processing data split : {}'.format(data_type))
        with jsonlines.open(os.path.join(SNLI_VE_root, SNLI_VE_files[data_type]), mode='w') as jsonl_writer:
            for _, index_list in data_index_dict.items():
                for idx in index_list:
                    jsonl_writer.write(all_data[idx])


def read_flickr_trainval(flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=40, tokenizer=None, max_bbox=None,  snlive_anno_dict=None, no_flickr=False):
    db = []
    ann_file_train = f"{mdetr_anno_path}/final_flickr_separateGT_train.json"
    with open(ann_file_train, "r") as fp:
        annotation_train = json.load(fp)
    ann_file_val = f"{mdetr_anno_path}/final_flickr_separateGT_val.json"
    with open(ann_file_val, "r") as fp:
        annotation_val = json.load(fp)

    imagename2idx = {} 
    for idx in tqdm(range(len(annotation_train['images']))):
        image_name = annotation_train["images"][idx]['file_name'].split(".")[0]
        imagename2idx[image_name] = {}
        imagename2idx[image_name]["idx"] = idx
        imagename2idx[image_name]["split"] = "train"

    for idx in tqdm(range(len(annotation_val['images']))):
        image_name = annotation_val["images"][idx]['file_name'].split(".")[0]
        imagename2idx[image_name] = {}
        imagename2idx[image_name]["idx"] = idx
        imagename2idx[image_name]["split"] = "val"


    db = [] 
    coco_train = torchvision.datasets.CocoDetection(flickr_img_path, ann_file_train)
    coco_val = torchvision.datasets.CocoDetection(flickr_img_path, ann_file_val)

    max_box_num = 0
    for image_name in tqdm(snlive_anno_dict.keys()):
        if image_name not in imagename2idx:
            continue

        idx = imagename2idx[image_name]["idx"]
        annotation = annotation_train if imagename2idx[image_name]["split"] == "train" else annotation_val
        coco = coco_train if imagename2idx[image_name]["split"] == "train" else coco_val
  
        # original flicker caption
        tokenized = tokenizer(annotation['images'][idx]['caption'])
        text_length = len(tokenized['input_ids'])
        if (max_len is None and not no_flickr) or (text_length < max_len and not no_flickr):
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
           
            max_box_num = max(max_box_num, len(coco_target))
            snlive_label = 0
            has_bbox = 1

            if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
                anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, snlive_label, has_bbox]
                db.append(anno)

        tokenized = tokenizer(snlive_anno_dict[image_name]["snlive_input"])
        text_length = len(tokenized['input_ids'])
        if max_len is None or text_length < max_len:
            image_id = annotation["images"][idx]["id"]
            image_path = os.path.join(flickr_img_path, annotation['images'][idx]['file_name'])
        
            with open(image_path, "rb") as fp:
                binary = fp.read()

            category_id_lst = []
            bbox_lst = []
            token_positive_lst = []
            w = float(annotation['images'][idx]['width'])
            h = float(annotation['images'][idx]['height'])
            snlive_label = snlive_anno_dict[image_name]["snlive_label"]
            has_bbox = 0

            anno = [idx, binary, image_path, w, h, [snlive_anno_dict[image_name]["snlive_input"]],  category_id_lst, bbox_lst, token_positive_lst, snlive_label, has_bbox]
            db.append(anno)
    print("Max box " + str(max_box_num))
    return db

def read_flickr_test(flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=40, tokenizer=None, max_bbox=None,  snlive_anno_dict=None):
    db = []
 
    idx = 0
    for image_name in tqdm(snlive_anno_dict.keys()):
        tokenized = tokenizer(snlive_anno_dict[image_name]["snlive_input"])
        text_length = len(tokenized['input_ids'])
        if max_len is None or text_length < max_len:
            image_id = int(image_name)
            image_path = os.path.join(flickr_img_path, image_name + ".jpg")
        
            with open(image_path, "rb") as fp:
                binary = fp.read()

            category_id_lst = []
            bbox_lst = []
            token_positive_lst = []
            with Image.open(image_path) as im:
                w = im.size[0]
                h = im.size[1]

            snlive_label = snlive_anno_dict[image_name]["snlive_label"]
            has_bbox = 0

            anno = [idx, binary, image_path, w, h, [snlive_anno_dict[image_name]["snlive_input"]],  category_id_lst, bbox_lst, token_positive_lst, snlive_label, has_bbox]
            db.append(anno)
            idx += 1
    return db


def make_arrow(snlive_root, flickr_img_path, flickr_dataset_path, mdetr_anno_path, dataset_root, max_len=40, tokenizer=None, max_bbox=None, no_flickr=False):
    
    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
    # SNLI-VE generation resource: SNLI dataset
    SNLI_root = snlive_root + '/snli_1.0'
    SNLI_files = {'dev': 'snli_1.0_dev.jsonl',
                  'test': 'snli_1.0_test.jsonl',
                  'train': 'snli_1.0_train.jsonl'}

    # SNLI-VE generation resource: Flickr30k file lists
    split_root = snlive_root + "/SNLI-VE/data"
    split_files = {'test': 'flickr30k_test.lst',
                   'train_val': 'flickr30k_train_val.lst'}

    # SNLI-VE generation destination
    SNLI_VE_root = snlive_root
    SNLI_VE_files = {'dev': 'snli_ve_dev.jsonl',
                     'test': 'snli_ve_test.jsonl',
                     'train': 'snli_ve_train.jsonl'}

    print('*** SNLI-VE Generation Start! ***')
    all_data, image_index_dict = prepare_all_data(SNLI_root, SNLI_files)
    split_data(all_data, image_index_dict, split_root, split_files, SNLI_VE_root, SNLI_VE_files)
    print('*** SNLI-VE Generation Done! ***')

    label2ID = {"entailment":0, "neutral":1, "contradiction":2}

    for split in ["train", "dev", "test"]:
        snlive_anno_dict = {}
        with jsonlines.open(SNLI_VE_root + f"/snli_ve_{split}.jsonl") as jsonl_file:
            for line in jsonl_file:
                flickrid = line['Flickr30K_ID']
                #caption1 = line['sentence1'] # ori_caption
                caption2 = line['sentence2'] 
                label = label2ID[line['gold_label']] 
                snlive_anno_dict[flickrid] = {}
                snlive_anno_dict[flickrid]["snlive_input"] = caption2
                snlive_anno_dict[flickrid]["snlive_label"] = label
                # ['Flickr30K_ID', 'annotator_labels', 'captionID', 'gold_label', 'pairID', 'sentence1', 'sentence1_binary_parse', 'sentence1_parse', 'sentence2', 'sentence2_binary_parse', 'sentence2_parse']
        if split == "train":
            print(f"Processing {split}")
            db =read_flickr_trainval(flickr_img_path=flickr_img_path, flickr_dataset_path=flickr_dataset_path, mdetr_anno_path=mdetr_anno_path, max_len=max_len, tokenizer=tokenizer, max_bbox=max_bbox, snlive_anno_dict=snlive_anno_dict, no_flickr=no_flickr)
        else:    
            print(f"Processing {split}")
            db = read_flickr_test(flickr_img_path=flickr_img_path, flickr_dataset_path=flickr_dataset_path, mdetr_anno_path=mdetr_anno_path, max_len=max_len, tokenizer=tokenizer, max_bbox=max_bbox, snlive_anno_dict=snlive_anno_dict)

        dataframe = pd.DataFrame(
                db, columns=["id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "snlive_label", "has_bbox"],
          )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        if not no_flickr or split != "train":
            filename = f"{dataset_root}/{split}.arrow"
        elif no_flickr and split == "train":
            filename = f"{dataset_root}/{split}_no_flicker.arrow"
        with pa.OSFile(
                filename, "wb"
            ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)           
