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
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer, get_pretrained_roberta_tokenizer
from vilt.utils.write_refcoco_refcocop_refcocog_mdetr_all import make_arrow as read_ref_all_train_val

class CustomCocoDetection(VisionDataset):
    """Coco-style dataset imported from TorchVision.
    It is modified to handle several image sources


    Args:
        root_coco (string): Path to the coco images
        root_vg (string): Path to the vg images
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root_coco: str,
        root_vg: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(CustomCocoDetection, self).__init__(root_coco, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root_coco = root_coco
        self.root_vg = root_vg

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        img_info = coco.loadImgs(img_id)[0]
        path = img_info["file_name"]
        dataset = img_info["data_source"]

        cur_root = self.root_coco if dataset == "coco" else self.root_vg
        #img = Image.open(os.path.join(cur_root, path)).convert("RGB")
        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)

        #return img, target
        return target

    def __len__(self) -> int:
        return len(self.ids)


def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

# https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md
def read_flickr_separateGT_train(coco_path, vg_img_path, flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None, max_bbox=None):
    db = []
    ann_file = f"{mdetr_anno_path}/final_flickr_separateGT_train.json"
    coco = torchvision.datasets.CocoDetection(flickr_img_path, ann_file)
    with open(ann_file, "r") as fp:
        annotation = json.load(fp)

    incorrect_idx = []
    skip_idx = 0
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        _, coco_target = coco.__getitem__(idx)
        
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

        if max_bbox is not None:
            if len(bbox_lst) > max_bbox:
                skip_idx += 1
                continue

        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']]]
            db.append(anno)
        else:
            incorrect_idx.append(idx)
        
    print("Total " + str(len(db)) + " flickr training pairs added")
    print("Total " + str(len(incorrect_idx)) + " flickr training pairs filtered")
    print("Skip " +  str(skip_idx) + " pairs")
    return db

def read_mix_train(coco_path, vg_img_path, flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None, max_bbox=None):
    db = []
    ann_file = f"{mdetr_anno_path}/final_mixed_train.json"
    coco = CustomCocoDetection(coco_path, vg_img_path, ann_file)
    with open(ann_file, "r") as fp:
        annotation = json.load(fp)
    incorrect_idx = []
    skip_idx = 0
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        coco_target = coco.__getitem__(idx)
 
        if "COCO" in annotation['images'][idx]['file_name']:
            image_path = os.path.join(coco_path,"train2014", annotation['images'][idx]['file_name'])
        else:
            image_path = os.path.join(vg_img_path, annotation['images'][idx]['file_name'])
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
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']]]
            db.append(anno)
        else:
            incorrect_idx.append(idx)

    print("Total " + str(len(db)) + " mixed training pairs added")
    print("Total " + str(len(incorrect_idx)) + " mixed training pairs filtered")
    print("Skip " +  str(skip_idx) + " pairs")
 
    return db


# https://github.com/ashkamath/mdetr/blob/main/.github/pretrain.md
def read_flickr_separateGT_val(coco_path, vg_img_path, flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None):
    db = []
    ann_file = f"{mdetr_anno_path}/final_flickr_separateGT_val.json"
    coco = torchvision.datasets.CocoDetection(flickr_img_path, ann_file)
    with open(ann_file, "r") as fp:
        annotation = json.load(fp)
    incorrect_idx = []
    skip_idx = 0
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        _, coco_target = coco.__getitem__(idx)

        imgID = annotation['images'][idx]["original_img_id"]
        sentID = annotation['images'][idx]["sentence_id"] 
        
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
            
        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']]]
            db.append(anno)
        else:
            incorrect_idx.append(idx)

    print("Total " + str(len(db)) + " flickr val pairs added")
    print("Total " + str(len(incorrect_idx)) + " flickr val pairs filtered")
    print("Skip " +  str(skip_idx) + " pairs")
 
    return db

   

def read_refexp_val(coco_path, vg_img_path, flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None):
    db = []
    ann_file = f"{mdetr_anno_path}/final_refexp_val.json"
    coco = torchvision.datasets.CocoDetection(coco_path + "/train2014", ann_file)
    with open(ann_file, "r") as fp:
        annotation = json.load(fp)
    incorrect_idx = []
    skip_idx = 0
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue

        _, coco_target = coco.__getitem__(idx)
        
        image_path = os.path.join(coco_path, "train2014" ,annotation['images'][idx]['file_name'])
        
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

        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']]]
            db.append(anno)
        else:
            incorrect_idx.append(idx)

    print("Total " + str(len(db)) + " ref val pairs added")
    print("Total " + str(len(incorrect_idx)) + " ref val pairs filtered")
    print("Skip " +  str(skip_idx) + " pairs")
 

    return db


def read_gqa_val(coco_path, vg_img_path, flickr_img_path, flickr_dataset_path, mdetr_anno_path, max_len=None, tokenizer=None):
    db = []
    ann_file = f"{mdetr_anno_path}/final_gqa_val.json"
    coco = torchvision.datasets.CocoDetection(vg_img_path, ann_file)
    with open(ann_file, "r") as fp:
        annotation = json.load(fp)
    incorrect_idx = []
    skip_idx = 0
    for idx in tqdm(range(len(annotation['images']))):
        if max_len is not None:
           tokenized = tokenizer(annotation['images'][idx]['caption'])
           text_length = len(tokenized['input_ids'])
           if text_length > max_len:
               skip_idx += 1
               continue
        _, coco_target = coco.__getitem__(idx)
        
        image_path = os.path.join(vg_img_path ,annotation['images'][idx]['file_name'])
        
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

        if len(category_id_lst) != 0 and len(bbox_lst) != 0 and len(token_positive_lst) != 0 and len(category_id_lst) == len(bbox_lst) == len(token_positive_lst):
            anno = [idx, binary, image_path, w, h, [annotation['images'][idx]['caption']],  category_id_lst, bbox_lst, token_positive_lst, [annotation['images'][idx]['tokens_negative']]]
            db.append(anno)
        else:
            incorrect_idx.append(idx)

    print("Total " + str(len(db)) + " gqa val pairs added")
    p:wqrint("Total " + str(len(incorrect_idx)) + " gqa val pairs filtered")
    print("Skip " +  str(skip_idx) + " pairs")
 
    return db


def make_arrow(coco_path, vg_img_path, flickr_img_path,flickr_dataset_path, mdetr_anno_path, ref_path, dataset_root, max_len=None, max_bbox=None, lang="bert"):

    if max_len is not None:
        if lang == "bert":
            tokenizer = get_pretrained_tokenizer("bert-base-uncased")
        elif lang == "roberta":
            tokenizer = get_pretrained_roberta_tokenizer("roberta-base")
    print("Processing ref all")
    ref_all_train, ref_all_val = read_ref_all_train_val(ref_path, mdetr_anno_path, dataset_root)
    
    # train
    print("Processing flickr_train_db")
    flickr_train_db = read_flickr_separateGT_train(coco_path, vg_img_path, flickr_img_path,flickr_dataset_path, mdetr_anno_path, max_len=max_len, tokenizer=tokenizer, max_bbox=max_bbox)

    print("Processing mix_train_db")
    mix_train_db = read_mix_train(coco_path, vg_img_path, flickr_img_path,flickr_dataset_path, mdetr_anno_path, max_len=max_len, tokenizer=tokenizer, max_bbox=max_bbox)
    train_db = flickr_train_db + mix_train_db + ref_all_train
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
    val_db = ref_all_val
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
