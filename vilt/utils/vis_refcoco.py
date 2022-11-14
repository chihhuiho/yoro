import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from refer.refer import REFER


def query(root, query_id):
    #for split in ["train", "val", "test"]:
    with open(f"{root}/annotations/instances_train2014.json", "r") as fp:
        annotations_train2014 = json.load(fp)
    # building dict to map image id to annotation
    imageID2Annot = {}
    for i in range(len(annotations_train2014["images"])):
        imageID2Annot[annotations_train2014["images"][i]['id']] = annotations_train2014["images"][i]
    split2folder = {"train":"train2014", "val":"val2014","test":"test2015"}

    refcocop = REFER(root, dataset='refcoco+', splitBy='unc') # refcocop : refcoco pluse
    for split in ["test"]:
        refer_ids = []
        refer_ids.extend(refcocop.getRefIds(split=split))
        refs = refcocop.loadRefs(ref_ids=refer_ids)
        db = []
        for ref_id, ref in tqdm(zip(refer_ids, refs)):
            if ref_id == query_id:
            #if split != "test":
                gt_x, gt_y, gt_w, gt_h = refcocop.getRefBox(ref_id=ref_id)
                gt_x /= imageID2Annot[ref['image_id']]['width']
                gt_y /= imageID2Annot[ref['image_id']]['height']
                gt_w /= imageID2Annot[ref['image_id']]['width']
                gt_h /= imageID2Annot[ref['image_id']]['height']
                image_path = os.path.join(root, "train2014", 'COCO_train2014_{:012d}.jpg'.format(ref['image_id']))
           
                return [image_path, imageID2Annot[ref['image_id']]['width'], imageID2Annot[ref['image_id']]['height'], ref['category_id'], [gt_x, gt_y, gt_w, gt_h] ] 
    return [] 

if __name__ == "__main__":
    info = query("/mnt/efs/hchihhu/datasets/raw/refcoco/", 28)
    print(info)
