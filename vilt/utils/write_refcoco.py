import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from vilt.utils.refer.refer import REFER


def make_arrow(root, dataset_root):
    #for split in ["train", "val", "test"]:
    with open(f"{root}/annotations/instances_train2014.json", "r") as fp:
        annotations_train2014 = json.load(fp)
    # building dict to map image id to annotation
    imageID2Annot = {}
    for i in range(len(annotations_train2014["images"])):
        imageID2Annot[annotations_train2014["images"][i]['id']] = annotations_train2014["images"][i]

    refcoco = REFER(root, dataset='refcoco', splitBy='unc')
    for split in ["testA", "testB", "train", "val", "test"]:
        refer_ids = []
        refer_ids.extend(refcoco.getRefIds(split=split))
        refs = refcoco.loadRefs(ref_ids=refer_ids)
        db = []
        for ref_id, ref in tqdm(zip(refer_ids, refs)):
            #if split != "test":
            gt_x, gt_y, gt_w, gt_h = refcoco.getRefBox(ref_id=ref_id)
            gt_x /= imageID2Annot[ref['image_id']]['width']
            gt_y /= imageID2Annot[ref['image_id']]['height']
            gt_w /= imageID2Annot[ref['image_id']]['width']
            gt_h /= imageID2Annot[ref['image_id']]['height']
            image_path = os.path.join(root, "train2014", 'COCO_train2014_{:012d}.jpg'.format(ref['image_id']))
            with open(image_path, "rb") as fp:
                binary = fp.read()


            for sent in ref['sentences']:
                db.append([sent['sent_id'], ref['ann_id'], ref['ref_id'], ref['image_id'], binary, image_path, imageID2Annot[ref['image_id']]['width'], imageID2Annot[ref['image_id']]['height'], [sent['sent']], ref['category_id'], [gt_x, gt_y, gt_w, gt_h] ] )
        
        dataframe = pd.DataFrame(
            db, columns=["sent_id", "ann_id", "ref_id", "image_id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/refcoco_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
