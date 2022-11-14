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


def make_arrow(coco_path, vg_img_path, flickr_img_path,flickr_dataset_path, mdetr_anno_path, ref_path, dataset_root, max_len=None, max_bbox=None, lang="bert"):

    if max_len is not None:
        if lang == "bert":
            tokenizer = get_pretrained_tokenizer("bert-base-uncased")
        elif lang == "roberta":
            tokenizer = get_pretrained_roberta_tokenizer("roberta-base")
    print("Processing ref all")
    ref_all_train, ref_all_val = read_ref_all_train_val(ref_path, mdetr_anno_path, dataset_root)
    train_db = ref_all_train
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
