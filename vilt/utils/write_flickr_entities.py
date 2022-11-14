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
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from prettytable import PrettyTable

#import util.dist as dist
import vilt.modules.dist_utils as dist
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _merge_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """
    Return the boxes corresponding to the smallest enclosing box containing all the provided boxes
    The boxes are expected in [x1, y1, x2, y2] format
    """
    if len(boxes) == 1:
        return boxes

    np_boxes = np.asarray(boxes)

    return [[np_boxes[:, 0].min(), np_boxes[:, 1].min(), np_boxes[:, 2].max(), np_boxes[:, 3].max()]]



def get_sentence_data(filename) -> List[Dict[str, Any]]:
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      filename - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(filename, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data["phrases"].append(
                {"first_word_index": index, "phrase": phrase, "phrase_id": p_id, "phrase_type": p_type}
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(filename) -> Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]]:
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      filename - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
          height - int representing the height of the image
          width - int representing the width of the image
          depth - int representing the depth of the image
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info: Dict[str, Union[int, List[str], Dict[str, List[List[int]]]]] = {}
    all_boxes: Dict[str, List[List[int]]] = {}
    all_noboxes: List[str] = []
    all_scenes: List[str] = []
    for size_element in size_container:
        assert size_element.text
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            assert box_id
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in all_boxes:
                    all_boxes[box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text)
                ymin = int(box_container[0].findall("ymin")[0].text)
                xmax = int(box_container[0].findall("xmax")[0].text)
                ymax = int(box_container[0].findall("ymax")[0].text)
                all_boxes[box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    all_noboxes.append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    all_scenes.append(box_id)
    anno_info["boxes"] = all_boxes
    anno_info["nobox"] = all_noboxes
    anno_info["scene"] = all_scenes

    return anno_info



def bbox_check(bbox):
    bbox_lst = np.clip(np.asarray(bbox), 0, 1).tolist()
    return bbox_lst

def read_flickr_entity(flickr_img_path, flickr_dataset_path, eval_type, max_len=None, tokenizer=None, max_bbox=None, split="train"):
    db = []
    if split == "test":
        max_bbox=None

    with open(flickr_dataset_path + "/" + split + ".txt") as f:
        img_id_lst = f.readlines()

    for img_id in tqdm(img_id_lst):
        img_id = img_id.rstrip()
        image_path = os.path.join(flickr_img_path, img_id + ".jpg")
        
        with open(image_path, "rb") as fp:
            binary = fp.read()

        sents = get_sentence_data(flickr_dataset_path + "/Sentences/" + img_id + ".txt") 
        anno = get_annotations(flickr_dataset_path + "/Annotations/" + img_id + ".xml")
        w = float(anno["width"])
        h = float(anno["height"])

        for sent in sents:
            for phrase in sent["phrases"]:
                phrase_id = phrase["phrase_id"]
                if phrase_id in anno["boxes"]:
                    # compute token positive
                    token_positive_lst = []
                    caption = phrase["phrase"]
                    token_positive_lst.append([[0, len(caption)]])
            
                    # compute all bboxes that the phrase referred to
                    all_bboxes_lst = []
                    for box in anno["boxes"][phrase_id]:
                        gt_x, gt_y, gt_w, gt_h = box[0], box[1], box[2]-box[0], box[3]-box[1]
                        gt_x /= w
                        gt_y /= h
                        gt_w /= w
                        gt_h /= h
                        all_bboxes_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
 
                    # Any box evaluate type
                    if eval_type == "seperate":
                        for box in anno["boxes"][phrase_id]:
                            category_id_lst = []
                            bbox_lst = []
 
                            gt_x, gt_y, gt_w, gt_h = box[0], box[1], box[2]-box[0], box[3]-box[1]
                            category_id_lst.append(0)
                            gt_x /= w
                            gt_y /= h
                            gt_w /= w
                            gt_h /= h
                            bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
                            db.append([img_id, phrase_id, binary, image_path, w, h, [caption],  category_id_lst, bbox_lst, token_positive_lst, all_bboxes_lst])
                    
                    # Merge evaluation type 
                    elif eval_type == "merged":
                        category_id_lst = []
                        bbox_lst = []
 
                        merged_box = _merge_boxes(anno["boxes"][phrase_id])[0]
                        gt_x, gt_y, gt_w, gt_h = merged_box[0], merged_box[1], merged_box[2]-merged_box[0], merged_box[3]-merged_box[1]
                       
                        ''' 
                        if len(anno["boxes"][phrase_id]) > 1:
                            print(anno["boxes"][phrase_id])
                            print(merged_box)
                            fig, ax = plt.subplots()
                            img = plt.imread(image_path)
                            ax.imshow(img)
                            plt.axis("off")
                            rect = patches.Rectangle((gt_x, gt_y), gt_w, gt_h, linewidth=1, edgecolor='g', facecolor='none')
                            ax.add_patch(rect)
                            plt.savefig(f"RRR_{split}.png", bbox_inches='tight', pad_inches=0)
                            plt.close(fig)
                        ''' 

                        category_id_lst.append(0)
                        gt_x /= w
                        gt_y /= h
                        gt_w /= w
                        gt_h /= h
                        bbox_lst.append(bbox_check([gt_x, gt_y, gt_w, gt_h]))
                        
                        db.append([img_id, phrase_id, binary, image_path, w, h, [caption],  category_id_lst, bbox_lst, token_positive_lst, all_bboxes_lst])
    
    return db


def make_arrow(flickr_img_path, flickr_dataset_path, dataset_root, eval_type, max_len=None, max_box=None):

    tokenizer = get_pretrained_tokenizer("bert-base-uncased")
             
    # train
    for split in ["train", "val", "test"]:
        print(f"Processing flickr {eval_type} {split}")
        db = read_flickr_entity(flickr_img_path, flickr_dataset_path, eval_type=eval_type, max_len=max_len, max_bbox=max_box, tokenizer=tokenizer, split=split)
        #print(len(db)) 
        dataframe = pd.DataFrame(
                db, columns=["img_id", "phrase_id", "image", "image_path", "width", "height", "caption", "category_id", "gt_bbox", "tokens_positive", "all_gt_bbox"],
          )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
                f"{dataset_root}/{split}_{eval_type}.arrow", "wb"
            ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
