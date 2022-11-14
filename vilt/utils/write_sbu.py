import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob


def make_arrow(root, dataset_root):
    f_caption = open(f"{root}/annotations/SBU_captioned_photo_dataset_captions.txt", "r")
    captions = f_caption.readlines()
    f_caption.close()

    f_url = open(f"{root}/annotations/SBU_captioned_photo_dataset_urls.txt", "r")
    urls = f_url.readlines()
    f_url.close()

    idx2caption = {}
    for idx, caption in enumerate(captions):
        idx2caption[idx] = caption.rstrip()

    idx2url = {}
    url2idx = {}
    db = []

    for idx, url in tqdm(enumerate(urls)):
        url = url.rstrip().split("/")[-1][:-4]
        idx2url[idx] = url.rstrip()
        img_name = url
        url2idx[img_name] = idx
        if os.path.isfile(f"{root}/image/{url}.jpg"):
            with open(f"{root}/image/{url}.jpg", "rb") as fp:
                binary = fp.read()
            db.append([binary, f"{url}.jpg", idx, idx2caption[idx]])
           
            '''
            if (idx+1)%100000 == 0:
               sub = int((idx+1)/100000)-1
               dataframe = pd.DataFrame(db, columns=["image", "image_path", "image_id", "caption"],)
               table = pa.Table.from_pandas(dataframe)
               os.makedirs(dataset_root, exist_ok=True)
               print("Saving " + f"sbu_{sub}.arrow with " + str(len(db)) + " data")
               with pa.OSFile(f"{dataset_root}/sbu_{sub}.arrow", "wb") as sink:
                   with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                       writer.write_table(table)
               del dataframe
               del table
               del db
               db = []
            '''
    dataframe = pd.DataFrame(db, columns=["image", "image_path", "image_id", "caption"],)
    table = pa.Table.from_pandas(dataframe)
    os.makedirs(dataset_root, exist_ok=True)
    print("Saving " + "train.arrow with " + str(len(db)) + " data")
    with pa.OSFile(f"{dataset_root}/train.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
 
