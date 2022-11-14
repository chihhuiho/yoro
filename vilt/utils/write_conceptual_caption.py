import json
import pandas as pd
import pyarrow as pa
import gc
import random
import os

from tqdm import tqdm
from glob import glob
from PIL import Image

def make_arrow(root, dataset_root, resize=False):
     correct_data_folder = "resized_512" if resize else "correct_data"
     for split in ["train","val"]:
         if split == "train":
             tsv_file = "Train_GCC-training.tsv"
         elif split == "val":
             tsv_file = "Validation_GCC-1.1.0-Validation.tsv"
         
         f = open(f"{root}/conceptual_captions_3.3m/conceptual_captions/utils/" + tsv_file)
         caption2url_data = f.readlines()
         f.close()

         idx2url = {}
         idx2caption = {}
         for idx in range(len(caption2url_data)):
             caption, url = caption2url_data[idx].rstrip().split("\t")
             idx2caption[idx] = caption
             idx2url[idx] = url
        
         
         db = {}
         print("========== Processing " + split + " dataset ==================")
         
         for cnt, img in tqdm(enumerate(os.listdir(f"{root}/conceptual_captions_3.3m/conceptual_captions/textract_scenetext_ocr/json_preds"))):
 
             if img[-4:] != "json":
                 continue

             img = img[:-4] + "jpg"
             if not os.path.exists(f"{root}/conceptual_captions_3.3m/conceptual_captions/utils/{correct_data_folder}/{img}"):
                 continue
             idx = int(img.split("_")[0])
             url_tmp = img.split("_")[1][:-4]
             if idx not in idx2caption:
                 continue
             
             if url_tmp != idx2url[idx-1].split("/")[-1][:-4]:
                 continue

             sub = int((cnt+1)/100000) 
             if sub not in db:
                 db[sub] = []
             
             caption = idx2caption[idx]

             image_path = f"{root}/conceptual_captions_3.3m/conceptual_captions/utils/{correct_data_folder}/{img}"
             if idx in [2513081, 242206, 1721664, 832451, 772233, 500473, 3201447, 2466714, 277590] and resize:
                 continue
 
             try:
                 im = Image.open(image_path)
                 if im.getpalette() is not None:
                     continue

                 im = im.convert('RGB')
                 if (im.size[0] < 10 or im.size[1] < 10 or im.size[0] > 10000 or im.size[1] > 10000):
                     continue
             except:
                 continue

             with open(image_path, "rb") as fp:
                 binary = fp.read()
             image_id = img[:-4]
             db[sub].append([binary, caption, image_id, split])
 
         for sub in db:
             dataframe = pd.DataFrame(
                   db[sub], columns=["image", "caption", "image_id", "split"],
            )
             table = pa.Table.from_pandas(dataframe)
             os.makedirs(dataset_root, exist_ok=True)
             with pa.OSFile(f"{dataset_root}/{split}_{sub}.arrow", "wb") as sink:
                 with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                     writer.write_table(table)
          
