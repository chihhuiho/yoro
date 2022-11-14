from vilt.utils.write_refcocop_mdetr import make_arrow as refcocop_mdetr_make_arrow
from vilt.utils.write_refcoco_mdetr import make_arrow as refcoco_mdetr_make_arrow
from vilt.utils.write_refcocog_mdetr import make_arrow as refcocog_mdetr_make_arrow
from vilt.utils.write_mdetr_pretrain import make_arrow as mdetr_pretrain_make_arrow
from vilt.utils.write_copsref import make_arrow as copsref_make_arrow
from vilt.utils.write_referitgame import make_arrow as referitgame_make_arrow
import os


# MAX TEXT LEN FOR OUR MODEL IS 40

# for Modulated detection pretraining
#print("======= Processing mdetr pretrain annotation =======")
mdetr_pretrain_make_arrow(coco_path="./datasets/raw/COCO", vg_img_path="./datasets/raw/GQA/images", flickr_img_path="./datasets/raw/F30K/flicker/flickr30k-images", flickr_dataset_path="./dataset/raw/F30K/flickr30k_entities", mdetr_anno_path="./datasets/raw/MDETR/mdetr_annotations", ref_path="./datasets/raw/refcoco", dataset_root="./datasets/arrow/mdetr_pretrain/maxlen40_maxbox1", max_len=40, max_bbox=1)

#print("======= REC task Processing refcoco+ with mdetr annotation =======")
refcocop_mdetr_make_arrow("./datasets/raw/refcoco", "./datasets/raw/MDETR/mdetr_annotations", "./datasets/arrow/refcocop_mdetr")

#print("======= REC task Processing refcoco with mdetr annotation =======")
refcoco_mdetr_make_arrow("./datasets/raw/refcoco", "./datasets/raw/MDETR/mdetr_annotations", "./datasets/arrow/refcoco_mdetr")

#print("======= REC task Processing refcocog with mdetr annotation =======")
refcocog_mdetr_make_arrow("./datasets/raw/refcoco", "./datasets/raw/MDETR/mdetr_annotations", "./datasets/arrow/refcocog_mdetr")

#print("======= REC task Processing referitgame/refclef =======")
referitgame_make_arrow(refcoco_root="./datasets/raw/refcoco", imageclef_root="./datasets/raw/IMAGECLEF/iaprtc12/images", dataset_root="./datasets/arrow/referitgame")

#print("======= REC task Processing Cops-Ref =======")
copsref_make_arrow(gqa_img_path="./datasets/raw/GQA/images", copsref_path="./datasets/raw/COPSREF", dataset_root="./datasets/arrow/COPSREF", max_len=40)


