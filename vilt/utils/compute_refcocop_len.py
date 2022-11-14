import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from vilt.utils.refer.refer import REFER


def compute_len(root):
    refcocop = REFER(root, dataset='refcoco+', splitBy='unc') # refcocop : refcoco pluse
    cnt = 0
    length = 0 
    for split in ["testA", "testB", "train", "val", "test"]:
        refer_ids = []
        refer_ids.extend(refcocop.getRefIds(split=split))
        refs = refcocop.loadRefs(ref_ids=refer_ids)
        db = []
        for ref_id, ref in tqdm(zip(refer_ids, refs)):
            for sent in ref['sentences']:
                length += len(sent['sent'].split(" "))
                cnt += 1 
       
    print("Avg len " + str(length/cnt))    
    print(str(cnt))    
