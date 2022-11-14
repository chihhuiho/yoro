# code modified from https://github.com/jackroos/VL-BERT
from .base_dataset import BaseDataset
import torch
from vilt.transforms.coco import box_cxcywh_to_xyxy

class ReferItGameDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        '''
        ReferItGame/RefClef Dataset
        :param image_set: image folder name
        :param data_path: path to dataset
        :param transform: transform
        :param test_mode: test mode means no labels available
        '''
        assert split in ["train", "val", "test"]  
        self.split = split
        if split == "train":
            names = ["referitgame_train"]
        elif split == "val":
            names = ["referitgame_val"]
        elif split == "test":
            names = ["referitgame_test"]   

        print("Loading referitgame")

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="caption",
            remove_duplicate=False,
        )



    def __getitem__(self, index):
        gt_bbox = self.table["gt_bbox"][index].as_py()
        ori_width = self.table["width"][index].as_py()
        ori_height = self.table["height"][index].as_py()

        gt_bbox = [[b[0]*ori_width, b[1]*ori_height, (b[0]+b[2])*ori_width, (b[1]+b[3])*ori_height] for b in gt_bbox] 
        
        target = {"boxes":torch.FloatTensor(gt_bbox).view(-1,4)}
        ret_lst = self.get_image(index, det=True, gt_bbox=target)
        image_tensor = ret_lst["image"]
        gt_bbox = ret_lst["gt_bbox"]

        text = self.get_text(index)["text"]
        category_id = self.table["category_id"][index].as_py()
        caption = self.table["caption"][index].as_py()
        tokens_positive = self.table["tokens_positive"][index].as_py()
        tokenized = self.tokenizer(caption, return_tensors="pt")
        positive_map = create_positive_map(tokenized, tokens_positive)

        width = image_tensor[0].shape[2]
        height = image_tensor[0].shape[1]

        return {
            "ref_id": index,
            "width": width,
            "height": height,
            "image": image_tensor,
            "text": text,
            "gt_bbox": gt_bbox,
            "category_id": category_id,
            "positive_map": positive_map,
            "tokens_positive": tokens_positive,
        }


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""

    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list: # begin and end
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)



