from .base_dataset import BaseDataset


class F30KCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        if split == "train":
            names = ["f30k_caption_karpathy_train", "f30k_caption_karpathy_val"]
        elif split == "val":
            names = ["f30k_caption_karpathy_test"]
        elif split == "test":
            names = ["f30k_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        ret = self.get_suite(index)
        if "false_image_0" in ret:
            ret["false_image_0"] = [ret["false_image_0"][0]]
        #print("===========")
        #print(type(ret["false_image_0"]))
        #print(type(ret["image"]))
        #print(type(ret["false_image_0"][0]))
        ret["image"][0] = ret["image"][0][0]
        #print("===========")
 
        #print("=============")
        #print(type(ret["image"]))
        #print(type(ret["false_image_0"]))
        #print(ret["false_image_0"][0].shape)
        #print("=============")
 
        return ret
