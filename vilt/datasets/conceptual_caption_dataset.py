from glob import glob
from .base_dataset import BaseDataset


class ConceptualCaptionDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        if split == "test":
            split = "val"
        self.split = split
        if split == "train":
            names = [f"GCC_resize_512/train_{i}" for i in range(24)]
        elif split == "val":
            #names = ["GCC/val_0"]
            names = []


        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        try:
            return self.get_suite(index)
        except:
            return None
