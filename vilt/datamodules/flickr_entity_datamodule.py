from vilt.datasets import FlickrEntityDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class FlickrEntityDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return FlickrEntityDataset

    @property
    def dataset_cls_no_false(self):
        return FlickrEntityDataset

    @property
    def dataset_name(self):
        return "flickr_entity"

    def setup(self, stage):
        super().setup(stage)

