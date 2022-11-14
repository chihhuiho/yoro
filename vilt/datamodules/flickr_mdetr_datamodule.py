from vilt.datasets import FlickrMdetrDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class FlickrMdetrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return FlickrMdetrDataset

    @property
    def dataset_cls_no_false(self):
        return FlickrMdetrDataset

    @property
    def dataset_name(self):
        return "flickr_mdetr"

    def setup(self, stage):
        super().setup(stage)

