from vilt.datasets import CleverMdetrDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class CleverMdetrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CleverMdetrDataset

    @property
    def dataset_cls_no_false(self):
        return CleverMdetrDataset

    @property
    def dataset_name(self):
        return "clever_mdetr"

    def setup(self, stage):
        super().setup(stage)

