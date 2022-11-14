from vilt.datasets import SnliveDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class SnliveDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return SnliveDataset

    @property
    def dataset_cls_no_false(self):
        return SnliveDataset

    @property
    def dataset_name(self):
        return "snlive"

    def setup(self, stage):
        super().setup(stage)

