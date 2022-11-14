from vilt.datasets import GQAMdetrDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class GQAMdetrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return GQAMdetrDataset

    @property
    def dataset_cls_no_false(self):
        return GQAMdetrDataset

    @property
    def dataset_name(self):
        return "gqa_mdetr"

    def setup(self, stage):
        super().setup(stage)

