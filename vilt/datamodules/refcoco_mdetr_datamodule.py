from vilt.datasets import RefCocoMdetrDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class RefCocoMdetrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RefCocoMdetrDataset

    @property
    def dataset_name(self):
        return "refcoco_mdetr"

    def setup(self, stage):
        super().setup(stage)

