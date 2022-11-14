from vilt.datasets import RefCocogMdetrDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class RefCocogMdetrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RefCocogMdetrDataset

    @property
    def dataset_name(self):
        return "refcocog_mdetr"

    def setup(self, stage):
        super().setup(stage)

