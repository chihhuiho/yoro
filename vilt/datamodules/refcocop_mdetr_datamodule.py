from vilt.datasets import RefCocoPMdetrDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class RefCocoPMdetrDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RefCocoPMdetrDataset

    @property
    def dataset_name(self):
        return "refcocop_mdetr"

    def setup(self, stage):
        super().setup(stage)

