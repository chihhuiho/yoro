from vilt.datasets import CopsRefDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class CopsRefDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CopsRefDataset

    @property
    def dataset_name(self):
        return "copsref"

    def setup(self, stage):
        super().setup(stage)

