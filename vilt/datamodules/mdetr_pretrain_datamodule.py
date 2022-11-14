from vilt.datasets import MdetrPretrainDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class MdetrPretrainDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MdetrPretrainDataset

    @property
    def dataset_name(self):
        return "mdetr_pretrain"

    def setup(self, stage):
        super().setup(stage)

