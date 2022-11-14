from vilt.datasets import ReferItGameDataset
from .datamodule_base import BaseDataModule
from collections import defaultdict


class ReferItGameDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ReferItGameDataset

    @property
    def dataset_name(self):
        return "referitgame"

    def setup(self, stage):
        super().setup(stage)

