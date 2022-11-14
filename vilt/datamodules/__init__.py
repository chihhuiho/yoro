from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .refcocop_mdetr_datamodule import RefCocoPMdetrDataModule
from .refcoco_mdetr_datamodule import RefCocoMdetrDataModule
from .refcocog_mdetr_datamodule import RefCocogMdetrDataModule
from .mdetr_pretrain_datamodule import MdetrPretrainDataModule
from .flickr_mdetr_datamodule import FlickrMdetrDataModule
from .flickr_entity_datamodule import FlickrEntityDataModule
from .gqa_mdetr_datamodule import GQAMdetrDataModule
from .clever_mdetr_datamodule import CleverMdetrDataModule
from .snlive_datamodule import SnliveDataModule
from .copsref_datamodule import CopsRefDataModule
from .referitgame_datamodule import ReferItGameDataModule


_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "refcocop_mdetr":RefCocoPMdetrDataModule,
    "refcocog_mdetr":RefCocogMdetrDataModule,
    "refcoco_mdetr":RefCocoMdetrDataModule,
    "mdetr_pretrain":MdetrPretrainDataModule,
    "flickr_mdetr":FlickrMdetrDataModule,
    "flickr_entity":FlickrEntityDataModule,
    "gqa_mdetr":GQAMdetrDataModule,
    "clever_mdetr":CleverMdetrDataModule,
    "snlive":SnliveDataModule,
    "copsref":CopsRefDataModule,
    "referitgame":ReferItGameDataModule,
}
