from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)

from .coco import (
coco_transform,
coco_transform_randaug,
coco_transform_randaug2,
coco_transform_randaug3,
coco_transform_randaug4
        )
_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
    "coco":coco_transform,
    "coco_randaug":coco_transform_randaug,
    "coco_randaug2":coco_transform_randaug2,
    "coco_randaug3":coco_transform_randaug3,
    "coco_randaug4":coco_transform_randaug4,
}


def keys_to_transforms(keys: list, size=224):
    return [_transforms[key](size=size) for key in keys]
