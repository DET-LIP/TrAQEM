import random
from typing import List, Tuple, Dict, Optional

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image: PIL.Image.Image, target: Dict, region: Tuple[int, int, int, int]) -> Tuple[PIL.Image.Image, Dict]:
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "clip_image" in target:
        fields.append("clip_image")

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, i: i + h, j: j + w]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            cropped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image: PIL.Image.Image, target: Dict) -> Tuple[PIL.Image.Image, Dict]:
    flipped_image = F.hflip(image)
    w, _ = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def resize(image: PIL.Image.Image, target: Optional[Dict], size: int, max_size: Optional[int] = None) -> Tuple[PIL.Image.Image, Optional[Dict]]:
    def get_size_with_aspect_ratio(image_size, size, max_size):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        return size, size

    size = get_size_with_aspect_ratio(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target["masks"] = interpolate(target["masks"][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image: PIL.Image.Image, target: Optional[Dict], padding: Tuple[int, int]) -> Tuple[PIL.Image.Image, Optional[Dict]]:
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))

    if target is None:
        return padded_image, None

    target = target.copy()
    target["size"] = torch.tensor(padded_image[::-1])

    if "masks" in target:
        target["masks"] = torch.nn.functional.pad(target["masks"], (0, padding[0], 0, padding[1]))

    return padded_image, target


# Add classes for the other transformations here...

class Compose(object):
    def __init__(self, transforms: List[object]):
        self.transforms = transforms

    def __call__(self, image: PIL.Image.Image, target: Optional[Dict] = None) -> Tuple[PIL.Image.Image, Optional[Dict]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        return f"Compose({', '.join([str(t) for t in self.transforms])})"
