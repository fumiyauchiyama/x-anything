# https://github.com/simoneangarano/segment-anything/blob/main/segment_anything/utils/data.py
from dataclasses import dataclass
import random
import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DistributedSampler
import json
import os

from PIL import Image

from pycocotools import mask as mask_utils
from x_anything.utils.transforms import ResizeLongestSide

logger = logging.getLogger()

MAX_RETRIES = 100


@dataclass
class SegmentationDataPoint:
    # H: height, W: width (differed by image)
    # C: channel
    # N: number of masks
    image: torch.Tensor # CHW format, torch.uint8
    masks: torch.Tensor # NHW tensor
    bboxes: torch.Tensor # N*4 tensor, XYXY format

@dataclass
class SA1BDatasetArgs:
    jpg_dir: Optional[str] = None
    json_dir: Optional[str] = None
    num_data: int = 223749 # for tar 1-20
    # You can get the number of data by running the following command:
    # ls "$YOUR_SA1B_JSON_DIR" | \
    #     grep -oP 'sa_\K\d+(?=\.json)' | \
    #     sort -n | \
    #     tail -n 1
    max_mask_num: int = 64
    img_size: int = 1024
    max_mask_area_ratio: float = 0.9


class SA1B_Dataset(Dataset):
    def __init__(self, args: SA1BDatasetArgs) -> None:
        assert args.jpg_dir is not None and args.json_dir is not None
        self.jpg_dir = args.jpg_dir
        self.json_dir = args.json_dir
        self.num_data = args.num_data
        self.max_mask_num = args.max_mask_num
        self.max_mask_area_ratio = args.max_mask_area_ratio

    def __len__(self) -> int:
        return self.num_data
    
    def __getitem__(self, idx) -> SegmentationDataPoint:
        for retry in range(MAX_RETRIES):
            try:
                img = Image.open(os.path.join(self.jpg_dir, f"sa_{idx+1}.jpg"))
                # convert img to tensor with HWC uint8 format
                img_numpy = np.array(img).astype(np.uint8)
                img_tensor =  torch.from_numpy(img_numpy)
                img_tensor = img_tensor.permute(2, 0, 1).contiguous()
                with open(os.path.join(self.json_dir, f"sa_{idx+1}.json")) as f:
                    json_data = json.load(f)
                anns = json_data['annotations']
                num_anns = len(anns)
                # sample up to self.max_mask_num annotation indexes
                ann_idx = random.sample(range(num_anns), num_anns)
                bboxes = []
                masks = []
                for idx in ann_idx:
                    ann = anns[idx]

                    area = ann['area']
                    if area <= self.max_mask_area_ratio * img_tensor.shape[1] * img_tensor.shape[2]:
                        # convert XYWH to XYXY
                        bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]]
                        bboxes.append(torch.tensor(bbox))
                        mask = mask_utils.decode(ann['segmentation'])
                        masks.append(torch.from_numpy(mask))

                    if len(masks) >= self.max_mask_num:
                        break

                return SegmentationDataPoint(
                    image=img_tensor,
                    masks=torch.stack(masks),
                    bboxes=torch.stack(bboxes),
                )
            except Exception as e:
                logging.warning(
                    f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                )
                idx = random.randrange(0, len(self.num_data))


class SegmentationBatch:
    def __init__(self, batch: List[SegmentationDataPoint]) -> None:
        self.images = [x.image for x in batch]
        self.masks = [x.masks for x in batch]
        self.bboxes = [x.bboxes for x in batch]

    def pin_memory(self):
        self.images = [x.pin_memory() for x in self.images]
        self.masks = [x.pin_memory() for x in self.masks]
        self.bboxes = [x.pin_memory() for x in self.bboxes]
        return self
    
def collate_wrapper(batch: List[SegmentationDataPoint]) -> SegmentationBatch:
    return SegmentationBatch(batch)

def get_dataloader(
        args: SA1BDatasetArgs, 
        rank,
        world_size,
        batch_size: int = 1, # set batch size to 1 for image singe there are up to 64 masks 
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    ) -> Tuple[DataLoader, DistributedSampler]:
    dataset = SA1B_Dataset(args)
    sampler: DistributedSampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=shuffle
        )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, # shuffle is done by sampler
        num_workers=num_workers, 
        collate_fn=collate_wrapper, 
        sampler=sampler,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return dataloader, sampler