# https://github.com/simoneangarano/segment-anything/blob/main/segment_anything/utils/data.py
from dataclasses import dataclass
import random
from typing import List, Tuple

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

@dataclass
class SegmentationDataPoint:
    image: torch.Tensor # HWC format, torch.uint8
    masks: torch.Tensor # NHW tensor
    areas: torch.Tensor # N tensor
    bboxes: torch.Tensor # N*4 tensor, XYXY format
    num_ann_samples: torch.Tensor # valid number of annotations in N tensor


@dataclass
class SA1BDatasetArgs:
    jpg_dir: str
    json_dir: str
    num_data: int = 223749 # for tar 1-20
    # You can get the number of data by running the following command:
    # ls "$YOUR_SA1B_JSON_DIR" | \
    #     grep -oP 'sa_\K\d+(?=\.json)' | \
    #     sort -n | \
    #     tail -n 1
    max_mask_num: int = 64


class SA1B_Dataset(Dataset):
    def __init__(self, args: SA1BDatasetArgs) -> None:
        self.jpg_dir = args.jpg_dir
        self.json_dir = args.json_dir
        self.num_data = args.num_data
        self.max_mask_num = args.max_mask_num

    def __len__(self) -> int:
        return self.num_data
    
    def __getitem__(self, idx) -> SegmentationDataPoint:
        img = Image.open(os.path.join(self.jpg_dir, f"sa_{idx+1}.jpg"))
        # convert img to tensor with HWC uint8 format
        img_tensor =  torch.from_numpy(np.array(img).astype(np.uint8))
        with open(os.path.join(self.json_dir, f"sa_{idx+1}.json")) as f:
            json_data = json.load(f)
        anns = json_data['annotations']
        num_anns = len(anns)
        # sample up to self.max_mask_num annotation indexes
        ann_idx = random.sample(range(num_anns), min(self.max_mask_num, num_anns))
        bboxes = torch.zeros((self.max_mask_num, 4)) # XYXY format
        masks = torch.zeros((self.max_mask_num, img_tensor.shape[0], img_tensor.shape[1]))
        areas = torch.zeros(self.max_mask_num)
        for i, idx in enumerate(ann_idx):
            ann = anns[idx]
            # convert XYWH to XYXY
            bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]]
            bboxes[i] = torch.tensor(bbox)
            mask = mask_utils.decode(ann['segmentation'])
            masks[i] = torch.from_numpy(mask)
            areas[i] = ann['area']
        
        return SegmentationDataPoint(
            image=img_tensor,
            masks=masks,
            areas=areas,
            bboxes=bboxes,
            num_ann_samples=torch.tensor([len(ann_idx)]),
        )


class SegmentationBatch:
    def __init__(self, batch: List[SegmentationDataPoint]) -> None:
        self.images = [x.image for x in batch]
        self.masks = [x.masks for x in batch]
        self.areas = torch.stack([x.areas for x in batch])
        self.bboxes = torch.stack([x.bboxes for x in batch])
        self.num_ann_samples = torch.stack([x.num_ann_samples for x in batch])

    def pin_memory(self):
        self.images = [x.pin_memory() for x in self.images]
        self.masks = [x.pin_memory() for x in self.masks]
        self.areas = self.areas.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.num_ann_samples = self.num_ann_samples.pin_memory()
        return self
    

def collate_wrapper(batch: List[SegmentationDataPoint]) -> SegmentationBatch:
    return SegmentationBatch(batch)


def get_dataloader(
        args: SA1BDatasetArgs, 
        batch_size: int, 
        rank,
        world_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    ) -> Tuple[DataLoader, DistributedSampler]:
    dataset = SA1B_Dataset(args)
    sampler = DistributedSampler(
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