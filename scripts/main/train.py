import torch

import random
from typing import List, Dict, Any

from x_anything.utils.transforms import ResizeLongestSide


def train_one_epoch(
        iter,
        epoch,
        model,
        data_loader,
        optimizer,
        sampler,
        scheduler,
        rank,
        world_size,
):
    model.train()
    transform = ResizeLongestSide(model.image_encoder.img_size)
    sampler.set_epoch(epoch)
    for batch in data_loader:
        iter += 1
        optimizer.zero_grad()

        # For now, we will decompose the batch for each image
        for idx in range(len(batch.images)):
            num_ann_samples = batch.num_ann_samples[idx].item()
            image = model.preprocess(batch.images[idx]) # HWC
            targets = batch.masks[idx][:num_ann_samples] # NHW
            areas = batch.areas[idx][:num_ann_samples] # N
            bboxes = batch.bboxes[idx][:num_ann_samples] # N4
            # Randomly sample which prompt to use for B masks
            is_point = [random.random() < 0.5 for _ in range(batch.num_ann_samples[idx].item())]
            batched_input = []
            for i in range(num_ann_samples):
                # skip if the mask is almost the whole image
                H, W = image.shape[:2]
                if areas[i] / (H * W) > 0.9:
                    continue

                if is_point[i]:
                    # Uniformly sample a point from the target mask
                    gt_mask = targets[i]
                    mask_indices = torch.nonzero(gt_mask)
                    if len(mask_indices) == 0:
                        continue
                    point_idx = random.choice(mask_indices)
                    point = point_idx.flip(0).float()
                    point = transform.apply_coords_torch(point, (H, W))
                    

                    batched_input.append({
                        'image': image,
                        'original_size': image.shape[:2],
                        'point_coords': bboxes[i].reshape(1, 2, 2),
                        'point_labels': torch.tensor([1]),
                        'mask_inputs': masks[i].reshape(1, 1, masks[i].shape[0], masks[i].shape[1]),
                    })
                else:
                    batched_input.append({
                        'image': image,
                        'original_size': image.shape[:2],
                        'boxes': bboxes[i].reshape(1, 4),
                        'mask_inputs': masks[i].reshape(1, 1, masks[i].shape[0], masks[i].shape[1]),
                    })

            batch = batch.pin_memory() if batch.is_cuda else batch
            batch = batch.to(rank)
            loss = model(batch)
            loss.backward()
            optimizer.step()    
            scheduler.step()

        batch = batch.pin_memory() if batch.is_cuda else batch
        batch = batch.to(rank)
        loss = model(batch)
        loss.backward()
        optimizer.step()    
        scheduler.step()
    
    return iter