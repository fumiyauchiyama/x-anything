from typing import List, Dict, Any, Optional, Tuple
import torch
import random
import logging
from dataclasses import dataclass, field

from torch.nn.parallel import DistributedDataParallel as DDP

from x_anything.utils.transforms import ResizeLongestSide
from x_anything.training.loss import MultiStepMultiMasksAndIous, CORE_LOSS_KEY
from x_anything.training.optim import OptimizationArgs, get_optimizer
from x_anything.training.data import SA1BDatasetArgs, SA1B_Dataset, get_dataloader
from x_anything.build_sam import SamWithTimmViTArgs, build_sam_with_timm

logger = logging.getLogger()

@dataclass
class TrainerArgs:
    num_epochs: int = 2
    save_steps: int = 1000
    model: SamWithTimmViTArgs = field(default_factory=SamWithTimmViTArgs)
    optim: OptimizationArgs = field(default_factory=OptimizationArgs)
    train_dataset: SA1BDatasetArgs = field(default_factory=SA1BDatasetArgs)

class SamTrainer:
    def __init__(
            self,
            args: TrainerArgs,
            world_size: int,
            rank: int,
            ) -> None:
        self.num_epochs = args.num_epochs
        self.save_steps = args.save_steps
        model = build_sam_with_timm(args=args.model).to(rank)
        self.model = DDP(model, device_ids=[rank])
        self.loss_fn = MultiStepMultiMasksAndIous(weight_dict={
            "loss_mask": 20.0,
            "loss_dice": 1.0,
            "loss_iou": 1.0,
        })
        self.optimizer, self.scheduler = get_optimizer(self.model.module, args.optim)
        self.dataloader, self.train_sampler = get_dataloader(
            args.train_dataset, rank=rank, world_size=world_size,
            )
        self.world_size = world_size
        self.rank = rank

        self.transform = ResizeLongestSide(self.model.module.image_encoder.img_size)

    def _sample_point_from_segmentation(self, masks: torch.Tensor) -> torch.Tensor:
        """Sample points from the segmentation masks.
        
        Args:
            (torch.Tensor): BHW segmentation masks.
        Returns:
            (torch.Tensor): Bx1x2 array of point prompts to the model. Each point coord is in (X,Y) in pixels.
        """
        sampled_points = []
        for mask in masks:
            # sample a point from the mask
            indices = torch.nonzero(mask)
            sampled_point = indices[torch.randint(0, indices.shape[0], (1,)).squeeze()]
            sampled_points.append(sampled_point)
        return torch.stack(sampled_points).unsqueeze(1)

    def _sample_point_from_err_region(
            self, pred_masks: torch.Tensor, target_masks: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calc XOR between pred_masks and target_masks, then sample points from the XOR region
        if each sampled point is in the target_masks, then it is a foreground point.
        Args:
            pred_masks (torch.Tensor): BHW predicted masks.
            target_masks (torch.Tensor): BHW target masks.
        Returns:
            (torch.Tensor): Bx1x2 array of point prompts to the model. Each point coord is in (X,Y) in pixels.
            (torch.Tensor): Bx1 array of point labels. 1 indicates a foreground point, 0 indicates a background point.
        """
        
        pred_masks = pred_masks > self.model.module.mask_threshold
        target_masks = target_masks > 0
        try:
            xor_masks = pred_masks ^ target_masks
        except:
            print(pred_masks.shape, target_masks.shape)
            raise
        points = self._sample_point_from_segmentation(xor_masks)

        # Check if the sampled points are foreground or background
        point_labels = []
        for i, point in enumerate(points):
            point_label = target_masks[i, point[0, 0], point[0, 1]].item()
            point_labels.append(point_label)
        return points, torch.tensor(point_labels).unsqueeze(1)

    def prepare_prompt(
            self, 
            iter_num: int, 
            original_size: Tuple[int, int],
            no_new_sample: bool = False,
            previous_masks: Optional[torch.Tensor] = None,
            previous_coords: Optional[torch.Tensor] = None,
            previous_labels: Optional[torch.Tensor] = None,
            previous_boxes: Optional[torch.Tensor] = None,
            target_masks: Optional[torch.Tensor] = None,
            bboxes: Optional[torch.Tensor] = None,
            ) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare the prompt for the model: Pointes, bboxes, and masks.
        Args:
            iter_num (int): The iteration number.
            original_size (Tuple[int, int]): The original size of the image.
            no_new_sample (bool): Whether to sample new points from the target masks.
            previous_masks (Optional[torch.Tensor]): The BHW previous masks.
            previous_coords (Optional[torch.Tensor]): The BN2 previous points.
            previous_labels (Optional[torch.Tensor]): The BN previous point labels.
            previous_boxes (Optional[torch.Tensor]): The B4 previous boxes.
            target_masks (Optional[torch.Tensor]): The target masks.
            bboxes (Optional[torch.Tensor]): N4 bboxes of the target masks. XYXY format.
        Returns:
            points (Tuple[(torch.Tensor), (torch.Tensor)] or None): A (BxNx2, BxN) array of point prompts to the
                model. Each point coord is in (X,Y) in pixels. 1 indicates a foreground point, 0 indicates a
                background point and -1 indicates not a point.
            perturbated_bboxes (torch.Tensor or None): A Bx4 array given a box prompt to the
                model, in XYXY format.
            mask_input (torch.Tensor or None): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form Bx1xHxW, where
                for SAM, H=W=256. Masks returned by a previous iteration of the
                predict method do not need further transformation.
        """
        point_coords, point_labels, points, perturbated_bboxes = None, None, None, None
        # aooly image transform
        if iter_num == 0:
            assert not no_new_sample
            # Select which to use as mask input between point and bbox
            is_initial_prompt_bbox = random.random() < 0.5
            if is_initial_prompt_bbox:
                assert bboxes is not None
                # Apply bbox
                # Calc each bbox sidelength to decide noise level
                bbox_side_lengths_h = bboxes[:, 3] - bboxes[:, 1]
                bbox_side_lengths_w = bboxes[:, 2] - bboxes[:, 0]
                standard_deviation_h = bbox_side_lengths_h * 0.1
                standard_deviation_w = bbox_side_lengths_w * 0.1
                noise_h1 = torch.clamp(torch.normal(mean=0, std=standard_deviation_h), min=-20, max=20).int()
                noise_h2 = torch.clamp(torch.normal(mean=0, std=standard_deviation_h), min=-20, max=20).int()
                noise_w1 = torch.clamp(torch.normal(mean=0, std=standard_deviation_w), min=-20, max=20).int()
                noise_w2 = torch.clamp(torch.normal(mean=0, std=standard_deviation_w), min=-20, max=20).int()
                perturbated_bboxes = bboxes.clone()
                perturbated_bboxes[:, 0] += noise_w1
                perturbated_bboxes[:, 1] += noise_h1
                perturbated_bboxes[:, 2] += noise_w2
                perturbated_bboxes[:, 3] += noise_h2
            else:
                # Apply points
                assert target_masks is not None
                # sample B1 point from the target_masks
                point_coords = self._sample_point_from_segmentation(target_masks) # B×1×2
                point_labels = torch.ones((point_coords.shape[0], 1), device=point_coords.device)
        else:
            if not no_new_sample:
                # Apply points
                assert previous_masks is not None and target_masks is not None
                original_size = tuple(target_masks.shape[-2:])
                input_size = tuple(previous_masks.shape[-2:])
                previous_masks_high_res = self.model.module.postprocess_masks(
                    previous_masks.unsqueeze(1), input_size, original_size
                    ).squeeze(1)
                logger.info(f"Previous masks shape: {previous_masks.shape}")
                logger.info(f"Previous masks high res shape: {previous_masks_high_res.shape}")
                point_coords, point_labels = self._sample_point_from_err_region(
                    previous_masks_high_res, target_masks
                    ) # B×1×2, B×1
            else:
                assert previous_masks is not None

        # Resize the points and boxes to the model input size
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords_torch(point_coords, original_size)
            point_coords = point_coords.to(self.rank)
            point_labels = point_labels.to(self.rank)
        if perturbated_bboxes is not None:
            perturbated_bboxes = self.transform.apply_boxes_torch(perturbated_bboxes, original_size)
            perturbated_bboxes = perturbated_bboxes.to(self.rank)

        # Combine the previous masks
        if previous_masks is not None:
            previous_masks = previous_masks.unsqueeze(1).to(self.rank)

        if not no_new_sample:
            # Concat previous points and new points
            if previous_coords is not None:
                assert previous_labels is not None
                if point_coords is not None:
                    assert point_labels is not None
                    point_coords = torch.cat([previous_coords, point_coords], dim=1)
                    point_labels = torch.cat([previous_labels, point_labels], dim=1)
                else:
                    point_coords = previous_coords
                    point_labels = previous_labels
            # Apply bbox
            if previous_boxes is not None:
                perturbated_bboxes = previous_boxes
        else:
            point_coords = previous_coords
            point_labels = previous_labels
            perturbated_bboxes = previous_boxes

        if point_coords is not None:
            points = (point_coords, point_labels)

        return points, perturbated_bboxes, previous_masks

    def predict_multi_step(
            self,
            image: torch.Tensor,
            target_masks: torch.Tensor,
            bboxes: torch.Tensor,
            ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        """Predicts the masks and iou.
        Args:
            image (torch.Tensor): CHW image tensor. Make sure image format is RGB.
            target_masks: NHW target masks.
            bboxes: N4 bboxes of the target masks. XYXY format.
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing the prediction of the masks and the iou.
        """
        # set image to the model
        image = image.unsqueeze(0).float().to(self.rank)
        transformed_image = self.transform.apply_image_torch(image)
        original_size = tuple(image.shape[-2:])
        input_size = tuple(transformed_image.shape[-2:])
        assert len(original_size) == 2 and len(input_size) == 2

        input_image = self.model.module.preprocess(transformed_image)
        features = self.model.module.image_encoder(input_image)

        # set target masks to the model
        target_masks = target_masks.to(self.rank)

        # decide when to input prompt without additional sample
        iter_no_new_sample: int = random.randint(1, 10)
        
        # init output list
        output_mask_batch: List[Dict[str, Any]] = []
        # init buffer
        mask_buffer: Optional[torch.Tensor] = None # Number of masks, low-res mask size * 2
        coords_buffer: Optional[torch.Tensor] = None # Number of masks, N, 2
        labels_buffer: Optional[torch.Tensor] = None # Number of masks, N
        boxes_buffer: Optional[torch.Tensor] = None # Number of masks, 4

        # iterate steps for 11 times
        for step in range(11):
            # prepare prompt
            if step == iter_no_new_sample or step == 10:
                point_inputs, bbox_inputs, mask_inputs = self.prepare_prompt(
                    step, 
                    no_new_sample=True, 
                    previous_masks=mask_buffer,
                    previous_coords=coords_buffer,
                    previous_labels=labels_buffer,
                    previous_boxes=boxes_buffer,
                    target_masks=None,
                    bboxes=bboxes,
                    original_size=original_size,
                    )
            else:
                point_inputs, bbox_inputs, mask_inputs = self.prepare_prompt(
                    step, 
                    no_new_sample=False, 
                    previous_masks=mask_buffer, 
                    previous_coords=coords_buffer,
                    previous_labels=labels_buffer,
                    previous_boxes=boxes_buffer,
                    target_masks=target_masks,
                    bboxes=bboxes,
                    original_size=original_size,
                    )
            
            sparse_embeddings, dense_embeddings = self.model.module.prompt_encoder(
                points=point_inputs,
                boxes=bbox_inputs,
                masks=mask_inputs,
            )

            # multimask_output=True for the first step.
            multimask_output: bool = step == 0

            # Predict masks
            low_res_masks, iou_pred = self.model.module.mask_decoder(
                image_embeddings=features,
                image_pe=self.model.module.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            ) # (BMHW, BM) including single mask and multiple masks

            logger.info(f"step: {step}")
            if point_inputs is not None:
                logger.info(f"point_inputs shape: {point_inputs[0].shape}, {point_inputs[1].shape}")
            if bbox_inputs is not None:
                logger.info(f"bbox_inputs shape: {bbox_inputs.shape}")
            if mask_inputs is not None:
                logger.info(f"mask_inputs shape: {mask_inputs.shape}")
            
            # Upscale the masks to the original image resolution
            masks = self.model.module.postprocess_masks(low_res_masks, input_size, original_size)

            # Append the output to the output list
            output_mask_batch.append({
                "multistep_pred_multimasks_high_res": masks,
                "multistep_pred_ious": iou_pred,
                "multistep_object_score_logits": [None for _ in range(len(iou_pred))],
            })

            # update mask buffer
            # choose the best iou mask
            best_mask_idx = torch.argmax(iou_pred, dim=1)
            best_low_res_masks = low_res_masks[torch.arange(len(low_res_masks)), best_mask_idx]
            mask_buffer = best_low_res_masks
            # update coords and labels buffer
            if point_inputs is not None:
                coords_buffer = point_inputs[0]
                labels_buffer = point_inputs[1]
            # update boxes buffer
            if bbox_inputs is not None:
                boxes_buffer = bbox_inputs

        return output_mask_batch, target_masks.unsqueeze(0).repeat(11, 1, 1, 1)

    def train_one_epoch(
            self,
            epoch,
            step,
            ) -> None:
        self.train_sampler.set_epoch(epoch)
        logger.info(f"Epoch {epoch} started")
        for batch_idx, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            for idx in range(len(batch.images)):
                output_masks, target = self.predict_multi_step(
                    batch.images[idx], 
                    batch.masks[idx], 
                    batch.bboxes[idx],
                    )
                loss = self.loss_fn(output_masks, target)
                loss[CORE_LOSS_KEY].backward()
                logger.info(f"Epoch {epoch}, iter {batch_idx}, loss: {loss[CORE_LOSS_KEY].item()}")
                self.optimizer.step()    
                self.scheduler.step()

                if step % self.save_steps == 0:
                    torch.save(self.model.module.state_dict(), f"outputs/ckpts/model_{step}.pth")

                step += 1

    def train(self) -> None:
        self.model.module.train()
        step = 0
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch, step)