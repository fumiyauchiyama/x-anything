# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from timm import create_model
from timm.data import create_transform

from .modeling import ImageEncoderViT, ImageEncoderTimmViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam


@dataclass
class ImageEncoderPreprocessArgs:
    img_size: int = 224
    interpolation: str = 'bicubic'
    is_training: bool = False
    no_aug: bool = True


@dataclass
class SamWithTimmViTArgs:
    model_name: str = 'vit_base_patch16_224'
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1

    preprocess: ImageEncoderPreprocessArgs = field(default_factory=ImageEncoderPreprocessArgs)

    vit_checkpoint: Optional[str] = None
    checkpoint: Optional[str] = None


def build_sam_with_timm(
    args: SamWithTimmViTArgs,
):
    vit = create_model(
        args.model_name,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
    )

    # Load checkpoint from OFDB
    # https://github.com/ryoo-nakamura/OFDB/blob/ea761ecb082756a374413704983f9e1f4acc5997/main.py#L157
    if args.vit_checkpoint is not None:
        ckpt = torch.load(args.vit_checkpoint, weights_only=True)
        if args.vit_checkpoint.endswith('.pth'):
            # ckpt_model = ckpt
            ckpt_model = ckpt['model']
        elif args.vit_checkpoint.endswith('.torch'):
            ckpt_model = ckpt['classy_state_dict']['base_model']['model']['trunk']
            ckpt_model['cls_token'] = ckpt_model.pop('class_token')
            ckpt_model['pos_embed'] = ckpt_model.pop('pos_embedding')
        
        state_dict = vit.state_dict()
            
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
            if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
                print(f'Remove key [{k}] from pretrained checkpoint')
                del ckpt_model[k]
        vit.load_state_dict(ckpt_model, strict=False)
        print(f'Checkpoint was loaded from {args.vit_checkpoint}\n')

    preprocess = create_transform(
        input_size=args.preprocess.img_size,
        interpolation=args.preprocess.interpolation,
        is_training=args.preprocess.is_training,
        no_aug=args.preprocess.no_aug,
    )

    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    sam = Sam(
        image_encoder=ImageEncoderTimmViT(
            vit=vit,
            img_size=image_size,
            out_chans=prompt_embed_dim,
            ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        vit_preprocess=preprocess,
    )
    
    if args.checkpoint is not None:
        with open(args.checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam