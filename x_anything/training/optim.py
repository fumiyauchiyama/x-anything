from dataclasses import dataclass

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

@dataclass
class OptimizationArgs:
    lr: float = 8e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.1
    warmup_iters: int = 250
    total_iters: int = 90000
    decay_iter_1: int = 60000
    decay_iter_2: int = 86666


def get_optimizer(model, config: OptimizationArgs):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

    assert config.warmup_iters > 0
    assert config.warmup_iters < config.decay_iter_1 < config.decay_iter_2 < config.total_iters

    def lr_lambda(iter):
        if iter < config.warmup_iters:
            return float(iter) / config.warmup_iters
        elif iter < config.decay_iter_1:
            return 1.0
        elif iter < config.decay_iter_2:
            return 0.1
        else:
            return 0.01
    
    lr_scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, lr_scheduler
