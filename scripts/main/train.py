import logging
import os
import json
from omegaconf import OmegaConf

from x_anything.training.logger import init_logger
from x_anything.training.distributed import setup_torch_distributed, get_global_rank, get_world_size
from scripts.main.trainer import SamTrainer, TrainerArgs

logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.structured(TrainerArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    setup_torch_distributed()
    init_logger()
    trainer = SamTrainer(
        cfg,
        world_size=get_world_size(),
        rank=get_global_rank()
        )
    trainer.train()

if __name__ == "__main__":
    main()