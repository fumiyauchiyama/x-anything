from itertools import chain
import logging
import multiprocessing as mp
import os
from typing import List, Optional, Tuple, Union

import torch
from torch.distributed import ReduceOp
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


logger = logging.getLogger()

def dist_max(x: Union[int, float]):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.MAX)
    return tensor


def dist_mean(x: Union[int, float]):
    tensor = torch.tensor(x).cuda()
    dist.all_reduce(tensor, op=ReduceOp.AVG)
    return tensor


def dist_mean_dict(x):
    r = dict()
    for k in x:
        r[k] = dist_mean(x[k])
        r[k] = r[k].item() if (r[k].dim() == 0) else r[k].tolist()
    return r


def get_is_torch_run() -> bool:
    return os.environ.get('LOCAL_RANK') is not None


def get_is_mpi_run() -> bool:
    return os.environ.get('OMPI_COMM_WORLD_SIZE') is not None


def get_global_rank() -> int:
    if get_is_torch_run():
        return int(os.environ['RANK'])
    elif get_is_mpi_run():
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    else:
        return 0
    

def get_local_rank() -> int:
    if get_is_torch_run():
        return int(os.environ['LOCAL_RANK'])
    elif get_is_mpi_run():
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        return 0
    

def get_world_size() -> int:
    if get_is_torch_run():
        return int(os.environ['WORLD_SIZE'])
    elif get_is_mpi_run():
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    else:
        return 1
    

def get_is_master() -> bool:
    return get_global_rank() == 0


def setup_torch_distributed() -> None:
    local_rank = get_local_rank()

    if get_is_torch_run():
        logger.info(f"Run launched with torchrun, local rank: {local_rank}")
    elif get_is_mpi_run():
        logger.info(f"Run launched with MPI, local rank: {local_rank}")
    else:
        logger.info("Single GPU job")

    logger.info(f"ENV: {os.environ}")

    # set GPU device
    assert 0 <= local_rank

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(init_method="env://", backend="nccl")
