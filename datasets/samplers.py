import math
import os
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from typing import Optional, List


class DistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset in distributed training.
    This is useful with `torch.nn.parallel.DistributedDataParallel`.
    
    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of processes participating in distributed training.
        rank: Rank of the current process within num_replicas.
        shuffle: Whether to shuffle the dataset indices at the beginning of each epoch.
    """

    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        num_replicas: Optional[int] = None, 
        rank: Optional[int] = None, 
        shuffle: bool = True
    ):
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available")

        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self) -> iter:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample the indices for the current rank
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """
    Sampler for multi-node distributed training.
    
    Args:
        dataset: Dataset to sample from.
        num_replicas: Number of processes participating in distributed training.
        rank: Rank of the current process within num_replicas.
        local_rank: Rank of the current process within the local node.
        local_size: Number of processes in the local node.
        shuffle: Whether to shuffle the dataset indices at the beginning of each epoch.
    """

    def __init__(
        self, 
        dataset: torch.utils.data.Dataset, 
        num_replicas: Optional[int] = None, 
        rank: Optional[int] = None, 
        local_rank: Optional[int] = None, 
        local_size: Optional[int] = None, 
        shuffle: bool = True
    ):
        if not dist.is_available():
            raise RuntimeError("Distributed package is not available")

        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_size is None:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts
        self.shuffle = shuffle

    def __iter__(self) -> iter:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Filter indices for local nodes
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # Add extra samples to make it evenly divisible
        indices += indices[: (self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # Subsample indices based on rank and num_parts
        indices = indices[self.rank // self.num_parts : self.total_size_parts : self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
