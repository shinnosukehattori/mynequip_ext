import torch
import ase
import ase.io
import time

from nequip.data import from_dict, AtomicDataDict
from . import AtomicDataset

from typing import Union, Dict, List, Optional, Callable, Any

# TODO: link "standard keys" under `include_keys` to docs


class MmapDataset(AtomicDataset):
    r"""``AtomicDataset`` for `ASE <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_-readable file formats.

    Args:
        file_path (str): path to ASE-readable file
        transforms (List[Callable]): list of data transforms
    """

    def __init__(
        self,
        file_path: str,
        transforms: List[Callable] = [],
    ):
        super().__init__(transforms=transforms)
        self.file_path = file_path

        t0 = time.time()
        self.data_list: List[Dict] = torch.load(file_path, weights_only=False, mmap=True)
        print("Time to load the dataset:", time.time() - t0, "#", file_path)

    def __len__(self) -> int:
        return len(self.data_list)

    def get_data_list(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        if isinstance(indices, slice):
            return [from_dict(x) for x in self.data_list[indices]]
        else:
            return [from_dict(self.data_list[index]) for index in indices]
