from typing import Union, List, Callable, Iterator

import torch
from nequip.data import AtomicDataDict
from torch.utils.data import IterableDataset


class AtomicDataset(torch.utils.data.Dataset):
    """The base class for all NequIP datasets.

    Subclasses must implement:

     - ``__len__()``
     - ``get_data_list()``

    Or may optionally directly override ``__getitem__`` and ``__getitems__`` at their own risk.

    Args:
        transforms (List[Callable]): list of data transforms
    """

    dtype: torch.dtype

    def __init__(
        self,
        transforms: List[Callable] = [],
    ):
        self.dtype = torch.get_default_dtype()
        self.transforms = transforms

    def __getitem__(
        self,
        index: Union[int, List[int], torch.Tensor, slice],
    ) -> AtomicDataDict.Type:
        if isinstance(index, slice):
            return self.__getitems__(index)
        elif isinstance(index, int):
            return self.__getitems__([index])[0]
        else:
            return self.__getitems__(index)

    def __getitems__(
        self,
        indices: Union[List[int], torch.Tensor, slice],
    ) -> List[AtomicDataDict.Type]:
        data_list: List[AtomicDataDict.Type] = self.get_data_list(indices)
        return [self._transform(data) for data in data_list]

    def _transform(self, x: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # TODO: this is very confusing
        # when training with a DataLoader, the transforms don't seem to mutate the underlying data
        # but if Datasets are called by index directly, e.g. dataset[[1,3]], the underlying dicts are mutated
        x = x.copy()
        for t in self.transforms:
            x = t(x)
        return x

    def num_atoms(self, indices: Union[List[int], torch.Tensor, slice]) -> List[int]:
        """
        Subclasses may override this.
        """
        # Note that get_data_list does _not_ call the transforms
        data_list = self.get_data_list(indices)
        return [AtomicDataDict.num_nodes(data) for data in data_list]

    def get_data_list(
        self,
        indices: Union[List[int], slice],
    ) -> List[AtomicDataDict.Type]:
        raise NotImplementedError(
            "Subclasses of AtomicDataset should define get_data_list"
        )


class IterativeAtomicDataset(IterableDataset):
    """An iterable Dataset with transformation functionality for atomic data.

    Args:
        transforms (List[Callable]): list of data transforms
    """

    dtype: torch.dtype

    def __init__(
        self,
        transforms: List[Callable] = [],
    ):
        self.dtype = torch.get_default_dtype()
        self.transforms = transforms

    def __iter__(self) -> Iterator[AtomicDataDict.Type]:
        for data in self.get_data_list():
            yield self._transform(data)

    def _transform(self, x: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = x.copy()
        for transform in self.transforms:
            x = transform(x)
        return x

    def num_atoms(self, indices: Union[List[int], torch.Tensor, slice]) -> List[int]:
        """
        Subclasses may override this.
        """
        data_list = self.get_data_list(indices)
        return [AtomicDataDict.num_nodes(data) for data in data_list]

    def get_data_list(self) -> List[AtomicDataDict.Type]:
        raise NotImplementedError("Subclasses should implement this method to return data list.")