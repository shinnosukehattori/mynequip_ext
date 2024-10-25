from nequip.data.datamodule._base_datamodule import NequIPDataModule

from omegaconf import ListConfig, DictConfig, OmegaConf
from typing import Union, List, Callable, Optional, Dict


class MmapDataModule(NequIPDataModule):
    """LightningDataModule for `Dumped ASE <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`.

    Interface similar to ``nequip.data.datamodule.NequIPDataModule``, except that all the datasets are given in terms of paths to relevant ASE-readable files.

    Args:
        seed (int): data seed for reproducibility
        train_file_path (str/List[str]): path to training dataset file
        val_file_path (str/List[str]): path(s) to validation dataset file
        test_file_path (str/List[str]): path(s) to test dataset file
        predict_file_path (str/List[str]): path(s) to prediction dataset file
        split_dataset (Dict/List[Dict]): dictionary or list of dictionaries with a ``file_path`` key, which is the path to the ASE-readable dataset file and the keys ``train``, ``val``, ``test``, ``predict`` which represent the subsets to split the dataset into and are either ``int`` s that sum up to the size of ``dataset`` or ``float`` s that sum up to 1 (at least 2, but not necessarily all of ``train``, ``val``, ``test``, ``predict`` must be provided if this option is used)
        transforms (List[Callable]): list of data transforms
        train_dataloader_kwargs (Dict): arguments of the training ``DataLoader``
        val_dataloader_kwargs (Dict): arguments of the validation ``DataLoader``
        test_dataloader_kwargs (Dict): arguments of the testing ``DataLoader``
        predict_dataloader_kwargs (Dict): arguments of the prediction ``DataLoader``
        stats_manager (Dict): dictionary that can be instantiated into a ``nequip.data.DataStatisticsManager`` object
    """

    def __init__(
        self,
        seed: int,
        # file paths
        train_file_path: Optional[Union[str, List[str]]] = [],
        val_file_path: Optional[Union[str, List[str]]] = [],
        test_file_path: Optional[Union[str, List[str]]] = [],
        predict_file_path: Optional[Union[str, List[str]]] = [],
        split_dataset: Optional[Union[Dict, List[Dict]]] = [],
        # data transforms
        transforms: List[Callable] = [],
        # dataloader kwargs
        train_dataloader_kwargs: Union[dict, DictConfig] = {},
        val_dataloader_kwargs: Union[dict, DictConfig] = {},
        test_dataloader_kwargs: Union[dict, DictConfig] = {},
        predict_dataloader_kwargs: Union[dict, DictConfig] = {},
        stats_manager: Optional[Dict] = None,
    ):

        # == first convert all dataset paths to lists if not already lists ==
        dataset_paths = []
        for paths in [
            train_file_path,
            val_file_path,
            test_file_path,
            predict_file_path,
            split_dataset,
        ]:
            # convert to primitives as later logic is based on types
            if isinstance(paths, ListConfig) or isinstance(paths, DictConfig):
                paths = OmegaConf.to_container(paths, resolve=True)
            assert (
                isinstance(paths, list)
                or isinstance(paths, str)
                or isinstance(paths, dict)
            )
            if not isinstance(paths, list):
                # convert str -> List[str]
                dataset_paths.append([paths])
            else:
                dataset_paths.append(paths)

        # == assemble config template ==
        dataset_config_template = {
            "_target_": "mynequip_ext.data.dataset.MmapDataset",
            "transforms": transforms,
        }

        # == populate train, val, test predict, split datasets ==
        dataset_configs = [[], [], [], []]
        for config, paths in zip(dataset_configs, dataset_paths[:-1]):
            for path in paths:
                dataset_config = dataset_config_template.copy()
                dataset_config.update({"file_path": path})
                config.append(dataset_config)

        # == populate split dataset ==
        split_config = []
        for path_and_splits in dataset_paths[-1]:
            assert (
                "file_path" in path_and_splits
            ), "`file_path` key must be present in each dict of `split_dataset`"
            dataset_config = dataset_config_template.copy()
            file_path = path_and_splits.pop("file_path")
            dataset_config.update({"file_path": file_path})
            path_and_splits.update(
                {"dataset": dataset_config}
            )  # now actually dataset_and_splits
            split_config.append(path_and_splits)

        super().__init__(
            seed=seed,
            train_dataset=dataset_configs[0],
            val_dataset=dataset_configs[1],
            test_dataset=dataset_configs[2],
            predict_dataset=dataset_configs[3],
            split_dataset=split_config,
            train_dataloader_kwargs=train_dataloader_kwargs,
            val_dataloader_kwargs=val_dataloader_kwargs,
            test_dataloader_kwargs=test_dataloader_kwargs,
            predict_dataloader_kwargs=predict_dataloader_kwargs,
            stats_manager=stats_manager,
        )
