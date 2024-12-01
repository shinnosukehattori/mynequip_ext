"""
Example of how to use NequIPLMDBDataset to convert an xyz file to LMDB format, using ase and some nequip ase utilities. This can be adapted to convert data from other formats, as long as there one writes code to convert data from the custom format to nequip's AtomicDataDict format.
"""

import numpy as np
import torch
import ase
import ase.io
from nequip.data.dataset import NequIPLMDBDataset
#from nequip.data.ase import from_ase
from nequip.data.dict import from_dict
from mynequip_ext.data import from_ase_to_dict
#from nequip.utils import download_url
import time
import torch


from typing import Union, Dict, List, Optional, Callable, Any
from ase.units import Bohr, Hartree, Angstrom, eV

from tqdm import tqdm



# make sure the data is saved in float64!
torch.set_default_dtype(torch.float64)



Hartree2eV = Hartree/eV
maxforce = 0.25 * Hartree2eV

map_50GB = 53687091200
map_5GB  =  5368709120
map_10GB = 10737418240
map_2GB  =  2147483648
map_1GB  =  1073741824

def process_atoms(args):
    atoms, key_mapping, include_keys, exclude_keys = args
    return from_ase_to_dict(
        atoms=atoms,
        key_mapping=key_mapping,
        include_keys=include_keys,
        exclude_keys=exclude_keys,
    )

def parallel_read_and_process(atoms_list, key_mapping, include_keys, exclude_keys, ncpu=4):
    import torch.multiprocessing as mp

    data_list = []
    print("converting from ase to AtomDataDict with mp: ncpu=", str(ncpu))
    ctx = mp.get_context("forkserver")
    with ctx.Pool(ncpu) as pool, tqdm(total=len(atoms_list)) as pbar:
        args = [(atoms, key_mapping, include_keys, exclude_keys) for atoms in atoms_list]
        for results in pool.imap(process_atoms, args):
            pbar.update()
            pbar.refresh()
            data_list.extend([results])
    print("DONE!")
    return data_list


if __name__ == "__main__":
    targets = [
                ["SPICE-2.0.1_test.xyz",  "SPICE-2.0.1_test.lmdb",  map_2GB],
                ["SPICE-2.0.1_train.xyz", "SPICE-2.0.1_train.lmdb", map_10GB],
                ["SPICE-2.0.1_valid.xyz", "SPICE-2.0.1_valid.lmdb", map_2GB],
            ]

    for (target, dumped, map_size) in targets:
        # Load the atoms from the ASE trajectory
        atoms_list = list(
            tqdm(
            ase.io.iread(filename=target, index=":", parallel=False),
            desc="Reading dataset with ASE...",
            )
        )
        #ignore large force data
        atoms_list = [ ase_data for ase_data in atoms_list if np.abs(ase_data.calc.results["forces"]).max() < maxforce]

        key_mapping = {}
        key_mapping = {
            "energy_formation": "total_energy",
        }
        include_keys = []
        exclude_keys = ["charges"]

        data_dicts = parallel_read_and_process(
            atoms_list=atoms_list,
            key_mapping=key_mapping,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            ncpu=24,
        )
        atomic_data_dicts = [from_dict(data) for data in data_dicts]

        # Create the dataset
        NequIPLMDBDataset.save_from_iterator(
            file_path=dumped,
            iterator=atomic_data_dicts,
            map_size = map_size,
            write_frequency=5000,  # increase this from default 1000 to speed up writing of very large datasets
        )
        print("Saved dataset to", dumped, "with", len(data_dicts), "entries", "from", target)
        print("keys:", data_dicts[0].keys())

        require_keys = ['pos', 'cell', 'pbc', 'atomic_numbers', 'forces', 'total_energy']
        assert all([key in data_dicts[0].keys() for key in require_keys])

