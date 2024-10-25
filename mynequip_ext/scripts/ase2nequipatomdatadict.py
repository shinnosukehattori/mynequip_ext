import numpy as np
import time
import torch
import ase
import ase.io

from mynequip_ext.data import from_ase_to_dict

from typing import Union, Dict, List, Optional, Callable, Any
from ase.units import Bohr, Hartree, Angstrom, eV

Hartree2eV = Hartree/eV
maxforce = 0.25 * Hartree2eV


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
    from tqdm import tqdm

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
                #["SPICE-1.1.3_train.xyz", "train.pt"],
                #["SPICE-1.1.3_test.xyz", "test.pt"],
                #["SPICE-1.1.3_valid.xyz", "valid.pt"],
                #["SPICE-2.0.1_valid.xyz", "valid.pt"],
                #["SPICE-2.0.1_train.xyz", "train.pt"],
                ["SPICE-2.0.1_test.xyz", "test.pt"],
                #["train_1.xyz", "dataset.pt"],
            ]

    for (target, dumped) in targets:
        # Load the atoms from the ASE trajectory
        atoms_list = ase.io.iread(filename=target, index=":", parallel=False)
        atoms_list = [ ase_data for ase_data in atoms_list if np.abs(ase_data.calc.results["forces"]).max() < maxforce]
        key_mapping = {}
        key_mapping = {
            "energy_formation": "total_energy",
            #"cell": "cell",
            #"energy": "total_energy",
            #"positions": "pos",
            #"numbers": "atom_type",
            #"cell": "cell",
            #"pbc": "pbc",
        }
        #include_keys = ["energy_formation"]
        include_keys = []
        exclude_keys = ["charges"]

        data_list = parallel_read_and_process(
            atoms_list=atoms_list,
            key_mapping=key_mapping,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            ncpu=24,
        )

        # Create the dataset
        torch.save(data_list, dumped)
        print("Saved dataset to", dumped, "with", len(data_list), "entries", "from", target)
        print("keys:", data_list[0].keys())

        require_keys = ['pos', 'cell', 'pbc', 'atomic_numbers', 'forces', 'total_energy']
        assert all([key in data_list[0].keys() for key in require_keys])

        t0 = time.time()
        dataset = torch.load(dumped, weights_only=False, mmap=True)
        print("Time to load the dataset:", time.time() - t0)
