
import torch
import ase
import ase.io

from nequip.data.AtomicDataDict import from_dict
from nequip.data import from_ase_to_dict

from typing import Union, Dict, List, Optional, Callable, Any


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
    print("converting from AtomDataDict to Torch", str(ncpu))
    data_list = [from_dict(x) for x in tqdm(data_list)]
    print("DONE!")
    return data_list

    
if __name__ == "__main__":
    #do_convert = False
    do_convert = True
    targets = [
                #["SPICE-1.1.3_train.xyz", "train.pt"],
                #["SPICE-1.1.3_test.xyz", "test.pt"],
                #["SPICE-1.1.3_valid.xyz", "valid.pt"],
                ["SPICE-2.0.1_train.xyz", "train.pt"],
                ["SPICE-2.0.1_test.xyz", "test.pt"],
                ["SPICE-2.0.1_valid.xyz", "valid.pt"],
            ]

    if do_convert:
        for (target, dumped) in targets:
            # Load the atoms from the ASE trajectory
            atoms_list = ase.io.read(filename=target, index=":", parallel=False)
            print(atoms_list)
            key_mapping = {}
            #key_mapping = {
            #    "positions": "positions",
            #    "numbers": "atom_type",
            #    "cell": "cell",
            #    "pbc": "pbc",
            #}
            include_keys = ["energy"]
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
    else:
        import time

        dumped = targets[0][1]
        t0 = time.time()
        dataset = torch.load(dumped)
        print(dataset[0])
        print(dataset[-1])
        print("Time to load the dataset:", time.time() - t0)
