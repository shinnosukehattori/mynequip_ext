from scipy.spatial import KDTree
import numpy as np
import itertools
import h5py
import ase
import ase.io
from ase.units import Bohr, Hartree, Angstrom, eV
import typer
from rich import print
from rich.table import Table
from rich.progress import Progress
from typing import Optional
#from nequip.data import AtomicDataDict as ADD

Hartree2eV = Hartree/eV
bohr2ang = Bohr/Angstrom

class SubsetsVisitor:
    def __init__(self):
        self.result = set()

    def __call__(self, name, ob):
        if isinstance(ob, h5py.Group) and "subset" in ob.keys():
            self.result.add(ob["subset"][0].decode("utf-8"))


def all_subsets(group):
    visitor = SubsetsVisitor()
    group.visititems(visitor)
    return visitor.result


# here's the result if visiting the spice HDF5 file and getting
# the names of all subsets
Default_datasets_ = {
    "SPICE Dipeptides Single Points Dataset v1.2",
    "SPICE DES370K Single Points Dataset v1.0",
    "SPICE PubChem Set 6 Single Points Dataset v1.2",
    "SPICE Ion Pairs Single Points Dataset v1.1",
    "SPICE Solvated Amino Acids Single Points Dataset v1.1",
    "SPICE PubChem Set 4 Single Points Dataset v1.2",
    "SPICE PubChem Set 5 Single Points Dataset v1.2",
    "SPICE DES370K Single Points Dataset Supplement v1.0",
    "SPICE DES Monomers Single Points Dataset v1.1",
    "SPICE PubChem Set 2 Single Points Dataset v1.2",
    "SPICE PubChem Set 1 Single Points Dataset v1.2",
    "SPICE PubChem Set 3 Single Points Dataset v1.2",
}

Default_datasets = {
    "SPICE PubChem Set 1 Single Points Dataset v1.3",
    "SPICE PubChem Set 2 Single Points Dataset v1.3",
    "SPICE PubChem Set 3 Single Points Dataset v1.3",
    "SPICE PubChem Set 4 Single Points Dataset v1.3",
    "SPICE PubChem Set 5 Single Points Dataset v1.3",
    "SPICE PubChem Set 6 Single Points Dataset v1.3",
    "SPICE PubChem Set 7 Single Points Dataset v1.0",
    "SPICE PubChem Set 8 Single Points Dataset v1.0",
    "SPICE PubChem Set 9 Single Points Dataset v1.0",
    "SPICE PubChem Set 10 Single Points Dataset v1.0",
    "SPICE PubChem Boron Silicon v1.0",
    "SPICE DES370K Single Points Dataset v1.0",
    "SPICE DES370K Single Points Dataset Supplement v1.1",
    "SPICE DES Monomers Single Points Dataset v1.1",
    "SPICE Water Clusters v1.0",
    "SPICE Dipeptides Single Points Dataset v1.3",
    "SPICE Amino Acid Ligand v1.0",
    "SPICE Solvated Amino Acids Single Points Dataset v1.1",
    "SPICE Solvated PubChem Set 1 v1.0",
    "SPICE Ion Pairs Single Points Dataset v1.2",
}


class GroupsInSubsetVisitor:
    def __init__(self, subset, transform_func=lambda x: x):
        self.subset = subset.encode("utf-8")
        self.transform_func = transform_func
        self.result = set()

    def __call__(self, name, ob):
        if (
            isinstance(ob, h5py.Group)
            and "subset" in ob.keys()
            and ob["subset"][0] == self.subset
        ):
            self.result.add(self.transform_func(ob))


def groups_in_subset_via_visitor(group, subset):
    visitor = GroupsInSubsetVisitor(subset)
    group.visititems(visitor)
    return visitor.result


def groups_in_subset(group, subset) -> list:
    keys = list(group.keys())
    bsubset = subset.encode("utf-8")
    result = []
    with Progress() as progress:
        task = progress.add_task(
            f"Finding molecules in subset {subset}", total=len(keys)
        )
        for key in keys:
            g = group[key]
            if g["subset"][0] == bsubset:
                result.append(g)
            progress.update(task, advance=1, description=f"Checking {key}")
    return result


def molecules_from_group(group):
    # a group here represents a top-level group in the
    # spice hdf5 file, representing an atom.  The
    # top level groups have the following related datasets,
    # see https://github.com/openmm/spice-dataset/tree/main/downloader
    # subset
    # smiles (canonical smiles string incl hydrogens and atom indices)
    # atomic_numbers (array length n containing atomic number,
    #   following indices in SMILES string)
    # conformations (array of shape (M,N,3) containing coordinates for M
    #   conformers
    # formation_energy (array of length M containing total energy of each
    #   conformation)
    # dft_total_energy (array of length M containing energy of each conformation)
    # dft_total_gradient (array of (M,N,3) containing gradient of energy, ie force)
    #
    # SPICE docs say "dft total energy" and "dft total gradient" are the most
    # commonly used ones, although sample allegro config (minimal.yml) wants
    # atomic_numbers, total_energy, forces, pos, so attempting this mapping
    num_conformers = group["conformations"].shape[0]
    result = []
    for i in range(num_conformers):
        mol = ase.Atoms(
            numbers=group["atomic_numbers"], positions=group["conformations"][i]
        )
        # adding the total energy as an info here makes this appear
        # in the "Properties" comment line of the extended XYZ format.  The
        # nequip training wants "total energy" but it looks as if the ASE loader
        # maps "energy" to "total energy"
        # see https://github.com/mir-group/nequip/blob/c56f48fcc9b4018a84e1ed28f762fadd5bc763f1/nequip/data/dataset.py#L808
        #mol.info["energy"] = group["dft_total_energy"][i]
        # adding this here sets the forces to come as additional XYZ coords per line

        mol.positions *= bohr2ang
        mol.arrays["forces"] = - group["dft_total_gradient"][i] * (Hartree2eV / bohr2ang)
        if "mbis_charges" in group:
            mol.arrays["charge"] = group["mbis_charges"][i]
        mol.info["energy"] = group["dft_total_energy"][i] * Hartree2eV
        mol.info["energy_formation"] = group["formation_energy"][i] * Hartree2eV
        mol.info["subset"] = group["subset"][0].decode("utf-8")
        mol.info["smiles"] = group["smiles"][0].decode("utf-8")
        result.append(mol)
    return result


app = typer.Typer()
hf = h5py.File("./SPICE-2.0.1.hdf5")
#hf = h5py.File("small.hdf5")
#hf = h5py.File("mid.hdf5")
#hf.visititems(lambda name, ob: print(name, ob))


@app.command()
def group(group: str, out: str = "-"):
    mols = molecules_from_group(hf[group])
    ase.io.write(out, mols, format="extxyz")


def subset_table() -> Table:
    result = Table("index", "Subset name")
    datasets = sorted(Default_datasets)
    for i, subset in enumerate(datasets):
        result.add_row(str(i + 1), subset)
    return result


def check_subset(subset: str):
    if subset not in Default_datasets:
        print("[bold red]Error:[/bold red]\n")
        print("Subset must be one of the following subsets")
        print(subset_table())
        raise typer.Exit()


@app.command()
def list_subsets():
    print(subset_table())


def get_subset_by_index_or_name(subset: str) -> str:
    """Gets the subset by index or name"""
    datasets = sorted(Default_datasets)
    if subset.isdigit() and int(subset) in range(1, len(datasets) + 1):
        return datasets[int(subset) - 1]
    else:
        return subset


@app.command()
def subset(subset: str, out: str = "-"):
    subset = get_subset_by_index_or_name(subset)
    check_subset(subset)
    print(f"Exporting subset {subset}")
    groups = groups_in_subset(hf, subset)
    results = []
    with Progress() as progress:
        task = progress.add_task("Adding molecules", total=len(groups))
        results = []
        for g in groups:
            results.append(molecules_from_group(g))
            progress.update(task, advance=1, description=f"Adding molecules: {g.name}")
        final_results = itertools.chain.from_iterable(results)
    ase.io.write(out, final_results, format="extxyz")


def r_min(positions, n_neighbors) -> float:
    kdtree = KDTree(positions)
    distances, _ = kdtree.query(positions, n_neighbors + 1)
    result = np.max(distances)
    return result


@app.command()
def subset_metrics(subset: str, neighbors: int = 1):
    subset = get_subset_by_index_or_name(subset)
    check_subset(subset)
    print(f"Finding metrics of subset {subset}")
    groups = groups_in_subset(hf, subset)
    with Progress() as progress:
        task = progress.add_task("Checking molecules", total=len(groups))
        atomic_symbols: set[str] = set()
        min_r = 0.0
        num_conformers = 0
        for g in groups:
            mols = molecules_from_group(g)
            num_conformers = num_conformers + len(mols)
            for mol in mols:
                this_symbols = set(mol.get_chemical_symbols())
                atomic_symbols = atomic_symbols | this_symbols
                this_r = r_min(mol.get_positions(), neighbors)
                min_r = max(min_r, this_r)
            progress.update(task, advance=1, description=f"Checking molecules: {g.name}")
    # print table of results
    table = Table("Metric", "Value")
    table.add_row("Atomic Symbols", str(sorted(atomic_symbols)))
    table.add_row(f"r_cutoff ({neighbors} neighbors)", str(min_r))
    table.add_row("Num base molecules", str(len(groups)))
    table.add_row("Total num conformers", str(num_conformers))
    print(table)


if __name__ == "__main__":
    app()
