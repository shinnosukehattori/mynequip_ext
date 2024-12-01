#from ase import Atoms
from ase.io import read, write
from sklearn.model_selection import train_test_split
#from collections import Counter


prefixs= [
    "SPICE-2.0.1_subset1",
    "SPICE-2.0.1_subset2",
    "SPICE-2.0.1_subset3",
    "SPICE-2.0.1_subset4",
    "SPICE-2.0.1_subset5",
    "SPICE-2.0.1_subset6",
    "SPICE-2.0.1_subset7",
    "SPICE-2.0.1_subset8",
    "SPICE-2.0.1_subset9",
    "SPICE-2.0.1_subset10",
    "SPICE-2.0.1_subset11",
    "SPICE-2.0.1_subset12",
    "SPICE-2.0.1_subset13",
    "SPICE-2.0.1_subset14",
    "SPICE-2.0.1_subset15",
    "SPICE-2.0.1_subset16",
    "SPICE-2.0.1_subset17",
    "SPICE-2.0.1_subset18",
    "SPICE-2.0.1_subset19",
    "SPICE-2.0.1_subset20"
]

for prefix in prefixs:
    print("loading ... ,", prefix)
    X = read(f"{prefix}.extxyz", index=":")
    train, test_valid = train_test_split(X, test_size=0.2, random_state=1)
    test, valid = train_test_split(test_valid, test_size=0.5, random_state=1)
    print("saving ...")
    write(f"{prefix}_train.xyz", images=train, format="extxyz")
    write(f"{prefix}_test.xyz", images=test, format="extxyz")
    write(f"{prefix}_valid.xyz", images=valid, format="extxyz")


