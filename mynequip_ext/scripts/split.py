#from ase import Atoms
from ase.io import read, write
from sklearn.model_selection import train_test_split
#from collections import Counter


prefix="SPICE-1.1.3"
print("loading ...")
X = read(f"{prefix}.xyz", index=":")
train, test_valid = train_test_split(X, test_size=0.2, random_state=1)
test, valid = train_test_split(test_valid, test_size=0.5, random_state=1)
print("saving ...")
write(f"{prefix}_train.xyz", images=train, format="extxyz")
write(f"{prefix}_test.xyz", images=test, format="extxyz")
write(f"{prefix}_valid.xyz", images=valid, format="extxyz")


