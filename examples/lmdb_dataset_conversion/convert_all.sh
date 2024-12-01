
ary="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
for i in $ary
do
    echo $i
    python h5py_to_extxyz.py subset $i --out SPICE-2.0.1_subset${i}.extxyz
done

#python split.py

#cat *_test.extxyz > SPICE-2.0.1_test.extxyz
#cat *_train.extxyz > SPICE-2.0.1_train.extxyz
#cat *_valid.extxyz > SPICE-2.0.1_valid.extxyz

#python example_lmdb_conversion.py
