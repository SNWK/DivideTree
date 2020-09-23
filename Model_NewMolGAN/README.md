# NewMolGAN


# data generation  

```
debug/prominence 100 102 282 284  -i ../data -o ../output -f "SRTM"  

debug/merge_divide_trees andes_peru ../output/prominence--10--78-divide_tree.dvt ../output/prominence--11--78-divide_tree.dvt  ../output/prominence--10--77-divide_tree.dvt  ../output/prominence--11--77-divide_tree.dvt  

debug/isolation -i ../data -o ../output  -- -11 -9  -78 -75  

python processs.py  
```