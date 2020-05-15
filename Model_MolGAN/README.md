# MolGAN
Pytorch implementation of MolGAN: An implicit generative model for small molecular graphs (https://arxiv.org/abs/1805.11973)  
This library refers to the following two source code.
* [nicola-decao/MolGAN](https://github.com/nicola-decao/MolGAN)
* [yunjey/StarGAN](https://github.com/yunjey/StarGAN)

## RUN

data generation:  
```
python generateDataset.py  

python dataGAN/sparse_molecular_dataset.py
```

train or test:  
```
python main.py
```

