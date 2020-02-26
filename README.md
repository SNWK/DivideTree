# DivideTree

## MCMC model


MCMC/MCMC.ipynb: includes analysis, mcmc procedure, generate and eveluation

Using Pymc3 library


## RNN model

RNN/train.py: model initialization and training process

RNN/model.py: RNN structure definition

RNN/generate.py: tree generation method. For convenience, the output will include a symbol sequence and the real prediction sequence.

RNN/helpers.py: includes some processing methods

Train:
```
python train.py ../data/regionTreeSeqs/andes_perulittle.txt --model lstm --n_epochs 10000 --print_every 1000 --hidden_size 256 --n_layers 2 --learning_rate 0.01 --chunk_len 9 --batch_size 50 --cuda
```

Generate:
```
python generate.py andes_peru.pt --cuda
```
