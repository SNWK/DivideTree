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

## Experiment  

### Original MolGAN (without reward)  

#### Data preprocess  

- Area: andes_peru, [-9.0874, -77.5737]  
- Radius: 90 km  
- Size of each tree: 20 - 100 peaks
- Number of trees: 626

**Normalization**  
- latitude and longitude normalization:
  - map to [0, 1] for each tree
- elevation and prominence normalization:
  - map to [0, 1] for the whole 626 trees dataset, $newx = \frac{x - xmin}{xmax - xmin}$, in feet:
    - elevation min: 570 
    - elevation max: 22099 
    - prominence min: 101 
    - prominence max: 9065 

#### Training  

config:  
(TODO: a global config controller)  
- g_conv_dim = [128, 256, 512]  
- d_conv_dim = [[100, 64], 100, [100, 64]]  
- batch_size = 2  
- num_iters = 200000  

Training time: 1.5 hours  

#### Result  
##### original result 
after 100000 iterations:  
![100000Png](res/test.png)   

MST sample:  
46 nodes
![46nodesSample](res/molganSample46.png)  

Evaluation: 
- using pretrained model to generate 100 samples (fixed size: 100)
- for each sample：
  - retireve 20 real data(20-100 long) from datasets
  - let sample size = real data size, sample[:size]
  - compute tree edit distance, compute KL-divergency

tree edit distance:     
- 20 (which means no nodes same, remove&insert penalty = 20)  

KL-divergency(elevation, prominence):  
- average: 0.44337702, 0.29986525
- min: 0.14103942  0.08884087  

Compare to other methods:  
![hmmRnnEval](res/hmm&rnn.png)


## TODO  

1. Reward machine:  
- connectivity: 1 / number of strong connected subgraphs  
- bad edges ratio: (all edges - MST edges) / all edges  




