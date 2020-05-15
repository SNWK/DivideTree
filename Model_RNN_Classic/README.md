# Classic RNN Model

Using a 8 long subsequence of the divide tree's sequence representation to predict the next peak.  

```
python train.py ../data/regionTreeSeqs/andes_perulittle.txt --model lstm --n_epochs 10000 --print_every 500 --hidden_size 256 --n_layers 2 --learning_rate 0.001000 --chunk_len 9 --batch_size 50 --cuda
```

## Data preprocessing

1. Sample disk-like areas from the whole region area. Disk radius is 20 km.

   1. For each disk-like area sample, collect all peaks out and then build the tree structure. 

      ```
      rootNode = genDivideTree(peaks)
      -- construct a adjacent table contains distances between each pair of peaks
      -- using minimum spanning tree algorithm to build the tree, store them in a tree structure
      ```

   2. Using DFS algorithm to generate the sequence representations of the tree. 

      ```
      seqOfTree = genFullSeq(rootNode, isDFS=True)
      ```

   3. There are three types of node in the sequence:

      1. left parenthesis, value=[-1, 0, 0, 0, 0]

      2. peak, value[0, delta x, delta y, delta elevation, delta prominence]

         For example, 

         there is a sequence representation [ A [ B ] ], B = [0, x_b - x_a, y_b - y_a, ele_b - ele_a, pro_b - pro_a]

2. Then we get a sequence dataset, each sequence represent one sampled area

3. data standardization: using z-score method

   $\mu$  and $\sigma$ are the mean and variance over all peaks

4. training sample generation, call function `random_training_set(chunk_len, batch_size)`. Randomly choose a sequence, and then randomly choose a 9 long subsequence from it. First 8 as input, and the last one is what i want to predict.


## Model

A classic RNN model in Pytorch

Input: batch of training data. Each has 5 dimensions as mentioned.

output: 7 dimensions, 1-3 for classification, 4-7 for prediction.

Hidden size: 256

Layer num: 2 

input layer: full connection, 5*256

output layer: full connection, 256*7

loss function: 

```python
loss1 = torch.nn.functional.cross_entropy(output.view(args.batch_size, -1)[:, :3], tagTarget[:,c].squeeze())

loss2 = weighted_mse_loss(output.view(args.batch_size, -1)[:, 3:], target[:,c].squeeze(), weights[1:])

loss = loss1 + loss2
```

optimizer: The learning rate is reduced by 10 times every 500 epochs

```python
torch.optim.SGD(decoder.parameters(), lr=args.learning_rate,  momentum=0.9)
```

## Generation

1. choose randomly the beginning 8 node in the sequences dataset.

2. input them one by one to build up the hidden vector.

3. using the hidden vector from the seventh run and the eighth input to predict the next one.

   ```python
   for p in range(lastlen-1):
   	_, hidden = RNN(last8seq[:,p], hidden)
   # predict   
   inp = last8seq[:,-1]
   output, hidden = RNN(inp, hidden)
   ```

4. And then using the last 8 node to predict the next one. repeat.
