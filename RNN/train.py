#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str, default="../data/regionTreeSeqs/andes_peru.txt")
argparser.add_argument('--model', type=str, default="lstm")
argparser.add_argument('--n_epochs', type=int, default=10000)
argparser.add_argument('--print_every', type=int, default=200)
argparser.add_argument('--hidden_size', type=int, default=512)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=1e-6)
argparser.add_argument('--chunk_len', type=int, default=19)
argparser.add_argument('--batch_size', type=int, default=50)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

allTrees, treesNum= read_file(args.filename)

def weighted_mse_loss(input, target, weight):
    if args.cuda:
        weight = torch.FloatTensor(weight).cuda()
    else:
        weight = torch.FloatTensor(weight)
    loss = torch.sum(weight * ((input - target) ** 2))
    
    return loss
    
def random_training_set(chunk_len, batch_size):
    inp = torch.FloatTensor(batch_size, chunk_len, 5)
    target = torch.FloatTensor(batch_size, chunk_len, 5)
    for bi in range(batch_size):
        treeIndex = random.randint(0, treesNum-1)
        tree = allTrees[treeIndex]
        index = random.randint(0, len(tree) - chunk_len - 1)
        chunk = tree[index: index + chunk_len + 1]
        inp[bi] = info_tensor(chunk[:-1])
        target[bi] = info_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target, weights):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        if args.model == 'gru':
            hidden = hidden.cuda()
        else:
            hidden = (hidden[0].cuda(), hidden[0].cuda())
    decoder.zero_grad()
    loss = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
#         print(output.view(args.batch_size, -1).size())
#         print(target[:,c].size())
        loss += weighted_mse_loss(output.view(args.batch_size, -1), target[:,c].squeeze(), weights)

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = RNN(
    5, # input size
    args.hidden_size,
    5, # output size
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
# criterion = nn.MSELoss() #  nn.CrossEntropyLoss() 

weight = [1, 1, 1, 1, 1]

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0


try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size), weight)
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            v, _ = generate(decoder, 20, cuda=args.cuda)
            print(v)

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

