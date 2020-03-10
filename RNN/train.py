#!/usr/bin/env python
import sys
sys.path.append("..") 

import torch
import torch.nn as nn
from torch.autograd import Variable
import visdom
import argparse
import os

from tqdm import tqdm

from RNN.helpers import *
from RNN.model import *
from RNN.generate import *

# visdom setup
viz = visdom.Visdom(env='RNN')
epochn = 0
tagLoss = 0
last4loss = 0
totalloss = 0
wintag = viz.line(
    X=np.array([epochn]),
    Y=np.array([tagLoss]),
    opts=dict(title='tagLoss'))

winlast4 = viz.line(
    X=np.array([epochn]),
    Y=np.array([last4loss]),
    opts=dict(title='last4loss'))

winTotal = viz.line(
    X=np.array([epochn]),
    Y=np.array([totalloss]),
    opts=dict(title='totalloss'))



# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str, default="../data/regionTreeSeqs/andes_peru.txt")
argparser.add_argument('--model', type=str, default="lstm")
argparser.add_argument('--n_epochs', type=int, default=10000)
argparser.add_argument('--print_every', type=int, default=200)
argparser.add_argument('--hidden_size', type=int, default=512)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=1e-6)
argparser.add_argument('--chunk_len', type=int, default=9)
argparser.add_argument('--batch_size', type=int, default=50)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

allTrees, treesNum= read_file_zscore(args.filename)

def weighted_mse_loss(input, target, weight):
    if args.cuda:
        weight = torch.FloatTensor(weight).cuda()
    else:
        weight = torch.FloatTensor(weight)
    loss = torch.sum(weight * ((input - target) ** 2))
    
    return loss
    
def random_training_set(chunk_len, batch_size):
    inp = torch.FloatTensor(batch_size, chunk_len, 5)
    tagTarget = torch.LongTensor(batch_size, chunk_len, 1)
    target = torch.FloatTensor(batch_size, chunk_len, 4)
    for bi in range(batch_size):
        treeIndex = random.randint(0, treesNum-1)
        tree = allTrees[treeIndex]
        index = random.randint(0, len(tree) - chunk_len - 1)
        chunk = np.array(tree[index: index + chunk_len + 1])
        inp[bi] = info_tensor(chunk[:-1])
        # sequence -> one
        # using last chunk_size-1 node to predict the next one
        target[bi] = info_tensor(chunk[1:, 1:])
        tagTarget[bi] = torch.LongTensor(chunk[1:, :1])
        
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
        tagTarget = tagTarget.cuda()
    return inp, target, tagTarget

def train(inp, target, tagTarget, weights):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        if args.model == 'gru':
            hidden = hidden.cuda()
        else:
            hidden = (hidden[0].cuda(), hidden[0].cuda())
    decoder.zero_grad()
    loss1 = 0
    loss2 = 0
    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss1 += torch.nn.functional.cross_entropy(output.view(args.batch_size, -1)[:, :3], tagTarget[:,c].squeeze())
        loss2 += weighted_mse_loss(output.view(args.batch_size, -1)[:, 3:], target[:,c].squeeze(), weights[1:])

    loss = loss1 + loss2
    loss.backward()
    decoder_optimizer.step()


    tagLoss = min(loss1.data[0].item()/args.chunk_len, 1)
    last4loss = min(loss2.data[0].item()/args.chunk_len, 2)
    totalloss = min(loss.data[0].item()/args.chunk_len, 3)

    viz.line(
        X=np.array([epochn]),
        Y=np.array([tagLoss]),
        win=wintag,
        update='append')

    viz.line(
        X=np.array([epochn]),
        Y=np.array([last4loss]),
        win=winlast4,
        update='append')

    viz.line(
        X=np.array([epochn]),
        Y=np.array([totalloss]),
        win=winTotal,
        update='append')

    return loss.data[0].item()/args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


if __name__ == "__main__":
    # Initialize models and start training

    decoder = RNN(
        5, # input size
        args.hidden_size,
        7, # output size
        model=args.model,
        n_layers=args.n_layers,
    )
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=args.learning_rate,  momentum=0.9)
    # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
    # criterion = nn.MSELoss() #  nn.CrossEntropyLoss() 

    if args.cuda:
        decoder.cuda()

    start = time.time()
    all_losses = []
    loss_avg = 0
    weight = [1, 0.01, 0.01, 0.01, 0.01] # weights of MSE

    try:
        print("Training for %d epochs..." % args.n_epochs)
        for epoch in tqdm(range(1, args.n_epochs + 1)):
            epochn = epoch
            loss = train(*random_training_set(args.chunk_len, args.batch_size), weight)
            loss_avg += loss
            if epoch%2000 == 0:
                # decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate*0.1)
                decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=args.learning_rate*0.1,  momentum=0.9)
            if epoch % args.print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
                v, _ = generate(decoder, 20, cuda=args.cuda)
                print(v,'\n\n')

        print("Saving...")
        save()

    except KeyboardInterrupt:
        print("Saving before quit...")
        save()

