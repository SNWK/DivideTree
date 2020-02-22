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
argparser.add_argument('filename', type=str, default="../data/demoData/andes_peru.txt")
argparser.add_argument('--model', type=str, default="lstm")
argparser.add_argument('--n_epochs', type=int, default=3000)
argparser.add_argument('--print_every', type=int, default=500)
argparser.add_argument('--hidden_size', type=int, default=1024)
argparser.add_argument('--n_layers', type=int, default=5)
argparser.add_argument('--learning_rate', type=float, default=0.1)
argparser.add_argument('--chunk_len', type=int, default=8)
argparser.add_argument('--batch_size', type=int, default=500)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

allSeq, allSeqLen= read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    inp = torch.FloatTensor(batch_size, chunk_len, 2)
    target = torch.LongTensor(batch_size, chunk_len, 1)
    for bi in range(batch_size):
        index = random.randint(0, allSeqLen-1000)
        chunk = allSeq[index]
        inp[bi] = loc_tensor(chunk[:-1])
        target[bi] = change2onehot(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
#         print(output.view(args.batch_size, -1).size())
#         print(target[:,c].size())
        loss += criterion(output.view(args.batch_size, -1), target[:,c].squeeze())

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training

decoder = DemoRNN(
    2, # input size
    args.hidden_size,
    100, # output size
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss() #  nn.MSELoss() 

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            drawResult(str(epoch), generate(decoder, 10, cuda=args.cuda))

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

