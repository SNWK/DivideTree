#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from helpers import *
from model import *
import matplotlib.pyplot as plt

def generate(decoder, predict_len=20, cuda=False):
    ini_points = [[0,0]]
    hidden = decoder.init_hidden(1)
    prime_input = Variable(loc_tensor(ini_points).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = ini_points

    # Use priming string to "build up" hidden state
    for p in range(len(ini_points) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        dis_change = output.tolist()[0]
        # print(p, dis_change)
        # Add predicted character to string and use as next input
        predicted_loc = [predicted[-1][0] + dis_change[0], predicted[-1][1] + dis_change[1]]
        predicted.append(predicted_loc)
        inp = Variable(loc_tensor([dis_change]))
        if cuda:
            inp = inp.cuda()

    return predicted

def drawResult(name, path):
    x = [v[0] for v in path]
    y = [v[1] for v in path]
    for point in path:
        plt.scatter(x, y, color='b')
    X = []
    Y = []
    for i in range(len(path) - 1):
        X.append([x[i], x[i+1]])
        Y.append([y[i], y[i+1]])
    for i in range(len(X)):
        plt.plot(X[i], Y[i], color='r')
    plt.savefig('./resultImg/'+ name + '.png')
    plt.clf()
    
# Run as standalone script
if __name__ == '__main__':

# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename
    
    for i in range(10):
        result = generate(decoder, **vars(args))
        drawResult('test'+str(i), result)

