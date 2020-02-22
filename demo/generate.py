#!/usr/bin/env python

import torch
import os
import argparse
import copy
import numpy as np

from helpers import *
from model import *
import matplotlib.pyplot as plt

predict_len=25
temperature=0.8
maxprelen = 8
# threshould = 0.18

minlen = 2
lennow = 0

def dfsaTree(seq, node, filename, threshould, cuda=False):
    global lennow
    if seq == []:
        lennow = 0
    p = copy.deepcopy(seq)
    p.append(node)
    result = []
    
    if len(p) < maxprelen:
        ini_points = p
    else:
        threshould = 0.21
        ini_points = p[-maxprelen:]
        
    if len(p) > predict_len:
        result.append(p)
        return result
    
    with torch.no_grad():
        decoder = torch.load(filename)
        hidden = decoder.init_hidden(1)
        prime_input = Variable(loc_tensor(ini_points).unsqueeze(0))
        if cuda:
            hidden = hidden.cuda()
            prime_input = prime_input.cuda()

        # Use priming string to "build up" hidden state
        for step in range(len(ini_points) - 1):
            _, hidden = decoder(prime_input[:,step], hidden)

        inp = prime_input[:,-1]
        output, hidden = decoder(inp, hidden)
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 3, replacement=True).tolist()
        del decoder
        torch.cuda.empty_cache()
    
    confidence = np.array(output_dist.tolist()) / sum(output_dist.tolist())
    hh = 0
    for label in set(top_i):
        # fake
        if confidence[label] < threshould:
            continue
        hh += 1
        result += dfsaTree(p, index2change(label), filename, threshould, cuda)
        
    if hh == 0 and lennow < minlen:
        result += dfsaTree(p, index2change(top_i[0]), filename, threshould, cuda)
        lennow += 1
    return result

def generateTree(decoder, predict_len=20, cuda=False, temperature=0.8):
    
    directions = [[[0, -0.003], [0, -0.003], [0, -0.003], [0, -0.003],[0, -0.003]], 
                  [[-0.003, 0.003],[-0.003, 0.003],[-0.003, 0.003],[-0.003, 0.003],[-0.003, 0.003]], 
                  [[0.003, 0.003],[0.003, 0.003], [0.003, 0.003], [0.003, 0.003] ,[0.003, 0.003]]]
    tree = []
    for i in range(len(directions)):
        ini_points = [[0,0]]
        ini_points += directions[i]
        hidden = decoder.init_hidden(1)
        prime_input = Variable(loc_tensor(ini_points).unsqueeze(0))
        if cuda:
            hidden = hidden.cuda()
            prime_input = prime_input.cuda()
        predicted = [[0,0]]

        # Use priming string to "build up" hidden state
        for p in range(len(ini_points) - 1):
            if p != 0:
                predicted.append([predicted[-1][0] + ini_points[p][0], predicted[-1][1] + ini_points[p][1]])
            _, hidden = decoder(prime_input[:,p], hidden)

        inp = prime_input[:,-1]

        for p in range(predict_len):
            output, hidden = decoder(inp, hidden)

            # dis_change = output.tolist()[0]
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            
            dis_change = index2change(top_i)

            # print(p, dis_change)
            # Add predicted character to string and use as next input
            predicted_loc = [predicted[-1][0] + dis_change[0], predicted[-1][1] + dis_change[1]]
            predicted.append(predicted_loc)
            inp = Variable(loc_tensor([dis_change]))
            if cuda:
                inp = inp.cuda()
        tree.append(predicted)
    return tree

def generate(decoder, predict_len=20, cuda=False, temperature = 0.8):
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

        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        dis_change = index2change(int(top_i))

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
    argparser.add_argument('-l', '--predict_len', type=int, default=10)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    decoder.flatten_parameters()
    del args.filename
    
#     for i in range(10):
#         result = generate(decoder, **vars(args))
#         drawResult('test'+str(i), result)
    result = dfsaTree([[0,0.01], [0,0.01], [0,0.01], [0,0.01]], [0,0.01], decoder, True)

