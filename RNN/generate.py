#!/usr/bin/env python

import torch
import os
import argparse
import copy
import numpy as np

from helpers import *
from model import *
import matplotlib.pyplot as plt
import sys
sys.path.append("..") 
from utils.divtree_gen import *

predict_len=25
temperature=0.8
maxprelen = 8
# threshould = 0.18

def generateTree(sequence):
    i = 0
    root = Treenode(i, 0, 0, 0, 0, None)
    queue = []
    node = root
    lastnode = root

    for seq in sequence:
        tag, rou, arc, ele, pro = seq
        # node
        if tag >= 0.5 and tag <= 1.5:
            i += 1
            t_node = Treenode(i, node.x + rou*math.cos(arc), node.y + rou*math.sin(arc), node.ele + ele, node.pro + pro, node)
            lastnode = t_node
            node.children.append(t_node)
        # leftp
        elif tag < 0.5:
            queue.append(node)
            node = lastnode
        # rightp
        elif tag > 1.5:
            node = queue.pop()
    
    return root

def generate(decoder, predict_len=20, cuda=False):
    datamean, datastd = getMeanStd('../data/regionTreeSeqs/andes_peru.txt')
    ini_points = [[0,0,0,0,0], [1,0,0,0,0], [0,0,0,0,0]]
    hidden = decoder.init_hidden(1)
    prime_input = Variable(info_tensor(ini_points).unsqueeze(0))

    if cuda:
        # hidden = hidden.cuda():
        hidden = (hidden[0].cuda(), hidden[1].cuda())
        prime_input = prime_input.cuda()
    predicted = ini_points

    # Use priming string to "build up" hidden state
    for p in range(len(ini_points) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    vsequence = ["[", "Node", "["]
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        output_list = output.data.view(-1).tolist()
        tag, rou, arc, ele, pro = output_list
        for i in range(1, len(output_list)):
            output_list[i] = output_list[i]*datastd[i] + datamean[i]
        # node
        if tag >= 0.5 and tag <= 1.5:
            vsequence.append("Node")
        # leftp
        elif tag < 0.5:
            vsequence.append("[")
        # rightp
        elif tag > 1.5:
            vsequence.append("]")

        predicted.append(output_list)
        inp = output
        if cuda:
            inp = inp.cuda()
    print(predicted[8])
    return vsequence, predicted

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
    del args.filename
    if args.cuda:
        decoder.cuda()
    v, _ = generate(decoder, 50, cuda=args.cuda)
    print(v)
#     for i in range(10):
#         result = generate(decoder, **vars(args))
#         drawResult('test'+str(i), result)

