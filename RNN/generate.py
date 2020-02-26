#!/usr/bin/env python
import sys
sys.path.append("..") 

import torch
import os
import argparse
import copy
import numpy as np
from RNN.helpers import *
from RNN.model import *
import matplotlib.pyplot as plt
from utils.divtree_gen import *
import random

predict_len=20
temperature=0.8
maxprelen = 8
# threshould = 0.18

def generateTree(sequence):
    i = -1
    root = Treenode(i, 0, 0, 0, 0, None)
    queue = []
    node = root
    lastnode = root

    for seq in sequence:
        tag, rou, arc, ele, pro = seq
        # node
        if tag >= -0.5 and tag <= 0.5:
            i += 1
            t_node = Treenode(i, node.x + rou*math.cos(arc), node.y + rou*math.sin(arc), node.ele + ele, node.pro + pro, node)
            lastnode = t_node
            node.children.append(t_node)
        # leftp
        elif tag < -0.5:
            queue.append(node)
            node = lastnode
        # rightp
        elif tag > 0.5:
            node = queue.pop()
    
    return root

def generate(decoder, predict_len=20, cuda=False, filename='null'):
    if filename == 'null':
        filename = '../data/regionTreeSeqs/andes_perulittle.txt'
    datamean, datastd = getMeanStd(filename)

    inilen = 8
    ex_file = open(filename, 'r')
    ex_lines = ex_file.readlines()
    ex_i = random.randint(0, len(ex_lines)-1)
    ini_points = []
    vsequence = []
    for i, node in enumerate(ex_lines[ex_i].split(',')):
        if i > inilen - 1:
            break
        node_data = [float(s) for s in node.split(';')]
        ini_points.append(node_data)
        if node_data[0] == -1:
            vsequence.append('[')
        elif node_data[0] == 0:
            vsequence.append('Node')
        else:
            vsequence.append(']')

    # print('initial seq is:\n', ini_points)
    print('initial vseq is:', vsequence)

    predicted = ini_points

    
    for p in range(predict_len):
        hidden = decoder.init_hidden(1)
        prime_input = Variable(info_tensor(ini_points[-inilen:]).unsqueeze(0))

        if cuda:
            # hidden = hidden.cuda():
            hidden = (hidden[0].cuda(), hidden[1].cuda())
            prime_input = prime_input.cuda()

        # Use priming string to "build up" hidden state
        for p in range(inilen-1):
            _, hidden = decoder(prime_input[:,p], hidden)
        # predict   
        inp = prime_input[:,-1]
        output, hidden = decoder(inp, hidden)
        output_list = output.data.view(-1).tolist()
        output = output_list[2:]
        tag = output_list[:3].index(max(output_list[:3]))
        output[0] = tag
        
        ini_points.append(output)

        for i in range(3, len(output_list)):
            output[i-2] = output_list[i]*datastd[i-2] + datamean[i-2] #math.tan(output_list[i]*math.pi/2.0)
        # node
        if tag == 1:
            vsequence.append("Node")
        # leftp
        elif tag == 0:
            vsequence.append("[")
        # rightp
        elif tag == 2:
            vsequence.append("]")
        predicted.append(output)

    print(predicted[10])
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

