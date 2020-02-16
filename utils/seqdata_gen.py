import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from utils.divtree_gen import *
import copy

def dfs(tree, seq, node):
    p = copy.deepcopy(seq)
    p.append(node)
    result = []
    if tree[node] == []:
        result.append(p)
    else:
        for childs in tree[node]:
            tree[childs[0]].remove((node, childs[1]))
            result += dfs(tree, p, childs[0])
    return result

def drawSeq(peaks, paths):
    peaks['latitude'] =  (peaks['latitude'] - peaks['latitude'].min()) / (peaks['latitude'].max() - peaks['latitude'].min())
    peaks['longitude'] =  (peaks['longitude'] - peaks['longitude'].min()) / (peaks['longitude'].max() - peaks['longitude'].min())
    
    for index, row in peaks.iterrows():
        plt.scatter(row['longitude'], row['latitude'], color='b')

    for p in paths:
        x = []
        y = []
        for i in range(len(p) - 1):
            v = p[i]
            w = p[i+1]
            lat1 = peaks['latitude'].loc[v]
            lon1 = peaks['longitude'].loc[v]
            lat2 = peaks['latitude'].loc[w]
            lon2 = peaks['longitude'].loc[w]    
            x.append([lon1, lon2])
            y.append([lat1, lat2])
        for i in range(len(x)):
            plt.plot(x[i], y[i], color=np.random.rand(3,))
    plt.ylabel('All Seqs')
    plt.show()

def drawTree(peaks, paths):
    peaks['latitude'] =  (peaks['latitude'] - peaks['latitude'].min()) / (peaks['latitude'].max() - peaks['latitude'].min())
    peaks['longitude'] =  (peaks['longitude'] - peaks['longitude'].min()) / (peaks['longitude'].max() - peaks['longitude'].min())
    
    for index, row in peaks.iterrows():
        plt.scatter(row['longitude'], row['latitude'], color='b')
    x = []
    y = []
    for p in paths:
        v = p[0]
        w = p[1]
        lat1 = peaks['latitude'].loc[v]
        lon1 = peaks['longitude'].loc[v]
        lat2 = peaks['latitude'].loc[w]
        lon2 = peaks['longitude'].loc[w]    
        x.append([lon1, lon2])
        y.append([lat1, lat2])
    for i in range(len(x)):
        plt.plot(x[i], y[i], color='r')
    plt.ylabel('All Seqs')
    plt.show()

def genSeq(peaks):
    vertices = peaks.index
    gsample = Graph()
    pairs = set()
    for v in vertices:
        for w in vertices:
            if v != w and (v, w) not in pairs and (w, v) not in pairs:
                pairs.add((v, w))
                lat1 = peaks['latitude'].loc[v]
                lon1 = peaks['longitude'].loc[v]
                lat2 = peaks['latitude'].loc[w]
                lon2 = peaks['longitude'].loc[w]
                dist = math.sqrt(pow(lat1 - lat2, 2) + pow(lon1 - lon2, 2))
                gsample.add_edge(v, w, dist)
    treesample = minimum_spanning_tree(gsample)

    # For test, draw the tree
    # drawTree(peaks, treesample)

    root = peaks['elevation in feet'].idxmax()
    tree = {}
    for edges in treesample:
        v, w, d = edges
        if v not in tree.keys():
            tree[v] = []
        if w not in tree.keys():
            tree[w] = []
        tree[v].append((w, d))
        tree[w].append((v, d))

    allSeq = dfs(tree, [], root)
    return allSeq

def test():
    tree = {}
    tree[1] = [(2,1), (3,1)]
    tree[2] = [(4,1), (5,1), (1,1)]
    tree[3] = [(6,1), (1,1)]
    tree[4] = [(2,1)]
    tree[5] = [(2,1)]
    tree[6] = [(3,1)]
    root = 1
    s = dfs(tree, [], root)
    print(s)
