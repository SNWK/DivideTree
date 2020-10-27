import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sampleDIvideTree import divSampler
import pickle

# for model GRAN
def generate(nums, saddleSize, filepath):
    testSampler = divSampler(filepath)
    DS_A = [] # m x (node_id, node_id)
    DS_graph_indicator = [] # n x graph_id of the node with node_id i
    DS_graph_labels = [0 for _ in range(nums)] # N x graph_id TODO: add more label(diff terrains)
    DS_node_labels = [] # n x node_id i
    DS_node_attributes = [] # n x features
    nid = 0
    for gid in tqdm(range(nums)):
        L, A, X = testSampler.sampleTree(saddleSize)
        for i in range(L):
            DS_node_labels.append(X[i][0])
            DS_node_attributes.append(X[i][1:])
            DS_graph_indicator.append(gid)

        for i in range(L-1):
            for j in range(i, L):
                if A[i][j] == 1:
                    DS_A.append([nid + i, nid + j])
        nid += L
    path = '../Model_GRAN/data/DT/'
    with open(path + 'DS_A.txt', 'w+') as f:
        for edge in DS_A:
            f.write("{}, {}\n".format(edge[0], edge[1]))

    with open(path + 'DS_graph_indicator.txt', 'w+') as f:
        for i in DS_graph_indicator:
            f.write("{}\n".format(i))
    
    with open(path + 'DS_graph_labels.txt', 'w+') as f:
        for i in DS_graph_labels:
            f.write("{}\n".format(i))

    with open(path + 'DS_node_labels.txt', 'w+') as f:
        for i in DS_node_labels:
            f.write("{}\n".format(i))

    with open(path + 'DS_node_attributes.txt', 'w+') as f:
        for features in DS_node_attributes:
            f.write("{}, {}, {}\n".format(features[0], features[1], features[2]))
    
if __name__ == "__main__":
    # 1000 x 61
    generate(1000, 30, 'dems/andes_peru.txt')