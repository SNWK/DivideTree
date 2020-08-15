from tarjan.tc import tc
from collections import defaultdict
import numpy as np 
import sys
sys.path.append("..")
from utils.divtree_gen import getMSTLenForReward

def calConnectivityRatio(A):
    G = dict()
    for i in range(len(A)):
        G[i] = []
        for j in range(len(A[i])):
            if A[i][j] == 1:
                G[i].append(j)
    groups = tc(G)
    groups:dict
    difGroups = set()
    for value in groups.values():
        difGroups.add(value)
    if len(difGroups) != 0:
        return 1. / len(difGroups)
    else:
        return 1.

def calConnectivityReward(A):
    reward = []
    for i in range(len(A)):
        reward.append(calConnectivityRatio(A[i]))
    return np.array(reward)

def calRedundancyRatio(A, X):
    allLen = np.sum(A)
    mstLen = getMSTLenForReward(X)
    if allLen != 0:
        return 1 - abs((allLen - mstLen)/allLen)
    else:
        return 1

def calRedundancyReward(A, X):
    reward = []
    for i in range(len(A)):
        reward.append(calRedundancyRatio(A[i], X[i]))
    return np.array(reward)

A = [[[0,1,0,0,0],
	[0,0,1,0,0],
	[1,0,0,0,0],
	[0,0,0,0,1],
	[0,0,0,1,0]],
    [[0,1,0,0,0],
	[0,0,1,0,0],
	[1,0,0,0,0],
	[0,0,0,0,1],
	[0,0,0,1,0]]]

X = [[[0,0], [0,1], [0,2], [1,3], [1,0.9]],
    [[0,0], [0,1], [0,2], [1,3], [1,0.9]]]

calConnectivityReward(A)
calRedundancyReward(A, X)