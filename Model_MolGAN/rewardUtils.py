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
            if i == j:
                G[i].append(j)
            elif A[i][j] == 1:
                G[i].append(j)
    groups = tc(G)
    groups:dict
    difGroups = set()
    for value in groups.values():
        difGroups.add(value)
    if len(difGroups) == 1:
        return 1.
    else:
        return 0.

def calConnectivityReward(A):
    reward = []
    for i in range(len(A)):
        reward.append(calConnectivityRatio(A[i]))
    return np.array(reward)

def evaConnectivityRatio(A):
    G = dict()
    for i in range(len(A)):
        G[i] = []
        for j in range(len(A[i])):
            if i == j:
                G[i].append(j)
            elif A[i][j] == 1:
                G[i].append(j)
    groups = tc(G)
    groups:dict
    difGroups = set()
    for value in groups.values():
        difGroups.add(value)
    if len(difGroups) != 0:
        return (1./len(difGroups))**(1/2)
    else:
        return 0.

def evaCalConnectivityReward(A):
    reward = []
    for i in range(len(A)):
        reward.append(evaConnectivityRatio(A[i]))
    return np.array(reward)


# A = [[
#     [1,0,0,0,0],
# 	[0,1,0,0,0],
# 	[0,0,1,0,0],
# 	[0,0,0,1,0],
# 	[0,0,0,0,1]],

#     [[0,1,0,0,0],
# 	[0,0,1,0,0],
# 	[1,0,0,0,0],
# 	[0,0,0,0,1],
# 	[0,0,0,1,0]]]

# X = [[[0,0], [0,1], [0,2], [1,3], [1,0.9]],
#     [[0,0], [0,1], [0,2], [1,3], [1,0.9]]]

# def testSubGraphNum(A):
#     G = dict()
#     for i in range(len(A)):
#         G[i] = []
#         for j in range(len(A[i])):
#             if A[i][j] == 1:
#                 G[i].append(j)
#     groups = tc(G)
#     groups:dict
#     difGroups = set()
#     for value in groups.values():
#         difGroups.add(value)
#     print(difGroups)
#     return len(difGroups)

# print(testSubGraphNum(A[0]))
# calRedundancyReward(A, X)