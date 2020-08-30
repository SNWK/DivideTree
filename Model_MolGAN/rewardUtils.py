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
    if len(difGroups) != 0:
        return (1./len(difGroups))**(1/2)
    else:
        return 0.

def calConnectivityReward(A):
    reward = []
    for i in range(len(A)):
        reward.append(calConnectivityRatio(A[i]))
    return np.array(reward)


def calTreeReward(A, X):
    rr = 1.

    # the edge number should be vNum - 1
    traceSum = np.trace(A)
    edgeNum = (np.sum(A)-traceSum)/ 2
    treeEdgeNum = len(X) - 1
    # if edgeNum == treeEdgeNum:
    #     rr = 1.
    # else:
    #     rr = 0.
    if edgeNum != 0:
        rr *= 1 - ((edgeNum - treeEdgeNum)/edgeNum)**2
    else: 
        rr = 0.
    # # all values at A's diagonal line should be zero
    # rr *= 1 - abs(traceSum/len(X))

    # # each node should have at least one edge 
    # A_fill = A.copy()
    # np.fill_diagonal(A_fill, 0)
    # rowSum = np.sum(A_fill, 1)
    # noedgeNum = np.sum(rowSum == 0)
    # rr *= 1 - abs(noedgeNum/len(X))
    return rr

def getTreeReward(A, X):
    reward = []
    for i in range(len(A)):
        reward.append(calTreeReward(A[i], X[i]))
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