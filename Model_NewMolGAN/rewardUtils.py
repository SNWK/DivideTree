from tarjan.tc import tc
from collections import defaultdict
import numpy as np 
import sys
sys.path.append("..")
from utils.divtree_gen import getMSTLenForReward
import math

def getConnectivityReward(A):
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
            return (1./len(difGroups))**(1/3)
        else:
            return 0.

    reward = []
    for i in range(len(A)):
        reward.append(calConnectivityRatio(A[i]))
    return np.array(reward)



def getTreeReward(A, X):
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
        return rr

    reward = []
    for i in range(len(A)):
        reward.append(calTreeReward(A[i], X[i]))
    return np.array(reward)



def getDistributionReward(A, X, distributions, isEva=False):
    def calProminence(edges, nodes):
        edges = edges.tolist()
        nodes = nodes.tolist()
        domInfo = []
        saddleInfo = {}
        eles = []
        proms = []
        isos = []
        isolist = []
        for i in range(len(nodes)):
            tp, lati, longi, ele = nodes[i]
            if tp == 0:
                # saddle
                saddleInfo[i] = ele
            else:
                # peaks
                domInfo.append([i, ele])
                isolist.append([lati, longi, ele])
                eles.append(ele)

        ridges = defaultdict(list)
        # ridges
        for i in range(len(edges)):
            for j in range(len(edges[0])):
                if i == j: continue
                if i in saddleInfo.keys() and j in saddleInfo.keys(): continue
                if i in saddleInfo.keys() and j not in saddleInfo.keys(): 
                    ridges[j].append(i)
                if i not in saddleInfo.keys() and j in saddleInfo.keys(): 
                    ridges[i].append(j)

        # calculate proms:
        #### if there are saddles connected to it, seem the lowest one as the key saddle
        #### if there is no saddle connected to it, pro = ele
        for p in domInfo:
            if p[0] not in ridges.keys():
                pro = p[1]
            else:
                saddleEle = [saddleInfo[s] for s in ridges[p[0]]]
                pro = max(p[1] - min(saddleEle), 0)
            proms.append(pro)
        
        # calculate isos:
        #### sort the peaks by ele
        #### calculate distance between it and its next one
        isolist = sorted(isolist, key=lambda x: x[2])
        for i in range(len(isolist)-1):
            isos.append(math.sqrt((isolist[i][0] - isolist[i+1][0])**2 + (isolist[i][1] - isolist[i+1][1])**2)/math.sqrt(2))
        return eles, proms, isos

    def histogramFromBins(values, bins, frequencies=False):
        h = np.histogram(np.clip(values, a_min=None, a_max=bins[-1]), bins=bins)[0]
        if frequencies:
            return h/np.sum(h)
        else:
            return h

    def kldistance(distribution, synthesisValues):
        hbins  = distribution['bins']
        hmids  = distribution['x']
        hReal  = distribution['hist']
        hSynth = histogramFromBins(synthesisValues, hbins, frequencies=False)
        hNorm  = np.round(synthesisValues.size * hReal/hReal.sum())
        # smooth
        hNorm += 1
        hSynth += 1
        phNorm = hNorm / sum(hNorm)
        phSynth = hSynth / sum(hSynth)
        distance = 0
        for i in range(len(phNorm)):
            distance += phSynth[i]*(math.log(phSynth[i]) - math.log(phNorm[i]))
        return distance

    eleAll, promAll, isoAll = [], [], []
    if len(A.shape) == 2:
        ele, prom, iso = calProminence(A, X)
        eleAll += ele
        promAll += prom
        isoAll += iso
    else:
        for i in range(len(A)): 
            ele, prom, iso = calProminence(A[i], X[i])
            eleAll += ele
            promAll += prom
            isoAll += iso

    if isEva:
        return eleAll, promAll, isoAll
        
    ekldist = kldistance(distributions['elevation'], np.array(eleAll))
    pkldist = kldistance(distributions['prominence'], np.array(promAll))
    ikldist = kldistance(distributions['isolation'], np.array(isoAll))

    reward = np.ones((len(A)))*(ekldist + pkldist + ikldist)
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