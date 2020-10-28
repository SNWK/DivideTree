import numpy as np
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import torch
def probMask(A, P, X, ii, jj):
    def cross_product(a, b):
        return a[0]*b[1] - a[1]*b[0]
    def iscross(pairData):
        A, B = pairData[0]
        C, D = pairData[1]
        AC = C - A
        AD = D - A
        BC = C - B
        BD = D - B
        CA = - AC
        CB = - BC
        DA = - AD
        DB = - BD
        
        return 1 if cross_product(AC,AD)*cross_product(BC,BD) < 0 and cross_product(CA,CB)*cross_product(DA,DB) < 0 else 0

    def calprobMask(pairData):
        Adj, Pro, Xfe, ii, jj = pairData
        Adj = Adj[:ii, :ii]
        Pro = Pro[:jj - ii,:]
        Xfe = Xfe[:jj]

        # existed ridges
        edgesList = []
        edgeSet = set()
        edgesIdxList = []
        for i in range(ii):
            for j in range(i):
                if Adj[i][j] == 1 and (i,j) not in edgeSet:
                    edgesList.append([Xfe[i][:2], Xfe[j][:2]])
                    edgesIdxList.append((i,j))
                    edgeSet.add((i,j))
        
        newNodes = Xfe[ii:jj]
        newp = Pro
        for idx in range(len(newNodes)):
            for pidx in range(len(Pro[idx])):
                newedge = [Xfe[pidx][:2], Xfe[ii+idx][:2]]
                for i, edge in enumerate(edgesList):
                    if pidx in edgesIdxList[i]:
                        continue
                    if iscross([edge, newedge]):
                        newp[idx,pidx] = 0
                        break
        
        return newp

    newP = []
    pool = ThreadPool(4)
    tasks = [[A[i], P[i], X[i], ii, jj] for i in range(len(A)) ]
    newP = pool.map(calprobMask, tasks)
    pool.close()
    pool.join()    
    return newP