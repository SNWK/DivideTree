import os
import os, sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append('..')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from utils.coords import *
from utils.divtree_reader import readDivideTree

class divSampler():
    def __init__(self, filesPath, selected_file):
        self.fpath = os.path.join(filesPath, selected_file)
        self.load()

    def load(self):
        # read divide tree
        self.data = readDivideTree(self.fpath, returnDEMCoords=False)
        numPeaks = self.data[2].size
        print("Data loading, Number of peaks {}".format(numPeaks))

    def getAllinfo(self):
        _, peakElevs, peakProms, peakDoms, peakIsos, _, _, _, _ = self.data
        return peakElevs, peakProms, peakDoms, peakIsos

    def sampleTree(self, size):
        peakCoords, peakElevs, peakProms, peakDoms, peakIsos, saddleCoords, saddleElevs, saddlePeaks, RidgeTree = self.data

        peakCoords[:,0] =(peakCoords[:,0]/2)
        peakCoords[:,1] =((peakCoords[:,1]+1)/2)
        saddleCoords[:,0] =(saddleCoords[:,0]/2)
        saddleCoords[:,1] =((saddleCoords[:,1]+1)/2)

        A = [[0 for i in range(size*2+1)] for j in range(size*2+1)]
        X = []

        peakQueue = []
        saddleQueue = []
        visitedSaddles = set()
        
        numSaddles = np.shape(saddlePeaks)[0]
        randomStartSaddle = np.random.randint(0,numSaddles) # We start by sampling a saddle node
        p1,p2 = saddlePeaks[randomStartSaddle] # We get the 2 associated peaks
        
        # Update the queues etc.
        visitedSaddles.add(randomStartSaddle)
        peakQueue.append(p1)
        peakQueue.append(p2)
        
        while len(peakQueue) > 0:
            p = peakQueue.pop(0) # Remove first element of the queue
            # Now access the saddles that have this peak linked to it
            s1 = set(np.where(saddlePeaks == p)[0])
            #print("Got this for saddles ",s1)
            s1 = s1.difference(visitedSaddles)
            saddleQueue.extend(list(s1))
            if len(saddleQueue) > 0:
                s = saddleQueue.pop(0)
                p1, p2 = saddlePeaks[s]
                # Update the queues etc.
                visitedSaddles.add(s)
                peakQueue.append(p1)
                peakQueue.append(p2)
            if len(visitedSaddles) > size-1:
                break
        
        for s in visitedSaddles:
            X.append([0, saddleCoords[s, 0], saddleCoords[s, 1], saddleElevs[s]])
        visitedSaddles = list(visitedSaddles)
        peaks = set(saddlePeaks[visitedSaddles].reshape(-1))
        peakIdxDict = dict()
        for p in peaks:
            X.append([1, peakCoords[p, 0], peakCoords[p, 1], peakElevs[p]])
            peakIdxDict[p] = len(X) - 1
        
        for i, ps in enumerate(list(saddlePeaks[visitedSaddles])):
            p1, p2 = ps
            p1 = peakIdxDict[p1]
            p2 = peakIdxDict[p2]
            A[i][p1] = 1
            A[i][p2] = 1
            A[p1][i] = 1
            A[p2][i] = 1
        
        return len(X), A, X


if __name__ == "__main__":
    testSampler = divSampler('dems', "andes_peru.txt")
    L, A, X = testSampler.sampleTree(5)
    print(L)
    print(np.array(A))
    print(np.array(X))