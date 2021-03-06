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
    def __init__(self, filePath):
        self.fpath = filePath
        self.load()
        # for per tree x,y normalization
        self.xdif = []
        self.ydif = []

    def load(self):
        # read divide tree
        self.data = list(readDivideTree(self.fpath, returnDEMCoords=False))

        # global normalization
        self.data[0][:,0] =(self.data[0][:,0]/2)
        self.data[0][:,1] =((self.data[0][:,1]+1)/2)
        self.data[5][:,0] =(self.data[5][:,0]/2)
        self.data[5][:,1] =((self.data[5][:,1]+1)/2)

        elemin = min(np.min(self.data[1]), np.min(self.data[6]))
        elemax = max(np.max(self.data[1]), np.max(self.data[6]))
        self.data[1] = self.normalization(self.data[1], elemin, elemax)
        self.data[6] = self.normalization(self.data[6], elemin, elemax)

        promin = np.min(self.data[2])
        promax = np.max(self.data[2])
        self.data[2] = self.normalization(self.data[2], promin, promax)

        dommin = np.min(self.data[3])
        dommax = np.max(self.data[3])
        self.data[3] = self.normalization(self.data[3], dommin, dommax)

        isomin = np.min(self.data[4])
        isomax = np.max(self.data[4])
        self.data[4] = self.normalization(self.data[4], isomin, isomax)
        
        print("Global Normallization")
        print(elemin, elemax, promin, promax, dommin, dommax, isomin, isomax)

        numPeaks = self.data[2].size
        print("Data loading, Number of peaks {}".format(numPeaks))

    def normalization(self, l, min, max):
        return (l - min) / (max - min)

    def getAllinfo(self):
        _, peakElevs, peakProms, peakDoms, peakIsos, _, _, _, _ = self.data
        return peakElevs, peakProms, peakDoms, peakIsos

    def histogramFromBins(self, values, bins, frequencies=False):
        h = np.histogram(np.clip(values, a_min=None, a_max=bins[-1]), bins=bins)[0]
        if frequencies:
            return h/np.sum(h)
        else:
            return h

    def getDistribution(self):
        elevation, prominence, dominance, isolation = self.getAllinfo()

        elevHistogramBins = np.append(np.arange(0, 1.0, 0.01), 1.0)
        promHistogramBins = np.append(np.arange(0, 1.0, 0.01), 1.0)
        domiHistogramBins = np.append(np.arange(0, 1.0, 0.01), 1.0)
        isolHistogramBins = np.append(np.arange(0, 1.0, 0.01), 1.0)

        h_elev       = self.histogramFromBins(elevation,   bins = elevHistogramBins)
        h_prom       = self.histogramFromBins(prominence,  bins = promHistogramBins)
        h_dominance  = self.histogramFromBins(dominance,   bins = domiHistogramBins)
        h_isolation  = self.histogramFromBins(isolation,   bins = isolHistogramBins)


        p_elev = h_elev/np.sum(h_elev)
        p_prom = h_prom/np.sum(h_prom)
        p_dominance = h_dominance/np.sum(h_dominance)
        p_isolation  = h_isolation/np.sum(h_isolation)
        
        x_elev = (elevHistogramBins[1:] + elevHistogramBins[:-1])/2
        x_prom = (promHistogramBins[1:] + promHistogramBins[:-1])/2
        x_isolation = (isolHistogramBins[1:] + isolHistogramBins[:-1])/2
        x_dominance  = (domiHistogramBins[1:] + domiHistogramBins[:-1])/2

        distributions = {
            'elevation' : {'hist': h_elev,       'pdf': p_elev,       'bins': elevHistogramBins, 'x': x_elev },
            'prominence': {'hist': h_prom,       'pdf': p_prom,       'bins': promHistogramBins, 'x': x_prom },
            'isolation' : {'hist': h_isolation,  'pdf': p_isolation,  'bins': isolHistogramBins, 'x': x_isolation },
            'dominance' : {'hist': h_dominance,  'pdf': p_dominance,  'bins': domiHistogramBins, 'x': x_dominance }
        }

        return distributions

    def getXYavgDif(self):
        avgx = sum(self.xdif) / len(self.xdif)
        avgy = sum(self.ydif) / len(self.ydif)
        print("x", avgx)
        print("y", avgy)

        return avgx, avgy

    def sampleTree(self, size):
        # 0         1          2          3         4         5             6            7            8
        peakCoords, peakElevs, peakProms, peakDoms, peakIsos, saddleCoords, saddleElevs, saddlePeaks, RidgeTree = self.data

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
        
        visitedSaddles = list(visitedSaddles)
        peaks = list(set(saddlePeaks[visitedSaddles].reshape(-1)))

        minx = min(np.min(saddleCoords[visitedSaddles, 0]), np.min(peakCoords[peaks, 0]))
        maxx = max(np.max(saddleCoords[visitedSaddles, 0]), np.max(peakCoords[peaks, 0]))
        self.xdif.append(maxx - minx) 
        miny = min(np.min(saddleCoords[visitedSaddles, 1]), np.min(peakCoords[peaks, 1]))
        maxy = max(np.max(saddleCoords[visitedSaddles, 1]), np.max(peakCoords[peaks, 1]))
        self.ydif.append(maxy - miny) 
        for s in visitedSaddles:
            X.append([0, self.normalization(saddleCoords[s, 0], minx, maxx), self.normalization(saddleCoords[s, 1], miny, maxy), saddleElevs[s]])
        peakIdxDict = dict()

        for p in peaks:
            X.append([1, self.normalization(peakCoords[p, 0], minx, maxx), self.normalization(peakCoords[p, 1], miny, maxy), peakElevs[p]])
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
    testSampler = divSampler('dems/andes_peru.txt')
    L, A, X = testSampler.sampleTree(5)
    print(L)
    print(np.array(A))
    print(np.array(X))