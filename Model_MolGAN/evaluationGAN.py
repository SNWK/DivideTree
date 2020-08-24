import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from PIL import Image
import os
import shutil
import subprocess
from tqdm import tqdm

from utils.shapefiles import sampleShapefileLocations
from utils.divtree_gen import *
from utils.seqdata_gen import *
from utils.seq2demodst import *
from utils.coords import *
from utils.noise import *
from utils.metrics import *
from analysis.peaksdata import *

from utils.treeEditDistance import *
import math

from main import testCaseGenerator
from solver import Solver
from torch.autograd import Variable
import torch

np.random.seed(42)

def calDatasetInfo():
    promEpsilon   = 30   # m,  minimum prominence threshold in the analysis
    diskRadius    = 90   # km, used for the analysis to normalize histograms 
    globalMaxElev = 9000 # m,  any value larger than any other peak elevation, used internally as initialization and undefineds
    terrainUnitKm  = 90  # km, size of terrain
    km2pixels = 1000/30  # 30 m/pixel

    # process each region (note: it takes a long time!)
    regionShapesDir = '../data/regionShapes'
    regionPeaksDir = 'data/regionPeaks'
    regionSeqsDir = 'data/regionSeqs'
    regionTreeSeqsDir = 'data/regionTreeSeqs'
    regionShapes = ['andes_peru.shp']
    regionName, filterCoords = 'andes_peru', [-9.0874, -77.5737] 
    peaksFile = '../data/regionPeaks/%s.csv' % regionName
    filterRadius = 90 # km
    filterHWidth = [km2deg(filterRadius), km2deg(filterRadius, filterCoords[0])]
    # read peaks file and filter region of interest
    df = pd.read_csv(peaksFile)
    filat = np.logical_and(df['latitude']  > filterCoords[0] - filterHWidth[0], 
                        df['latitude'] < filterCoords[0] + filterHWidth[0])
    filon = np.logical_and(df['longitude'] > filterCoords[1] - filterHWidth[1], 
                        df['longitude'] < filterCoords[1] + filterHWidth[1])
    df = df[np.logical_and(filat, filon)]

    print('Peaks:', df.shape[0])

    # compute distributions
    df = addExtraColumns(df)
    distributions = computeDistributions(df, diskRadius)

    diskRadius = 15
    sampleLocations = sampleShapefileLocations(os.path.join(regionShapesDir, 'andes_peru.shp'), diskRadius)
    return df, distributions, sampleLocations


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

def generateSample(size, draw=True):
    z = solver.sample_z(1)
    z = Variable(torch.from_numpy(z)).to(solver.device).float()
    # Z-to-target
    edges_logits, nodes_logits = solver.G(z)
    (edges_hat) = solver.postprocess((edges_logits), solver.post_method)
    A = torch.max(edges_hat, -1)[1]
    rewardR = solver.evaluationReward(A, nodes_logits)[0][0]
    # print(nodes_logits.data.cpu().numpy())
    edges = A.data.cpu().numpy()
    edgeNums = np.sum(edges)/2.
    nodes = nodes_logits.data.cpu().numpy()[0]
    pointList = []
    emin, emax, pmin, pmax = 570, 22099, 101, 9065
    for i in range(size):
        x = nodes[i][0]
        y = nodes[i][1]
        e = nodes[i][2] * (emax - emin) + emin
        p = nodes[i][3] * (pmax - pmin) + pmin
        node = [x, y, e, p]
        pointList.append(node)

    if draw: drawResult(pointList, edges, size)
    return pointList, rewardR

def drawOrigin(edges, nodes):
    X = []
    Y = []
    length = len(nodes)
    for i in range(length):
        for j in range(length):
            if edges[i][j] == 1:
                X.append([nodes[i][0], nodes[j][0]])
                Y.append([nodes[i][1], nodes[j][1]])

    for i in range(len(X)):
        plt.plot(X[i], Y[i], color='r') 

def drawMST(edges):
    X = []
    Y = []
    for v, w in edges:
        X.append([v[0], w[0]])
        Y.append([v[1], w[1]])
    for i in range(len(X)):
        plt.plot(X[i], Y[i], color='r')

def drawResult(pointList, edges, size):
    apointlist = np.array(pointList[:size])
    peaks = [list(p) for p in apointlist[:, :2]]
    edgesMST = getTreeHMC(peaks)
    plt.rcParams['figure.figsize'] = (12.0, 4.0)
    # in order
    plt.subplot(121)
    plt.scatter(apointlist[:,0], apointlist[:,1])
    drawOrigin(edges, apointlist)
    plt.scatter(apointlist[0,0], apointlist[0,1], c='y')
    plt.title('generated graph')
    # in MST
    plt.subplot(122)
    plt.scatter(apointlist[:,0], apointlist[:,1])
    drawMST(edgesMST)
    plt.scatter(apointlist[0,0], apointlist[0,1], c='y')
    plt.title('in MST')
    plt.savefig('molganSample' + str(size) + '.png')
    plt.savefig('res/molganSample' + str(size) + '.png')
    # plt.show()
    plt.clf()

'''
============================================================
initial the terrain area, calculate distributions 

'''
df, distributions, sampleLocations = calDatasetInfo()

'''
============================================================
initial the molGAN Solver
'''
solver = testCaseGenerator(110000)

def calEdgeNum(iter):
    totalNums = 0
    times = 100
    solver = testCaseGenerator(iter)
    for i in tqdm(range(times)):
        _, edgeNums = generateSample(15, draw=True)
        totalNums += edgeNums
        time.sleep(1)
    print(totalNums / times)

def calDistance():
    times = 0
    # generateSample(40)
    evalData = []
    for sample in tqdm(range(1)):
        pointlistFullSize, _ = generateSample(15)
        maxTime = 300
        time = 0
        # choose one realtree as tree A
        for di,diskCenter in enumerate(sampleLocations):
            if time > maxTime:
                break
            # tree A
            peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
            if len(peaks) not in range(5, 15):
                continue
            else:
                time += 1
            rootNode = genDivideTree(peaks)
            A = buildTree(rootNode)
            # tree B
            predictLen = len(peaks)
            pointlist = pointlistFullSize[:predictLen] 
            broot = genDivideTreePredict(pointlist)
            B = buildTree(broot)
            # tree edit distance
            dist = getDistance(A,B) / predictLen
            # kl distance
            apointlist = np.array(pointlist)
            elepre = [feet2m(apointlist[i][2]) for i in range(predictLen)]
            propre = [feet2m(apointlist[i][3]) for i in range(predictLen)]
            ekldist = kldistance(distributions['elevation'], np.array(elepre))
            pkldist = kldistance(distributions['prominence'], np.array(propre))
            # print(predictLen, dist, ekldist, pkldist)
            evalData.append([predictLen, dist, ekldist, pkldist])

        evalDataNp = np.array(evalData)
    print(evalDataNp.mean(0))
    print(evalDataNp.min(0))

# calEdgeNum()
def compareIteration():
    maxIteration = 0
    maxReward = 0
    times = 100
    for i in range(1, 21):
        solver = testCaseGenerator(i*10000)
        totalReward = 0
        for j in range(times):
            _, reward = generateSample(15, draw=False)
            totalReward += reward
        aveReward = totalReward/times
        if aveReward >= maxReward:
            maxReward = aveReward
            maxIteration = i
        print("Iteration: ", i*10000, "   Average Reward: ", aveReward)
        # time.sleep(1)
    # print(totalNums / times)
    print("Max Iteration: ", maxIteration*10000, "   max Reward: ", maxReward)

# compareIteration()
calEdgeNum(170000)