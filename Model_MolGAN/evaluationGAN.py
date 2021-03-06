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
from math import radians, cos, sin, asin, sqrt

from rebuildSaddles import rebuildDivideTree
from main import testCaseGenerator
from solver import Solver
from torch.autograd import Variable
import torch

np.random.seed(42)

barColor  = (216/255, 226/255, 238/255, 1.0)
edgeColor = (137/255, 151/255, 168/255, 1.0)

def printHistogramsDistances(hbins, hReal, hSynth):
    hdiff = np.abs(hReal - hSynth)
    print('Max', np.max(hdiff), 'Sum', np.sum(hdiff), 'Avg', np.mean(hdiff))
    print('EMD', np.diff(hbins)[0]*np.abs(np.cumsum(hReal) - np.cumsum(hSynth)).sum())
    
def histogramsComparison(distribution, synthesisValues, pos, fig):
    hbins  = distribution['bins']
    hmids  = distribution['x']
    hReal  = distribution['hist']
    hSynth = histogramFromBins(synthesisValues, hbins, frequencies=False)
    hNorm  = np.round(synthesisValues.size * hReal/hReal.sum())

    # fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(4,2,1 + pos*2)
    _ = ax.bar (hmids, hSynth, width=np.diff(hbins), color=barColor, edgecolor=edgeColor)
    _ = ax.plot(hmids, hNorm, color='r')
    plt.title('Blue: generated data distribution') 

    ax = fig.add_subplot(4,2,2 + pos*2)
    _ = ax.bar (hmids, hNorm, width=np.diff(hbins), color='g')
    _ = ax.plot(hmids, hNorm, color='r')
    
    printHistogramsDistances(hbins, hReal/hReal.sum(), hSynth/hSynth.sum())
    print('Per bin differences (synthesis - target)')
    print(hSynth - hNorm)


def calDatasetInfo():
    promEpsilon   = 30   # m,  minimum prominence threshold in the analysis
    diskRadius    = 8   # km, used for the analysis to normalize histograms 
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

 
def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

def generateSample(size, draw=True):
    z = solver.sample_z(1)
    z = Variable(torch.from_numpy(z)).to(solver.device).float()
    # Z-to-target
    edges_logits, nodes_logits = solver.G(z)
    (edges_hat) = solver.postprocess((edges_logits), solver.post_method)
    edges_hat = (edges_hat + edges_hat.permute(1,0,2))/2
    A = torch.max(edges_hat, -1)[1]
    rewardR = solver.reward(A, nodes_logits)[0][0]
    # print(nodes_logits.data.cpu().numpy())
    edges = A.data.cpu().numpy()
    edgeNums = (np.sum(edges)-np.trace(edges))/2.
    nodes = nodes_logits.data.cpu().numpy()[0]
    pointList = []
    xmin, xmax, ymin, ymax, emin, emax, pmin, pmax, dmin, dmax, imin, imax = -78.3892, -71.3783, -16.16, -7.865, 173.73600000000002,  6735.7752,  30.7848,  2763.012,  0.005143981037873821,  0.7036450079239303,  0.050030301715518456,  2207.6431
    
    # TODO: X, Y
    # [-9.0874, -77.5737]
    # disk radius = 8km
    
    for i in range(size):
        # x = nodes[i][0] * (xmax - xmin) + xmin
        # y = nodes[i][1] * (ymax - ymin) + ymin
        x = (nodes[i][0] - 0.5)*0.6 - 9.0874
        y = (nodes[i][1] - 0.5)*0.14 - 77.5737
        e = nodes[i][2] * (emax - emin) + emin
        p = nodes[i][3] * (pmax - pmin) + pmin
        d = nodes[i][4] * (dmin - dmax) + dmin
        i = nodes[i][5] * (imin - imax) + imin
        node = [x, y, e, p, d, i]
        pointList.append(node)

    if draw: drawResult(pointList, edges, size)
    return pointList, edgeNums, rewardR

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
        plt.plot(X[i], Y[i], color='g')

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
    # plt.show()
    plt.clf()

def drawDivideTree(peaks, saddles, saddlePeaks):
    # draw ridges
    for i in range(len(saddlePeaks)):
        s = saddles[i]
        p0 = peaks[saddlePeaks[i][0]]
        p1 = peaks[saddlePeaks[i][1]]
        plt.plot([p0[0], s[0]], [p0[1], s[1]], color='r')
        plt.plot([p1[0], s[0]], [p1[1], s[1]], color='r')
    
    # draw peaks
    for i in range(len(peaks)):
        plt.scatter(peaks[i][0], peaks[i][1], c='g', marker='v')
    
    # draw saddles
    for i in range(len(saddles)):
        plt.scatter(saddles[i][0], saddles[i][1], c='b')

    plt.title('rebuilded DivideTree')
    plt.savefig('rebuildedDivideTree.png')
    plt.show()
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
solver = testCaseGenerator(10000)

def calEdgeNum(iter, isDraw):
    totalNums = 0
    totalRewards = 0
    times = 100
    solver.restore_model(iter)
    for i in tqdm(range(times)):
        _, edgeNums, r = generateSample(20, draw=isDraw)
        totalNums += edgeNums
        totalRewards += r
    print("Avg edgeNums: ", totalNums / times)
    print("Avg rewards: ", totalRewards / times)

def calDistance(iter):
    solver.restore_model(iter)
    times = 0
    # generateSample(40)
    evalData = []
    for sample in tqdm(range(50)):
        # pointlistFullSize, _, _ = generateSample(20, draw=False)
        pointlistFullSize = rebuild(iter, draw=False, needIter=False)
        maxTime = 300
        time = 0
        # choose one realtree as tree A
        for di,diskCenter in enumerate(sampleLocations):
            if time > maxTime:
                break
            # tree A
            peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
            if len(peaks) not in range(10, 20):
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
            elepre = [apointlist[i][2] for i in range(predictLen)]
            propre = [apointlist[i][3] for i in range(predictLen)]
            dompre = [apointlist[i][4] for i in range(predictLen)]
            isopre = [apointlist[i][5] for i in range(predictLen)]
            ekldist = kldistance(distributions['elevation'], np.array(elepre))
            pkldist = kldistance(distributions['prominence'], np.array(propre))
            dkldist = kldistance(distributions['dominance'], np.array(dompre))
            ikldist = kldistance(distributions['isolation'], np.array(isopre))

            # print(predictLen, dist, ekldist, pkldist)
            evalData.append([predictLen, dist, ekldist, pkldist, dkldist, ikldist])

    evalDataNp = np.array(evalData)
    print("predictLen", "treeDist", "ekldist", "pkldist", "dkldist", "ikldist")
    print("avg: ", evalDataNp.mean(0))
    print("min: ", evalDataNp.min(0))

# calEdgeNum()
def compareIteration():
    maxIteration = 0
    maxReward = -100
    times = 100
    for i in range(1, 21):
        solver.restore_model(i*10000)
        totalReward = 0
        for j in range(times):
            _, _, reward = generateSample(15, draw=False)
            totalReward += reward
        aveReward = totalReward/times
        if aveReward >= maxReward:
            maxReward = aveReward
            maxIteration = i
        print("Iteration: ", i*10000, "   Average Reward: ", aveReward)
        # time.sleep(1)
    # print(totalNums / times)
    print("Max Iteration: ", maxIteration*10000, "   max Reward: ", maxReward)
    return maxIteration


def drawDistributions(iter):
    solver.restore_model(iter)
    generatePointlist = []
    times = 50
    for j in range(times):
        pointList, _, _ = generateSample(20, draw=False)
        generatePointlist += pointList

    fig = plt.figure(figsize=(16, 20))
    histogramsComparison(distributions['elevation'], np.array(generatePointlist)[:,2], 0, fig)
    plt.title('Elevation Distribution') 
    histogramsComparison(distributions['prominence'], np.array(generatePointlist)[:,3], 1, fig)
    plt.title('Prominence Distribution') 
    histogramsComparison(distributions['dominance'], np.array(generatePointlist)[:,4], 2, fig)
    plt.title('Dominence Distribution') 
    histogramsComparison(distributions['isolation'], np.array(generatePointlist)[:,5], 3, fig)
    plt.title('Isolation Distribution') 
    plt.savefig('distribution' + str(iter) + '.png')

def drawRebuildDistributions(iter):
    solver.restore_model(iter)
    generatePointlist = []
    times = 50
    for j in range(times):
        pointList = rebuild(iter, draw=False, needIter=False)
        generatePointlist += pointList

    fig = plt.figure(figsize=(16, 20))
    histogramsComparison(distributions['elevation'], np.array(generatePointlist)[:,2], 0, fig)
    plt.title('Elevation Distribution') 
    histogramsComparison(distributions['prominence'], np.array(generatePointlist)[:,3], 1, fig)
    plt.title('Prominence Distribution') 
    histogramsComparison(distributions['dominance'], np.array(generatePointlist)[:,4], 2, fig)
    plt.title('Dominence Distribution') 
    histogramsComparison(distributions['isolation'], np.array(generatePointlist)[:,5], 3, fig)
    plt.title('Isolation Distribution') 
    plt.savefig('rebuildDistribution' + str(iter) + '.png')

def rebuild(iter, draw=False, needIter=True):
    if needIter:
        solver.restore_model(iter)
    peaks, _, _ = generateSample(20, draw=False)
    peakProms = np.array(peaks)[:,3]
    edgesMST = getMstRidges(peaks)
    # e = getTreeHMC(peaks)
    # drawMST(e)
    # plt.show()
    saddles, saddlePeaks, saddleElevs, ridgeTree, peakElevs, peakCoords = rebuildDivideTree(peaks, edgesMST) 
    if draw:
        drawDivideTree(peaks, saddles, saddlePeaks)
    promSaddle, promParent, promValues, KeySaddleMatrix = computeProminences(ridgeTree, peakElevs, saddleElevs, saddlePeaks)
    peakDoms = peakProms / peakElevs   
    isolDists, isolCoords = computeIsolations(peakCoords, peakElevs)
    for i in range(len(peaks)):
        peaks[i][3] = peakProms[i]
        peaks[i][4] = peakDoms[i]
        peaks[i][5] = isolDists[i]
        
    return peaks
# compareIteration()
# calEdgeNum(140000, True)
# calDistance(140000)

def run():
    maxIteration = compareIteration()
    print("cal avg edgeNUM and rewards ing...")
    calEdgeNum(maxIteration*10000, True)
    print("draw rebuild distribution...")
    drawDistributions(maxIteration*10000)
    drawRebuildDistributions(maxIteration*10000)
    print("cal avg edgeNUM and rewards ing...")
    calDistance(maxIteration*10000)

run()

# calEdgeNum(30000, True)
# drawDistributions(110000)
# rebuild(190000)

# [-9.0874, -77.5737]
# d = haversine(-9.0874, -77.5737, -9.3874, -77.5737)
# longitude radius 0.3, latitude radius 0.07
# (x - 0.5)*0.6 - 9.0874
# (y - 0.5)*0.14 - 77.5737solver = testCaseGenerator(iter)