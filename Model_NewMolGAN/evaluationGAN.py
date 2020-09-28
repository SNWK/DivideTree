import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from PIL import Image
import os
from tqdm import tqdm


import math
from math import radians, cos, sin, asin, sqrt

from main import testCaseGenerator
from solver import Solver
from sampleDIvideTree import *
from rewardUtils import *

from torch.autograd import Variable
import torch

np.random.seed(42)

barColor  = (216/255, 226/255, 238/255, 1.0)
edgeColor = (137/255, 151/255, 168/255, 1.0)

dataSampler = divSampler('dems/andes_peru.txt')
distributions = dataSampler.getDistribution()

def printHistogramsDistances(hbins, hReal, hSynth):
    hdiff = np.abs(hReal - hSynth)
    print('Max', np.max(hdiff), 'Sum', np.sum(hdiff), 'Avg', np.mean(hdiff))
    print('EMD', np.diff(hbins)[0]*np.abs(np.cumsum(hReal) - np.cumsum(hSynth)).sum())
    
def histogramsComparison(distribution, synthesisValues, pos, fig):
    hbins  = distribution['bins']
    hmids  = distribution['x']
    hReal  = distribution['hist']
    hSynth = dataSampler.histogramFromBins(synthesisValues, hbins, frequencies=False)
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

def generateSample(size, draw=True, itr=0):
    
    z = solver.sample_z(1)
    z = Variable(torch.from_numpy(z)).to(solver.device).float()
    # Z-to-target
    edges_logits, nodes_logits = solver.G(z)
    # Postprocess with Gumbel softmax
    (edges_hat) = solver.postprocess((edges_logits), solver.post_method)
    edges_hat = (edges_hat + edges_hat.permute(1,0,2))/2
    A = torch.max(edges_hat, -1)[1]

    t1, t2 = torch.split(nodes_logits, [2,3], dim=2)
    (t1) = solver.postprocess((t1), solver.post_method)
    t1 = torch.max(t1, -1)[1]
    t1 = torch.reshape(t1,(1, t1.shape[0], 1))
    nodes_hat = torch.cat((t1,t2), 2)

    edges = A.data.cpu().numpy()
    nodes = nodes_hat.data.cpu().numpy()[0]
    x_r = nodes_hat.data.cpu().numpy()
    rewardR = solver.evaReward(edges[np.newaxis, :], x_r)[0]

    if draw: solver.drawTree(edges, nodes, itr)

    return edges, nodes, rewardR

'''
============================================================
initial the molGAN Solver
''' 
solver = testCaseGenerator(10000)



# calEdgeNum()
def compareIteration():
    maxIteration = 0
    maxReward = -100
    times = 100
    for i in range(1, 21):
        solver.restore_model(i*10000)
        totalReward = 0
        for j in range(times):
            if j == 0:
                _, _, reward = generateSample(31, draw=True, itr=i)
            else:
                _, _, reward = generateSample(31, draw=False)
            totalReward += reward
        aveReward = totalReward/times
        if aveReward >= maxReward:
            maxReward = aveReward
            maxIteration = i
        print("Iteration: ", i*10000, "   Average Reward: ", aveReward)
    print("Max Iteration: ", maxIteration*10000, "   max Reward: ", maxReward)
    return maxIteration


def drawDistributions(iter):
    solver.restore_model(iter)
    times = 50
    eleAll = []
    promAll = []
    isoAll = []
    for j in tqdm(range(times)):
        edges, nodes, _ = generateSample(31, draw=False)
        eles, proms, isos = getDistributionReward(edges[np.newaxis, :], nodes[np.newaxis, :], distributions, isEva=True)
        eleAll += eles
        promAll += proms
        isoAll += isos

    fig = plt.figure(figsize=(16, 15))
    histogramsComparison(distributions['elevation'], np.array(eleAll), 0, fig)
    plt.title('Elevation Distribution') 
    histogramsComparison(distributions['prominence'], np.array(promAll), 1, fig)
    plt.title('Prominence Distribution') 
    histogramsComparison(distributions['isolation'], np.array(isoAll), 2, fig)
    plt.title('Isolation Distribution') 
    plt.savefig('distribution' + str(iter) + '.png')

def run():
    maxIteration = compareIteration()
    print("draw distribution...")
    drawDistributions(maxIteration*10000)
    

run()

# calEdgeNum(30000, True)
# drawDistributions(110000)
# rebuild(190000)

# [-9.0874, -77.5737]
# d = haversine(-9.0874, -77.5737, -9.3874, -77.5737)
# longitude radius 0.3, latitude radius 0.07
# (x - 0.5)*0.6 - 9.0874
# (y - 0.5)*0.14 - 77.5737solver = testCaseGenerator(iter)