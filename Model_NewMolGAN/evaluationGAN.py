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

def generateSample(size, draw=True):
    
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
    if draw: solver.drawTree(edges, nodes)

    return edges, nodes, 0

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

def run():
    edges, nodes, rewardR = generateSample(31)
    print(getDistributionReward(edges, nodes, distributions, isEva=True))
    # maxIteration = compareIteration()
    # print("cal avg edgeNUM and rewards ing...")
    # calEdgeNum(maxIteration*10000, True)
    # print("draw rebuild distribution...")
    # drawDistributions(maxIteration*10000)
    

run()

# calEdgeNum(30000, True)
# drawDistributions(110000)
# rebuild(190000)

# [-9.0874, -77.5737]
# d = haversine(-9.0874, -77.5737, -9.3874, -77.5737)
# longitude radius 0.3, latitude radius 0.07
# (x - 0.5)*0.6 - 9.0874
# (y - 0.5)*0.14 - 77.5737solver = testCaseGenerator(iter)