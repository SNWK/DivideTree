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

from hmmlearn import hmm

np.random.seed(42)

promEpsilon   = 30   # m,  minimum prominence threshold in the analysis
diskRadius    = 30   # km, used for the analysis to normalize histograms 
globalMaxElev = 9000 # m,  any value larger than any other peak elevation, used internally as initialization and undefineds

terrainUnitKm  = 90  # km, size of terrain
km2pixels = 1000/30  # 30 m/pixel

# process each region (note: it takes a long time!)
regionShapesDir = '../data/regionShapes'
regionPeaksDir = 'data/regionPeaks'
regionSeqsDir = 'data/regionSeqs'
regionTreeSeqsDir = 'data/regionTreeSeqs'


regionShapes = ['andes_peru.shp']

#regionName, filterCoords = 'pyrenees', [42.5893, 0.9377] # pyrenees: aiguestortes
#regionName, filterCoords = 'alps', [45.8325,  7.0]  # mont blanc
#regionName, filterCoords = 'alps', [44.8742,  6.5]  # ecrins
#regionName, filterCoords = 'alps', [46.4702, 11.9492] # dolomites
#regionName, filterCoords = 'alps', [46.0159, 7.74318] # valais
#regionName, filterCoords = 'sahara', [30.38, 8.69] # sahara dunes
#regionName, filterCoords = 'andes_chile', [-21.4483, -68.0708] # chile
#regionName, filterCoords = 'karakoram', [35.8283, 76.3608] # karakoram
#regionName, filterCoords = 'colorado', [39.0782,-106.6986] # colorado
#regionName, filterCoords = 'yangshuo', [24.9917, 110.4617] # yangshuo
#regionName, filterCoords = 'himalaya', [28.7150, 84.2000] # himalaya: annapurna
#regionName, filterCoords = 'himalaya', [27.8575, 86.8267] # himalaya: everest
#regionName, filterCoords = 'norway', [62.1167, 6.8075] # norway
#regionName, filterCoords = 'alaska', [62.9500, -151.0908] # alaska
#regionName, filterCoords = 'patagonia', [-50.8925, -73.1533] # patagonia
#regionName, filterCoords = 'andes_aconcagua', [-32.6533, -70.0108] # aconcagua
regionName, filterCoords = 'andes_peru', [-9.0874, -77.5737] # huascaran
#regionName, filterCoords = 'rockies', [50.8003, -116.29517] # canadian rockies
#regionName, filterCoords = 'appalachians', [35.3855, -83.2380] # appalachians
#regionName, filterCoords = 'highlands', [56.9667, -3.5917] # highlands

peaksFile = '../data/regionPeaks/%s.csv' % regionName

filterRadius = 90 # km
filterHWidth = [km2deg(filterRadius), km2deg(filterRadius, filterCoords[0])]
print("filter scope: ",filterCoords[0] - filterHWidth[0], filterCoords[0] + filterHWidth[0],
      filterCoords[1] - filterHWidth[1], filterCoords[1] + filterHWidth[1])

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

import pickle
bfsseqdata_file_name = 'HMM_Seq_bfs.data'
dfsseqdata_file_name = 'HMM_Seq_dfs.data'
diskRadius = 30
if os.path.exists(bfsseqdata_file_name) and os.path.exists(dfsseqdata_file_name):
    with open(bfsseqdata_file_name, 'rb') as seqdata_file:
        bfsTrees = pickle.load(seqdata_file)
    with open(dfsseqdata_file_name, 'rb') as seqdata_file:
        dfsTrees = pickle.load(seqdata_file)
else:
    print(os.path.join(regionShapesDir, 'andes_peru.shp'))
    sampleLocations = sampleShapefileLocations(os.path.join(regionShapesDir, 'andes_peru.shp'), diskRadius)
    print(regionName, ": ", len(sampleLocations), "samples")
    bfsTrees = []
    dfsTrees = []
    # compute sequences
    for di,diskCenter in tqdm(enumerate(sampleLocations)):
        peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
        if peaks.shape[0] < 50:
            continue
        rootNode = genDivideTree(peaks)
        seqOfTree_bfs = genFullSeqHMM(rootNode, isDFS=False)
        seqOfTree_dfs = genFullSeqHMM(rootNode, isDFS=True)
        bfsTrees.append(seqOfTree_bfs)
        dfsTrees.append(seqOfTree_dfs)

    with open(bfsseqdata_file_name, 'wb') as seqdata_file:
        pickle.dump(bfsTrees, seqdata_file)
    with open(dfsseqdata_file_name, 'wb') as seqdata_file:
        pickle.dump(dfsTrees, seqdata_file)

# bfsseqdata_file_name = 'HMM_Seq_bfs_big.data'
# dfsseqdata_file_name = 'HMM_Seq_dfs_big.data'

# if os.path.exists(bfsseqdata_file_name) and os.path.exists(dfsseqdata_file_name):
#     with open(bfsseqdata_file_name, 'rb') as seqdata_file:
#         bfsTreeBig = pickle.load(seqdata_file)
#     with open(dfsseqdata_file_name, 'rb') as seqdata_file:
#         dfsTreeBig = pickle.load(seqdata_file)
# else:
#     rootNode = genDivideTree(df)
#     bfsTreeBig = genFullSeqHMM(rootNode, isDFS=False)
#     dfsTreeBig = genFullSeqHMM(rootNode, isDFS=True)

#     with open(bfsseqdata_file_name, 'wb') as seqdata_file:
#         pickle.dump(bfsTreeBig, seqdata_file)
#     with open(dfsseqdata_file_name, 'wb') as seqdata_file:
#         pickle.dump(dfsTreeBig, seqdata_file)

# little dataset
lengths = []
bfsTrees_flat = []
dfsTrees_flat = []
for i, seq in enumerate(bfsTrees):
    lengths.append(len(seq))
    bfsTrees_flat += bfsTrees[i]
    dfsTrees_flat += dfsTrees[i]



diskRadius = 30
sampleLocations = sampleShapefileLocations(os.path.join(regionShapesDir, 'andes_peru.shp'), diskRadius)
gridsearch = dict()
time_1 = time.time()
for n_state in range(2,10):
    for n_mix in range(1,10):
        try:
            gmm = hmm.GMMHMM(n_state, n_mix)
            # gmm = hmm.GaussianHMM(4)
            gmm = gmm.fit(bfsTrees_flat, lengths)
            # choose one realtree as tree A
            for di,diskCenter in tqdm(enumerate(sampleLocations)):
                # tree A
                peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
                if len(peaks) == 0:
                    continue
                if len(peaks) > 200:
                    continue
                rootNode = genDivideTree(peaks)
                A = buildTree(rootNode)

                highestIidx = peaks['elevation in feet'].idxmax()
                highest = [peaks['longitude'].loc[highestIidx], 
                        peaks['latitude'].loc[highestIidx], peaks['elevation in feet'].loc[highestIidx], 
                        peaks['prominence in feet'].loc[highestIidx]]
                # tree B
                predictLen = len(peaks)
                for btime in range(3):
                    pointlist = [highest]
                    deltaslist = [[0.0, 0.0, 0.0, 0.0]]
                    predict = gmm.sample(predictLen)[0]
                    for i in range(predictLen):
                        deltas = predict[i]
                        deltaslist.append(deltas)
                        point = [0,0,0,0]
                        for i in range(4):
                            point[i] = pointlist[-1][i] + deltas[i]
                        pointlist.append(point)
                    broot = genDivideTreePredict(pointlist)
                    B = buildTree(broot)
                    # distance
                    dist = getDistance(A,B)
                    print(n_state, n_mix, dist)
                    if (n_state, n_mix) not in gridsearch.keys():
                        gridsearch[(n_state, n_mix)] = [dist, 1]
                    else:
                        gridsearch[(n_state, n_mix)][0] += dist
                        gridsearch[(n_state, n_mix)][1] += 1

        except BaseException:
            continue

time_2 = time.time()
print(dist, time_2 - time_1)

with open('gridsearch_new.dict', 'wb') as f:
    pickle.dump(gridsearch, f)

print('n_state, n_mix, edit_dist')
gridsearch_list = []
for k in gridsearch.keys():
    gridsearch_list.append([k[0], k[1], gridsearch[k][0]/gridsearch[k][1]] )
for n in sorted(gridsearch_list, key=lambda y: y[2]):
    print(n)