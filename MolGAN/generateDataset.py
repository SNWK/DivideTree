import os, sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append('..')
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
from utils.shapefiles import sampleShapefileLocations
from analysis.peaksdata import filterPeaksHaversineDist
from utils.divtree_gen import *
from utils.seqdata_gen import *
from utils.seq2demodst import *
from utils.coords import *
# process each region (note: it takes a long time!)
regionShapesDir = '../data/regionShapes'
regionPeaksDir = '../data/regionPeaks'
regionSeqsDir = '../data/regionSeqs'
regionTreeSeqsDir = '../data/regionTreeSeqs'

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

diskRadius = 15 # 626 in (20,100)

good = 0
bad = 0

for region in regionShapes:
    st = time.time()
    # sample stats locations inside polygon, separated at least 1/2 radius distance
    sampleLocations = sampleShapefileLocations(os.path.join(regionShapesDir, region), diskRadius)
    print(region, ": ", len(sampleLocations), "samples")
    # region peaks DB
    df = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
    xmin = df['elevation in feet'].min()
    xmax = df['elevation in feet'].max()
    ymin = df['prominence in feet'].min()
    ymax = df['prominence in feet'].max()
    df['elevation in feet'] = (df['elevation in feet'] - xmin) / (xmax - xmin)
    df['prominence in feet'] = (df['prominence in feet'] - ymin) / (ymax - ymin)
    # 570 22099 101 9065
    print(xmin, xmax, ymin, ymax)
    allTrees = []
    # compute sequences
    for di,diskCenter in tqdm(enumerate(sampleLocations)):
        # filter peaks in disk using haversine distance
        peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
        
        if peaks.shape[0] < 20 or peaks.shape[0] > 100:
            continue
        good += 1
        L, A, X = genDataMolGAN(peaks)
        allTrees.append([L, A, X])
    with open('dataGAN/data.pkl', 'wb') as f:
        pickle.dump(allTrees, f, pickle.HIGHEST_PROTOCOL)

print(good)
