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
from analysis.peaksdata import filterPeaksHaversineDist, addExtraColumns
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

diskRadius = 8 # 626 in (20,100) 1210 in (10,20)

good = 0
bad = 0

sta = 0

for region in regionShapes:
    st = time.time()
    # sample stats locations inside polygon, separated at least 1/2 radius distance
    sampleLocations = sampleShapefileLocations(os.path.join(regionShapesDir, region), diskRadius)
    print(region, ": ", len(sampleLocations), "samples")
    # region peaks DB
    df = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
    # latitude  longitude  elevation in feet  key saddle latitude  key saddle longitude  prominence in feet  isolation latitude  isolation longitude  isolation in km

    df = addExtraColumns(df)
    elemin = df['elev'].min()
    elemax = df['elev'].max()
    promin = df['prom'].min()
    promax = df['prom'].max()
    dommin = df['dom'].min()
    dommax = df['dom'].max()
    isomin = df['isolation'].min()
    isomax = df['isolation'].max()

    # normalizaiton
    df['elev'] = (df['elev'] - elemin) / (elemax - elemin)
    df['prom'] = (df['prom'] - promin) / (promax - promin)
    df['dom'] = (df['dom'] - dommin) / (dommax - dommin)
    df['isolation'] = (df['isolation'] - isomin) / (isomax - isomin)
    # 173.73600000000002 6735.7752 30.7848 2763.012 0.005143981037873821 0.7036450079239303 0.050013523578808845 2207.6431
    print(elemin, elemax, promin, promax, dommin, dommax, isomin, isomax)
    allTrees = []
    # compute sequences
    for di,diskCenter in tqdm(enumerate(sampleLocations)):
        # filter peaks in disk using haversine distance
        peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
        
        if peaks.shape[0] < 10 or peaks.shape[0] > 20:
            continue
        good += 1
        sta += peaks.shape[0]
        L, A, X = genDataMolGAN(peaks)
        allTrees.append([L, A, X])
    with open('dataGAN/data20.pkl', 'wb') as f:
        pickle.dump(allTrees, f, pickle.HIGHEST_PROTOCOL)

print(good, sta/good)
