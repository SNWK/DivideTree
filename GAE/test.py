import pandas as pd
import os
import sys
sys.path.append('..')

from tqdm import tqdm

from utils.shapefiles import sampleShapefileLocations
from analysis.peaksdata import filterPeaksHaversineDist
from utils.divtree_gen import *
from utils.seqdata_gen import *
from utils.seq2demodst import *
regionShapesDir = 'data/regionShapes'
regionPeaksDir = 'data/regionPeaks'
regionSeqsDir = 'data/regionSeqs'
regionTreeSeqsDir = 'data/regionTreeSeqs'

regionShapes = ['andes_peru.shp']

region = regionShapes[0]
# allpeaksdf = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
diskRadius = 20


sampleLocations = sampleShapefileLocations('../data/regionShapes/andes_peru.shp', diskRadius)
df = pd.read_csv('../data/regionPeaks/andes_peru.csv')
for di,diskCenter in tqdm(enumerate(sampleLocations)):
    # filter peaks in disk using haversine distance
    peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
    print(peaks.index)
    break