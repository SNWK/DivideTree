import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models import *
import cPickle as pickle
import os, random

import sys
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from utils.shapefiles import sampleShapefileLocations
from analysis.peaksdata import filterPeaksHaversineDist
from utils.divtree_gen import *
from utils.seqdata_gen import *
from utils.seq2demodst import *
# process each region (note: it takes a long time!)
regionShapesDir = '../data/regionShapes'
regionPeaksDir = '../data/regionPeaks'
regionSeqsDir = '../data/regionSeqs'
regionTreeSeqsDir = '../data/regionTreeSeqs'

regionShapes = ['andes_peru.shp']

region = regionShapes[0]
allpeaksdf = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))



def tree_extract(peaks):
    # region peaks DB
    xmin = peaks['longitude'].min()
    xmax = peaks['longitude'].max()
    ymin = peaks['latitude'].min()
    ymax = peaks['latitude'].max()
    peaks['longitude'] = (peaks['longitude'] - xmin) / (xmax - xmin)
    peaks['latitude'] = (peaks['latitude'] - ymin) / (ymax - ymin)
    
    rootidx, edges = mstGnn(peaks)
    return rootidx, edges, peaks

def get_feature(p, peaks):
    return [peaks['longitude'].loc[p], peaks['latitude'].loc[p], 
        peaks['elevation in feet'].loc[p], peaks['prominence in feet'].loc[p]]

