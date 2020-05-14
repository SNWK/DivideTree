import torch
import torch.nn as nn

import math, random, sys, os
import cPickle as pickle
o_path = os.getcwd()
sys.path.append(o_path)
from models.tree import *


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

# export PYTHONPATH=~/ex_code/icml18-jtnn
def tensorize(peaks, assm=True):
    tree = DTree(peaks)
    return tree

if __name__ == "__main__":
    # python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16
    diskRadius = 20
    all_data = []
    for region in regionShapes:
        st = time.time()
        # sample stats locations inside polygon, separated at least 1/2 radius distance
        sampleLocations = sampleShapefileLocations(os.path.join(regionShapesDir, region), diskRadius)
        print(region, ": ", len(sampleLocations), "samples")
        # region peaks DB
        df = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
        print(len(sampleLocations))
        allTrees = []
        # compute sequences
        for di,diskCenter in enumerate(sampleLocations):
            # filter peaks in disk using haversine distance
            peaks = filterPeaksHaversineDist(df, diskCenter, diskRadius)
            t_tree = tensorize(peaks)
            all_data.append(t_tree)
            print(di, len(peaks))
            if di > 100:
                break
    with open('tensors-1.pkl', 'wb') as f:
        pickle.dump(all_data, f, pickle.HIGHEST_PROTOCOL)

