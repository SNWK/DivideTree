import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models import *
import cPickle as pickle
import sys, os, random
o_path = os.getcwd()
sys.path.append(o_path)
from models.encoder import Encoder
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


class DTreeFolder(object):

    def __init__(self, data_folder, batch_size, shuffle=True):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = DTreeDataset(batches)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader
        
class DTreeDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx])


def tensorize(tree_batch):
    set_batch_nodeID(tree_batch) # we don't want to change itx, itx related to peak information 
    jtenc_holder, mess_dict = Encoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    
    return tree_batch, jtenc_holder

def set_batch_nodeID(mol_batch):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes.keys():
            mol_tree.nodes[node].graphid = tot
            tot += 1
