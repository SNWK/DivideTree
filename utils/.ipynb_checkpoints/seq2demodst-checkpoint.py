import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# process each region (note: it takes a long time!)
regionShapesDir = 'data/regionShapes'
regionPeaksDir = 'data/regionPeaks'
regionSeqsDir = 'data/regionSeqs'
demoDatasetDir = 'data/demoData'
diskRadius = 30

regionShapes = ['andes_peru.shp']

def pre1():
    for region in regionShapes:
        st = time.time()
        # region peaks DB
        peaks = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
        fdata_in = open(os.path.join(regionSeqsDir, region.replace('.shp', '.txt')), 'r')

        # preprocess -> 10 long seqs
        allSeqs = []
        for line in fdata_in.readlines():
            seq = line.split()
            if len(seq) >= 20:
                for i in range(len(seq) - 20):
                    tmp = seq[i : i+20]
                    if tmp not in allSeqs:
                        allSeqs.append(tmp)
        fout = open(os.path.join(demoDatasetDir, region.replace('.shp', 'pre20.txt')), 'w') 
        for s in allSeqs:
            fout.write(" ".join([str(v) for v in s]))
            fout.write('\n')
        fout.close()
        print("done pre")

def pre2():
    for region in regionShapes:
        st = time.time()
        # region peaks DB
        peaks = pd.read_csv(os.path.join(regionPeaksDir, region.replace('.shp', '.csv')))
        peaks['latitude'] =  (peaks['latitude'] - peaks['latitude'].min()) / (peaks['latitude'].max() - peaks['latitude'].min())
        peaks['longitude'] =  (peaks['longitude'] - peaks['longitude'].min()) / (peaks['longitude'].max() - peaks['longitude'].min())
        
        fdata_in = open(os.path.join(demoDatasetDir, region.replace('.shp', 'pre20.txt')), 'r')

        allSeqs = []
        for line in fdata_in.readlines():
            seq = line.split()
            tmp = [int(v) for v in seq]
            allSeqs.append(tmp)

        new_allSeqs = []
        for seq in allSeqs:
            nseq = []
            for i in range(len(seq) - 1):
                v = seq[i]
                w = seq[i+1]
                lat1 = peaks['latitude'].loc[v]
                lon1 = peaks['longitude'].loc[v]
                lat2 = peaks['latitude'].loc[w]
                lon2 = peaks['longitude'].loc[w]
                nseq.append((lon2-lon1, lat2-lat1))
            new_allSeqs.append(nseq)

        fout = open(os.path.join(demoDatasetDir, region.replace('.shp', '20.txt')), 'w') 
        for s in new_allSeqs:
            fout.write(" ".join([",".join([str(l) for l in v]) for v in s]))
            fout.write('\n')
        fout.close()

        print('%s: %3d samples, %d s'%(region, len(new_allSeqs), time.time() - st)) 
    print('done!')

pre1()
pre2()