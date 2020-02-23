# RNN

import string
import random
import time
import math
import torch
import numpy as np

def getMeanStd(filename):
    file_in = open(filename, 'r')
    allpoints = []
    for line in file_in.readlines():
        for s in line.split(','):
            allpoints.append([float(c) for c in s.split(';')])
    allpoints = np.array(allpoints)
    datamean = allpoints.mean(axis=0)
    datastd = allpoints.std(axis=0)
    file_in.close()
    return datamean, datastd

def read_file(filename):
    file_in = open(filename, 'r')
    allSeq = []
    datamean, datastd = getMeanStd(filename)
    for line in file_in.readlines():
        tmp = []
        for s in line.split(','):
            p = [float(c) for c in s.split(';')]
            # z score
            g = [(p[i]-datamean[i])/datastd[i] for i in range(len(p))]
            g[0] = p[0]
            tmp.append(g)
            # if len([float(c) for c in s.split(';')]) == 4:
            #     print([float(c) for c in s.split(';')])
        allSeq.append(tmp)
    file_in.close()
    return allSeq, len(allSeq)

# Turning a location change into a tensor

def info_tensor(seq):
    tensor = torch.FloatTensor(seq)
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

