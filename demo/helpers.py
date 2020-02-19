import string
import random
import time
import math
import torch

onehotLength = 10000
leftb = -0.01
rightb = 0.01
blockLength = (rightb-leftb) / math.sqrt(onehotLength)

def change2onehot(change):
    l = []
    for i in change:
        v, w = change
        if v < leftb: v = leftb
        if w < leftb: w = leftb
        if v > rightb: v = rightb
        if v > rightb: v = rightb
        vn = int( (v - leftb) / blockLength )
        wn = int( (w - leftb) / blockLength )
        t = [0 for z in range(onehotLength)]
        t[ math.sqrt(onehotLength)*vn + wn ] = 1
        l.append(t)
    tensor = torch.IntTensor(l)
    return tensor

def index2change(idx):
    # idx in onehot
    vn = int(idx / math.sqrt(onehotLength))
    wn = idx%math.sqrt(onehotLength)
    v = random.random( leftb + vn*blockLength, leftb + (vn+1)*blockLength)
    w = random.random( leftb + wn*blockLength, leftb + (wn+1)*blockLength)
    return [v, w]



def read_file(filename):
    file_in = open(filename, 'r')
    allSeq = []
    for line in file_in.readlines():
        tmp = []
        for s in line.split():
            tmp.append([float(c) for c in s.split(',')])
        allSeq.append(tmp)
    return allSeq, len(allSeq)

# Turning a location change into a tensor

def loc_tensor(seq):
    tensor = torch.FloatTensor(seq)
    return tensor

# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

