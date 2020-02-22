import string
import random
import time
import math
import torch

onehotLength = 100
leftb = -0.01
rightb = 0.01
blockLength = (rightb-leftb) / math.sqrt(onehotLength)

def change2onehot(changelist):
    l = []
    for change in changelist:
        v, w = change
        if v < leftb: v = leftb
        if w < leftb: w = leftb
        if v > rightb: v = rightb
        if v > rightb: v = rightb
        vn = min(int( (v - leftb) / blockLength ), math.sqrt(onehotLength)-1)
        wn = min(int( (w - leftb) / blockLength ), math.sqrt(onehotLength)-1)
        l.append([int(math.sqrt(onehotLength)*vn) + wn])
    tensor = torch.LongTensor(l)
    return tensor

def index2change(idx):
    # idx in onehot
    amfi = 1
    vn = int(idx / math.sqrt(onehotLength))
    wn = idx%math.sqrt(onehotLength)
    v = random.uniform( leftb + vn*blockLength*amfi, leftb + (vn+1)*blockLength*amfi)
    w = random.uniform( leftb + wn*blockLength*amfi, leftb + (wn+1)*blockLength*amfi)
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

