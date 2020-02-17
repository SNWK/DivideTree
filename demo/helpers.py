# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch

# Reading and un-unicode-encoding data


def read_file(filename):
    file_in = open(filename, 'r')
    allSeq = []
    for line in file_in.readlines:
        tmp = []
        for s in line.split():
            tmp.append(s.split(','))
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

