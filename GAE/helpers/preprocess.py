import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import cPickle as pickle

from models import *

# export PYTHONPATH=~/ex_code/icml18-jtnn
def tensorize(featrue, assm=True):
    mol_tree = DTree(featrue)
    return mol_tree

if __name__ == "__main__":
    # python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16
    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path")
    parser.add_option("-n", "--split", dest="nsplits", default=10)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)
    # multiprocessing 
    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)
    # COC(=O)CNC(=O)c1cc(Br)c2c(c1)OCCO2
    # c1cc2ccc[c:8]3[s:8][nH:8][c:8](c1)[c:12]23
    with open(opts.train_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    all_data = pool.map(tensorize, data)

    le = (len(all_data) + num_splits - 1) / num_splits

    for split_id in xrange(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

