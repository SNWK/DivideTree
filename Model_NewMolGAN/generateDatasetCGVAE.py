from ast import dump
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sampleDIvideTree import divSampler

import json
def generate(nums, saddleSize, filepath):
    testSampler = divSampler(filepath)
    
    allData = []
    for i in tqdm(range(nums)):
        L, A, X = testSampler.sampleTree(saddleSize)
        d = dict()
        # QED: should be the score of the graph (?)
        d["targets"] = [[1]]
        d["smiles"] = "None"

        d["node_features"] = []
        for node in X:
            if node[0] == 0:
                # saddles
                d["node_features"].append([1, 0] + node[1:])
            else:
                # peaks
                d["node_features"].append([0, 1] + node[1:])

        d["graph"] = []
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                if A[i][j] == 1:
                   d["graph"].append([i, 0, j]) 

        allData.append(d)

    with open('data31_valid.json', 'w') as f:
        json.dump(allData, f)

if __name__ == "__main__":
    generate(1000, 15, 'dems/andes_peru.txt')