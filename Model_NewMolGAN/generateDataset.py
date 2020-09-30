import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sampleDIvideTree import divSampler
import pickle

def generate(nums, saddleSize, filepath):
    testSampler = divSampler(filepath)
    
    allTrees = []
    for i in tqdm(range(nums)):
        L, A, X = testSampler.sampleTree(saddleSize)
        allTrees.append([L, A, X])
        with open('dataGAN/data31.pkl', 'wb') as f:
            pickle.dump(allTrees, f, pickle.HIGHEST_PROTOCOL)
    xdif, ydif = testSampler.getXYavgDif()

if __name__ == "__main__":
    generate(3000, 15, 'dems/andes_peru.txt')