import random
import collections
import numpy as np

def rebuildSaddle(peak0, peak1, ele):
    # no disturbances on elevation now 
    elevation = ele

    long0 = peak0[0]
    lat0 = peak0[1]
    long1 = peak1[0]
    lat1 = peak1[1]
    coord = [long0, lat0]

    v = [long1 - long0, lat1 - lat0]
    vn = [lat1 - lat0, -long1 + long0]
    disturbances = random.random()
    # disturb from 0.4 to 0.6
    coord = [(0.4 + 0.2*disturbances)*v[i] + coord[i] for i in range(2)]
    # disturb from -0.1 to 0.1
    disturbancesN = random.random()
    coord = [(0.2*disturbancesN - 0.1)*vn[i] + coord[i] for i in range(2)]

    return [coord[0], coord[1], elevation]

def treeBFS(parent, idx, peaks, peaksAdj, saddles):
    saddleEle = peaks[idx][2] - peaks[idx][3]
    peak0 = peaks[parent]
    peak1 = peaks[idx]
    newSaddle = rebuildSaddle(peak0, peak1, saddleEle)
    saddles.append(newSaddle+[parent, idx])
    for child in peaksAdj[idx]:
        if child != parent:
            treeBFS(idx, child, peaks, peaksAdj, saddles)

def rebuildDivideTree(peaks, ridges):
    '''
    Inn:
        peaks: peaks[i] = [lat, long, ele, prom, dom, iso]
        ridges:[[i,j], ...]
    Out:
        peakElevs: peakElevs[i] = elev
        peakCoords: peakCoords[i] = [latitude, longitude] done
        ridgeTree: RidgeTree[i,j] == saddle id connecting peak i to peak j, -1 ortherwise
        saddlePeaks: saddlePeaks[i] = [peak0, peak1]
    '''
    peakElevs = list()
    peakCoords = list()
    ridgeTree = -1 * np.ones((len(peaks), len(peaks)))
    saddlePeaks = list()
    saddles = list()
    saddleElevs = list()

    peakAdj = collections.defaultdict(list)
    for i in range(len(ridges)):
        peakAdj[ridges[i][0]].append(ridges[i][1])
        peakAdj[ridges[i][1]].append(ridges[i][0])
    
    for i in range(len(peaks)):
        peakCoords.append([peaks[i][0], peaks[i][1]])
        peakElevs.append(peaks[i][2])

    # random choose one peak
    peakidx = random.randint(0, len(peaks))

    for child in peakAdj[peakidx]:
        treeBFS(peakidx, child, peaks, peakAdj, saddles)
    
    for i in range(len(saddles)):
        saddlePeaks.append(saddles[i][3:])
        saddleElevs.append(saddles[i][2])
        ridgeTree[saddles[i][3],saddles[i][4]] = i
        ridgeTree[saddles[i][4], saddles[i][3]] = i

    return np.array(saddles), np.array(saddlePeaks), np.array(saddleElevs), np.array(ridgeTree).astype(int), np.array(peakElevs), np.array(peakCoords)