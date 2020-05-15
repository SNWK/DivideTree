## Automatically Terrain Divide Tree Generation 

Wang KAI, 2020/5/14  

### TODO:  

1. Model_MolGAN/solver.py reward machine:
   1. cycle ratio
   2. connectivity
   3. KL divergence
2. Model_GAE_Jtnn/ rewrite

### Files:  

analysis/: peaks data extraction code  

data/: dataset  

utils/: some functions for data preprocess  

Model_HMM_base/: hidden markov model, baseline  

Model_RNN_Classic/: a classic RNN model

Model_GAE_Jtnn/: a model combines the idea from Jtnn  

Model_MolGAN/: a GGAN model with reward machine  


### Problem

We know divide tree's power in terrain generation area from Oscar's work. One biggest shortcoming of his method is high time cost. (About 60 seconds for 1000 peaks generation.)

What I want to do is using sequence prediction method / GNN to solve the divide tree generation problem. 


### Real Data Extraction

Library: 
- https://github.com/edwardearl/winprom
- https://github.com/akirmse/mountains

The second one is an update vesion of the first one. 
The author is Andrew Kirmse.

- blog: http://www.andrewkirmse.com/prominence?pli=1#TOC-Anti-prominence
- paper: https://journals.sagepub.com/doi/10.1177/0309133317738163

#### DEM data

"SRTM":  

- Shuttle Radar Topography Mission -- NASA

- SRTM3(for world, 3 arc-seconds): http://www.webgis.com/srtm3.html, download here

"NED13-ZIP","NED1-ZIP":

- Higher resolution, America.
- download: https://viewer.nationalmap.gov/basic/



