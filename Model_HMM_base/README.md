# HMM

## Approach

Library: hmmlearn

API: https://hmmlearn.readthedocs.io/en/latest/api.html

```python
gmm = hmm.GMMHMM(4, 2)
gmm = gmm.fit(bfsTreeBig)
gmm.score(bfsTreeBig)
```
## Experiment

- Dataset Size:
  - extract data by 30 km-radius disk from the area contains 4314 peaks.
  - most of sample areas contain less than 100 peaks because that the real data area is sparse.
  ![data distribution](note.assets/hmm_dataset.png)

- Sequence order:
  - BFS
  - DFS

- GMMHMM parameters:
  - n_state
  - n_mix

### Tree similarity measure

Method: Tree edit distance, KL-divergence

Edit operation:  
- remove, insert: cost = 10
- replace: $cost=\Sigma_i^4(weight_i * (nodeA_i - nodeB_i)^2)$  
  weight=[10000, 10000, 0.0001, 0.0001], because that the magnitude of dx and dy is 0.01, magnitude of dele and dpro is 100. from statistic

KL-divergence:  
- $KL(p|q) = \Sigma (p_i*(log(p_i) - log(q_i)))$
- there are some 0 item in statistic data, smooth the distribution by $q_i = (q_i + 1)/(sum(q) + len(q))$ 


### Grid search for the best parameters

number_state is in (2,15)
number_mixture is in (1,15)

- For each parameter set, train the gmmHMM model.
- For each real data which contains less than 200 peaks, use the model to predict five same length trees, compute the edit distance and the kl-divergence with the real distribution, then average them.
- Finally, sort and get the result

### Grid search result

BFS:  (best 5)  
[n_stat, n_mix, avg_distance]  
[2, 3, 1.486046541447855]  
[3, 2, 1.4906619066012352]  
[2, 1, 1.5101223851934553]  
[3, 5, 1.5114664796077668]  
[3, 1, 1.520689998373974]  

DFS:  (best 5)  
[n_stat, n_mix, avg_distance]  
[2, 5, 1.493269329102852]  
[4, 1, 1.4987174252788127]  
[2, 6, 1.5011613078149078]  
[3, 7, 1.5221518806947931]   
[2, 9, 1.5310414057449682]  

### Prediction

BFS:
- prediction
![hmm_23_bf](note.assets/hmm_23_bfs.png)  
- elevation difference  
  kl-divergence: 0.37995966468916076
![hmm_23_ele](note.assets/hmm_23_ele.png)  
- prominence difference   
  kl-divergence:  0.5359863864553517
![hmm_23_pro](note.assets/hmm_23_pro.png)

DFS:
- prediction
![hmm_25_dfs](note.assets/hmm_25_dfs.png)  
- elevation difference   
  kl-divergence:  0.5518457839992101
![hmm_25_ele](note.assets/hmm_25_ele.png)  
- prominence difference 
  kl-divergence:  0.6420304930445037
![hmm_25_pro](note.assets/hmm_25_pro.png)
