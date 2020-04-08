import pickle

with open('gridsearch_new_bfs.dict', 'rb') as f:
    gridsearch = pickle.load(f)

print('n_state, n_mix, kl_dist, edit_dist')
gridsearch_1 = []
gridsearch_2 = []
gridsearch_3 = []
for k in gridsearch.keys():
    gridsearch_1.append([k[0], k[1], (gridsearch[k][0]/20 + gridsearch[k][2])/gridsearch[k][1]] )
    gridsearch_2.append([k[0], k[1], (gridsearch[k][0])/gridsearch[k][1]] )
    gridsearch_3.append([k[0], k[1], (gridsearch[k][2])/gridsearch[k][1]] )

for n in sorted(gridsearch_1, key=lambda y: y[2]):
    print(n)

print("-----")
for n in sorted(gridsearch_2, key=lambda y: y[2]):
    print(n)
print("-----")
for n in sorted(gridsearch_3, key=lambda y: y[2]):
    print(n)