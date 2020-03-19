import pickle

with open('gridsearch_new.dict', 'rb') as f:
    gridsearch = pickle.load(f)

print('n_state, n_mix, edit_dist')
gridsearch_list = []
for k in gridsearch.keys():
    gridsearch_list.append([k[0], k[1], gridsearch[k][0]/gridsearch[k][1]] )
for n in sorted(gridsearch_list, key=lambda y: y[2]):
    print(n)