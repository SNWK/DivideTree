import pickle

with open('../../data/regionTreeSeqs/andes_peruDGMGL.txt', 'rb') as f:
    data = pickle.load(f)

new = [data[0] for i in range(1000)]

with open('../oneData.txt', 'wb') as f:
    data = pickle.dump(new, f)
