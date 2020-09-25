# https://github.com/nicola-decao/MolGAN/blob/master/utils/sparse_molecular_dataset.py

import pickle, os
import numpy as np

from datetime import datetime


class SparseMolecularDataset():

    def load(self, filename, subset=1):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

        self.train_idx = np.random.choice(self.train_idx, int(len(self.train_idx) * subset), replace=False)
        self.validation_idx = np.random.choice(self.validation_idx, int(len(self.validation_idx) * subset),
                                               replace=False)
        self.test_idx = np.random.choice(self.test_idx, int(len(self.test_idx) * subset), replace=False)

        self.train_count = len(self.train_idx)
        self.validation_count = len(self.validation_idx)
        self.test_count = len(self.test_idx)

        self.__len = self.train_count + self.validation_count + self.test_count

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def generate(self, filename, add_h=False, filters=lambda x: True, size=None, validation=0.1, test=0.1):
        self.log('Extracting {}..'.format(filename))

        with open(filename, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data[:size]

        self.log('Extracted {} '.format(len(self.data)))

        self._generate_encoders_decoders()
        self._generate_AX()

        self.data = np.array(self.data)
        self.data_A = np.stack(self.data_A)
        self.data_X = np.stack(self.data_X)
        self.data_D = np.stack(self.data_D)
        self.data_Le = np.stack(self.data_Le)
        self.data_Lv = np.stack(self.data_Lv)

        self._generate_train_validation_test(validation, test)

    def _generate_encoders_decoders(self):
        self.log('Creating atoms encoder and decoder..')
        self.atom_num_types = 4 # feature dimensions
        self.log('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))

        self.log('Creating bonds encoder and decoder..')
        bond_labels = [0,1]

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        print(self.bond_encoder_m, self.bond_decoder_m)
        self.bond_num_types = len(bond_labels)
        self.log('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))


    def _generate_AX(self):
        self.log('Creating features and adjacency matrices..')

        data = []
        data_A = []
        data_X = []
        data_D = []
        data_Le = []
        data_Lv = []

        max_length = max(tree[0] for tree in self.data)
        print(max_length)
        for i, mol in enumerate(self.data):
            A = self._genA(mol, connected=True, max_length=max_length)
            D = np.count_nonzero(A, -1)
            if A is not None:
                data_A.append(A)
                X = self._genX(mol, max_length=max_length)
                data_X.append(X)
                data_D.append(D)
                
                data.append(A)
                L = D - A
                Le, Lv = np.linalg.eigh(L)

                data_Le.append(Le)
                data_Lv.append(Lv)
        self.log(date=False)
        self.log('Created {} features and adjacency matrices  out of {} molecules!'.format(len(data),
                                                                                           len(self.data)))

        self.data = data
        self.data_A = data_A
        self.data_X = data_X
        self.data_D = data_D
        self.data_Le = data_Le
        self.data_Lv = data_Lv
        self.__len = len(self.data)

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol[0]
        A = np.array(mol[1])
        A = np.pad(A,((0,max_length-mol[0]),(0,max_length-mol[0])),'constant',constant_values = (0,0))

        degree = np.sum(A[:mol[0], :mol[0]], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):

        max_length = max_length if max_length is not None else mol[0]
        X = mol[2]
        X += [[0, 0, 0, 0] for i in range(max_length - mol[0]) ]
        X = np.array(X)
        return np.array(X)

    def _generate_train_validation_test(self, validation, test):

        self.log('Creating train, validation and test sets..')

        validation = int(validation * len(self))
        test = int(test * len(self))
        train = len(self) - validation - test

        self.all_idx = np.random.permutation(len(self))
        self.train_idx = self.all_idx[0:train]
        self.validation_idx = self.all_idx[train:train + validation]
        self.test_idx = self.all_idx[train + validation:]

        self.train_counter = 0
        self.validation_counter = 0
        self.test_counter = 0

        self.train_count = train
        self.validation_count = validation
        self.test_count = test

        self.log('Created train ({} items), validation ({} items) and test ({} items) sets!'.format(
            train, validation, test))

    def _next_batch(self, counter, count, idx, batch_size):
        if batch_size is not None:
            if counter + batch_size >= count:
                counter = 0
                np.random.shuffle(idx)

            output = [obj[idx[counter:counter + batch_size]]
                      for obj in (self.data, self.data_A, self.data_X, self.data_D, self.data_Le, self.data_Lv)]

            counter += batch_size
        else:
            output = [obj[idx] for obj in (self.data, self.data_A, self.data_X, self.data_D, self.data_Le, self.data_Lv)]

        return [counter] + output

    def next_train_batch(self, batch_size=None):
        out = self._next_batch(counter=self.train_counter, count=self.train_count,
                               idx=self.train_idx, batch_size=batch_size)
        self.train_counter = out[0]

        return out[1:]

    def next_validation_batch(self, batch_size=None):
        out = self._next_batch(counter=self.validation_counter, count=self.validation_count,
                               idx=self.validation_idx, batch_size=batch_size)
        self.validation_counter = out[0]

        return out[1:]

    def next_test_batch(self, batch_size=None):
        out = self._next_batch(counter=self.test_counter, count=self.test_count,
                               idx=self.test_idx, batch_size=batch_size)
        self.test_counter = out[0]

        return out[1:]

    @staticmethod
    def log(msg='', date=True):
        print(str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) + ' ' + str(msg) if date else str(msg))

    def __len__(self):
        return self.__len


if __name__ == '__main__':
    data = SparseMolecularDataset()
    data.generate('dataGAN/data31.pkl')
    data.save('dataGAN/31nodes.sparsedataset')

    # data = SparseMolecularDataset()
    # data.generate('data/qm9_5k.smi', validation=0.00021, test=0.00021)  # , filters=lambda x: x.GetNumAtoms() <= 9)
    # data.save('data/qm9_5k.sparsedataset')
