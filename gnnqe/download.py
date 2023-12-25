import os
import pickle

# from gnnqe import dataset

if __name__ == "__main__":
    # path = '/data/xls/emb_code/GNN-QE-master/kg-datasets/FB15k-237-betae/id2ent.pkl'
    path = 'E:\A_Study\Experiment\GNN-QE-master\kg-datasets\FB15k-237-betae\id2ent.pkl'
    with open(path, "rb") as fin:
        entity_vocab = pickle.load(fin)
        print(entity_vocab)