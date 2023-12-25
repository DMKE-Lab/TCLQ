import time, datetime
import numpy as np
import pickle
import json

root = 'icews18/'
file_name = 'ts2id'
rename = 'ts2id'
rename2 = 'id2ts'

with open(root + file_name + '.json', 'r') as f:
    a = np.array(json.load(f))

list_data = a.tolist()
ts2id = {}
id2ts = {}
id2rel = {}
for k, v in list_data.items():
    dtime = time.strptime(k, '%Y-%m-%d')
    ttime = int(time.mktime(dtime))
    ts2id[ttime] = v
    id2ts[v] = ttime
    # id2rel[v] = k

with open(root + "dataset/" + rename + '.pkl', "wb") as fo:  # write
    pickle.dump(ts2id, fo)
    fo.close()

with open(root + "dataset/" + rename2 + '.pkl', "wb") as fo:  # write
    pickle.dump(id2ts, fo)
    fo.close()

with open(root + "dataset/" + rename + '.pkl', "rb") as fo:  # read
    A = pickle.load(fo, encoding='bytes')
    print(A)
    print('len:', len(A))
