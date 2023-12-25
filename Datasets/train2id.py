import numpy as np
import pickle
import time, datetime
import json

root = 'icews18/'
file_name = 'valid'
rename = 'valid'

quadruple = []
quadruple2id = []
num_sample = 0

# x = 1098738024
# date = time.localtime(x)

# 加载id
with open('icews18/dataset/ent2id.pkl', "rb") as f:
    ent2id = pickle.load(f, encoding='bytes')
    f.close()

with open('icews18/dataset/rel2id.pkl', "rb") as f:
    rel2id = pickle.load(f, encoding='bytes')
    f.close()

with open(root + "dataset/" + rename + '.txt', "w") as ft:  # write
    with open(root + file_name + '.txt', 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t, stime = [x for x in line.split('\t')]
            stime = stime.replace('\n', '')
            dtime = time.strptime(stime, '%Y-%m-%d')
            ttime = int(time.mktime(dtime))

            quadruple.append((h, r, t, ttime))
            quadruple2id.append((ent2id[h], rel2id[r], ent2id[t], ttime))
            ft.write(str(ent2id[h]) + '\t' + str(rel2id[r]) + '\t' + str(ent2id[t]) + '\t' + str(ttime) + '\n')
            num_sample += 1
        f.close()
    ft.close()

with open(root + "dataset/" + rename + '_word.pkl', "wb") as fo:  # write
    pickle.dump(quadruple, fo)
    fo.close()

with open(root + "dataset/" + rename + '.pkl', "wb") as fo:  # write
    pickle.dump(quadruple2id, fo)
    fo.close()

with open(root + "dataset/" + rename + '.pkl', "rb") as fo:  # read
    A = pickle.load(fo, encoding='bytes')
    print(A)
    print('len:', len(A))
