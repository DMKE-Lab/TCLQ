import time, datetime
import numpy as np
import pickle
import os
import json
import dill
from collections import defaultdict


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


root = 'icews18/'
file_name = 'ts2id'
rename = 'ts2id'
rename2 = 'id2ts'

e = 'e'
r = 'r'
n = 'n'
u = 'u'
t = 't'
st = 'st'
et = 'et'
valid_test_query_structures = [
    [e, [r, [st, et]]],
    [e, [r, [st, et], r, [st, et]]],
    [e, [r, [st, et], r, [st, et], r, [st, et]]],
    [[e, [r, [st, et]]], [e, [r, [st, et]]]],
    [[e, [r, [st, et]]], [e, [r, [st, et]]], [e, [r, [st, et]]]],
    [[e, [r, [st, et], r, [st, et]]], [e, [r, [st, et]]]],
    [[[e, [r, [st, et]]], [e, [r, [st, et]]]], [r, [st, et]]],
    # negation 7
    [[e, [r, [st, et]]], [e, [r, [st, et], n]]],
    [[e, [r, [st, et]]], [e, [r, [st, et]]], [e, [r, [st, et], n]]],
    [[e, [r, [st, et], r, [st, et]]], [e, [r, [st, et], n]]],
    [[e, [r, [st, et], r, [st, et], n]], [e, [r, [st, et]]]],
    [[[e, [r, [st, et]]], [e, [r, [st, et], n]]], [r, [st, et]]],
    # union 12
    [[e, [r, [st, et]]], [e, [r, [st, et]]], [u]],
    [[[e, [r, [st, et]]], [e, [r, [st, et]]], [u]], [r, [st, et]]]
]

train_query_structures = [
    [e, [r, [st, et]]],
    [e, [r, [st, et], r, [st, et]]],
    [e, [r, [st, et], r, [st, et], r, [st, et]]],
    [[e, [r, [st, et]]], [e, [r, [st, et]]]],
    [[e, [r, [st, et]]], [e, [r, [st, et]]], [e, [r, [st, et]]]],
    # negation 7
    [[e, [r, [st, et]]], [e, [r, [st, et], n]]],
    [[e, [r, [st, et]]], [e, [r, [st, et]]], [e, [r, [st, et], n]]],
    [[e, [r, [st, et], r, [st, et]]], [e, [r, [st, et], n]]],
    [[e, [r, [st, et], r, [st, et], n]], [e, [r, [st, et]]]],
    [[[e, [r, [st, et]]], [e, [r, [st, et], n]]], [r, [st, et]]],
    # union 12
]
train_query_names = ['1pts', '2pts', '3pts', '2its', '3its', '2ints', '3ints', 'pints', 'pnits', 'inpts']  #3its不应该有
valid_query_names = ['1pts', '2pts', '3pts', '2its', '3its', 'pits', 'ipts', '2ints', '3ints', 'pints', 'pnits',
                     'inpts', '2uts', 'upts']
test_query_names = ['1pts', '2pts', '3pts', '2its', '3its', 'pits', 'ipts', '2ints', '3ints', 'pints', 'pnits',
                    'inpts', '2uts', 'upts']


def gen_queries(dataset, name, dic='data'):
    queries = defaultdict(set)
    index = 0
    if name == 'train':
        query_names = train_query_names
        query_structures = train_query_structures
    elif name == 'valid':
        query_names = valid_query_names
        query_structures = valid_test_query_structures
    elif name == 'test':
        query_names = test_query_names
        query_structures = valid_test_query_structures
    else:
        print('Error!!')
        exit(-1)
    for query_name in query_names:
        qs = list2tuple(query_structures[index])
        if os.path.exists('./data/%s/%s-%s-queries.pkl' % (dataset, name, query_name)):
            with open('./data/%s/%s-%s-queries.pkl' % (dataset, name, query_name), 'rb') as f:
                q = pickle.load(f, encoding='bytes')
                queries[qs] = q[qs]
                index = index + 1
        else:
            queries[qs] = set()
            index = index + 1
    with open('./data/%s/%s/%s-queries-ts.pkl' % (dataset, dic, name), 'wb') as w:
        pickle.dump(queries, w, protocol=pickle.HIGHEST_PROTOCOL)
    print('Success :./data/%s/%s/%s-queries-ts.pkl' % (dataset, dic, name))


def gen_answers(dataset, name, dic='data'):
    answers = defaultdict(set)
    index = 0
    if name == 'train':
        query_names = train_query_names
        query_structures = train_query_structures
    elif name == 'valid':
        query_names = valid_query_names
        query_structures = valid_test_query_structures
    elif name == 'test':
        query_names = test_query_names
        query_structures = valid_test_query_structures
    else:
        print('Error!!')
        exit(-1)
    for query_name in query_names:
        qs = list2tuple(query_structures[index])
        if os.path.exists('./data/%s/%s-%s-queries.pkl' % (dataset, name, query_name)):
            with open('./data/%s/%s-%s-fn-answers.pkl' % (dataset, name, query_name), 'rb') as f:
                ans = pickle.load(f, encoding='bytes')
                for key in ans:
                    answers[key] = ans[key]
                index = index + 1
    with open('./data/%s/%s/%s-answers-ts.pkl' % (dataset, dic, name), 'wb') as w:
        pickle.dump(answers, w, protocol=pickle.HIGHEST_PROTOCOL)
    print('Success :./data/%s/%s/%s-answers-ts.pkl' % (dataset, dic, name))


def gen_v_t_queries(dataset, name, dic='data'):
    queries = defaultdict(set)
    index = 0
    if name == 'valid':
        query_names = valid_query_names
        query_structures = valid_test_query_structures
    elif name == 'test':
        query_names = test_query_names
        query_structures = valid_test_query_structures
    else:
        print('Error!!')
        exit(-1)
    for query_name in query_names:
        qs = list2tuple(query_structures[index])
        if os.path.exists('./data/%s/%s-%s-queries.pkl' % (dataset, name, query_name)):
            with open('./data/%s/%s-%s-fn-answers.pkl' % (dataset, name, query_name), 'rb') as f:
                ans = pickle.load(f, encoding='bytes')
                for key in ans:
                    queries[key] = ans[key]
        index = index + 1




if __name__ == '__main__':
    dataset = 'icews18'
    name = 'test'
    dic = 'data'
    # gen_queries(dataset, name, dic)
    # gen_answers(dataset, name, dic)

    # with open('./data/%s/%s/%s-queries-ts.pkl' % (dataset, dic, name), 'rb') as f1:
    #     # 只能用dill来读取数据
    #     # dil = dill.load(f, encoding='bytes')
    #     queries = pickle.load(f1, encoding='bytes')  # 出错
    # print(queries.keys())

    # with open('./data/%s/%s/%s-answers-ts.pkl' % (dataset, dic, name), 'rb') as f1:
    #     # 只能用dill来读取数据
    #     # dil = dill.load(f, encoding='bytes')
    #     queries = pickle.load(f1, encoding='bytes')  # 出错
    # print(queries.keys())

    with open('./data/%s/train-1pt-fn-answers.pkl' % dataset, 'rb') as f1:
        # 只能用dill来读取数据
        # dil = dill.load(f, encoding='bytes')
        queries = pickle.load(f1, encoding='bytes')  # 出错
    print(queries.keys())

    # with open('./%s/%s-easy-answers.pkl' % ('FB15k-237-betae', 'valid'), 'rb') as f1:
    #     # 只能用dill来读取数据
    #     # dil = dill.load(f, encoding='bytes')
    #     pic = pickle.load(f1, encoding='bytes')  # 出错

    # with open('./%s/%s-hard-answers.pkl' % (dataset, name), 'rb') as f1:
    #     # 只能用dill来读取数据
    #     # dil = dill.load(f, encoding='bytes')
    #     hard_answers = pickle.load(f1, encoding='bytes')  # 出错

    # with open('./%s/%s-answers.pkl' % ('FB15k-237-betae', 'train'), 'rb') as f2:
    #     train_answers = pickle.load(f2, encoding='bytes')

    # with open('./%s/%s-queries.pkl' % ('FB15k-237-betae', 'valid'), 'rb') as f2:
    #     train_queries = pickle.load(f2, encoding='bytes')
    # print(train_queries)
