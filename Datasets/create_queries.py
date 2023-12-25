import pickle
import dill
import os.path as osp
from functools import partial

import numpy as np
import click
from collections import defaultdict
import random
from copy import deepcopy
import time
import pdb
import logging
import os


def set_logger(save_path, query_name, print_on_screen=False):
    '''
    Write logs to checkpoint and console
    '''

    log_file = os.path.join(save_path, '%s.log' % (query_name))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def index_dataset(dataset_name, force=False):
    print('Indexing dataset {0}'.format(dataset_name))
    base_path = 'data/{0}/'.format(dataset_name)
    # files = ['train.txt', 'valid.txt', 'test.txt']
    # indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']
    files = ['train.txt']
    indexified_files = ['train_indexified.txt']
    return_flag = True
    for i in range(len(indexified_files)):
        if not osp.exists(osp.join(base_path, indexified_files[i])):
            return_flag = False
            break
    if return_flag and not force:
        print("index file exists")
        return

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    entid, relid = 0, 0

    with open(osp.join(base_path, files[0])) as f:
        lines = f.readlines()
        file_len = len(lines)

    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(base_path, indexified_p), "w")
        with open(osp.join(base_path, p), 'r') as f:
            for i, line in enumerate(f):
                print('[%d/%d]' % (i, file_len), end='\r')
                e1, rel, e2 = line.split('\t')
                e1 = e1.strip()
                e2 = e2.strip()
                rel = rel.strip()
                rel_reverse = '-' + rel
                rel = '+' + rel
                # rel_reverse = rel+ '_reverse'

                if p == "train.txt":
                    if e1 not in ent2id.keys():
                        ent2id[e1] = entid
                        id2ent[entid] = e1
                        entid += 1

                    if e2 not in ent2id.keys():
                        ent2id[e2] = entid
                        id2ent[entid] = e2
                        entid += 1

                    if not rel in rel2id.keys():
                        rel2id[rel] = relid
                        id2rel[relid] = rel
                        assert relid % 2 == 0
                        relid += 1

                    if not rel_reverse in rel2id.keys():
                        rel2id[rel_reverse] = relid
                        id2rel[relid] = rel_reverse
                        assert relid % 2 == 1
                        relid += 1

                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
        fw.close()

    with open(osp.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('num entity: %d, num relation: %d' % (len(ent2id), len(rel2id)))
    print("indexing finished!!")


def construct_graph(base_path, indexified_files, is_time, is_time_interval):
    # knowledge graph
    # kb[e][rel] = set([e, e, e])
    if is_time:
        ent_in, ent_out = defaultdict(lambda: defaultdict(lambda: defaultdict(set))), defaultdict(
            lambda: defaultdict(lambda: defaultdict(set)))
    else:
        ent_in, ent_out = defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    for indexified_p in indexified_files:
        with open(osp.join(base_path, indexified_p)) as f:
            for i, line in enumerate(f):
                if len(line) == 0:
                    continue
                if is_time:
                    e1, rel, e2, t = line.split('\t')
                    e1 = int(e1.strip())
                    e2 = int(e2.strip())
                    rel = int(rel.strip())
                    t = int(t.strip())
                    '''
                    {234: defaultdict(<function construct_graph.<locals>.<lambda>.<locals>.<lambda> at 0x000001B6132E4CA0>, 
                        {93: defaultdict(<class 'set'>, 
                            {1514736000: 
                                {22204}
                            })
                        })
                    }
                    '''
                    ent_out[e1][rel][t].add(e2)
                    ent_in[e2][rel][t].add(e1)
                else:
                    e1, rel, e2 = line.split('\t')
                    e1 = int(e1.strip())
                    e2 = int(e2.strip())
                    rel = int(rel.strip())
                    ent_out[e1][rel].add(e2)
                    ent_in[e2][rel].add(e1)

    return ent_in, ent_out


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


def write_links(dataset, ent_out, small_ent_out, max_ans_num, name, is_time, is_time_interval):
    if is_time_interval:
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        num_more_answer = 0
        for ent in ent_out:
            for rel in ent_out[ent]:
                # 1. 基本 时间区间 的数据
                # tmp_ans = defaultdict(set)
                for t in ent_out[ent][rel]:
                    if len(ent_out[ent][rel][t]) <= max_ans_num:
                        # [e, [r, [st, et]]]
                        queries[('e', ('r', ('st', 'et')))].add((ent, (rel, (t[0], t[1]))))
                        # queries[('e', ('r', ('t',)))].add((ent, (rel, (t,))))
                        tp_answers[(ent, (rel, (t[0], t[1])))] = small_ent_out[ent][rel][t]
                        fn_answers[(ent, (rel, (t[0], t[1])))] = ent_out[ent][rel][t]
                        # 组合
                        # tmp_ans[(t[0], t[1])] = ent_out_ts[ent][rel][t]
                    else:
                        num_more_answer += 1
                # # 2. 对基础区间进行 合并
                # while len(tmp_ans) > 1:
                #     tmp_keys = sorted(tmp_ans.keys(), key=lambda x: x[0])
                #     tmp_dic = defaultdict(set)
                #     tmp = set()
                #     flag = False
                #     left = tmp_keys[0][0]
                #     right = tmp_keys[0][1]
                #     for key1 in tmp_keys:
                #         if key1 == tmp_keys[-1] and flag == False:
                #             # 如果是最后一个元素 且 是奇数
                #             tmp = tmp_ans[key1] | tmp_dic.pop((left, right))
                #             right = key1[1]
                #             tmp_dic[left, right] = tmp
                #             continue
                #         if flag:
                #             right = key1[1]
                #             tmp = tmp | tmp_ans[key1]
                #             tmp_dic[(left, right)] = tmp
                #             tmp = set()
                #             flag = not flag
                #         else:
                #             left = key1[0]
                #             right = key1[1]
                #             tmp = tmp_ans[key1]
                #             flag = not flag
                #     tmp_ans = tmp_dic
                #     for key2, value2 in tmp_dic.items():
                #         queries[('e', ('r', ('st', 'et')))].add((ent, (rel, (key2[0], key2[1]))))
                #         tp_answers[(ent, (rel, (key2[0], key2[1])))] = small_ent_out[ent][rel][key2]
                #         fn_answers[(ent, (rel, (key2[0], key2[1])))] = value2
    elif is_time:
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        num_more_answer = 0
        for ent in ent_out:
            for rel in ent_out[ent]:
                for t in ent_out[ent][rel]:
                    if len(ent_out[ent][rel][t]) <= max_ans_num:
                        queries[('e', ('r', ('t',)))].add((ent, (rel, (t,))))
                        # queries[('e', ('r', ('t',)))].add((ent, (rel, (t,))))
                        tp_answers[(ent, (rel, (t,)))] = small_ent_out[ent][rel][t]
                        fn_answers[(ent, (rel, (t,)))] = ent_out[ent][rel][t]
                    else:
                        num_more_answer += 1
    else:
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        num_more_answer = 0
        for ent in ent_out:
            for rel in ent_out[ent]:
                if len(ent_out[ent][rel]) <= max_ans_num:
                    queries[('e', ('r',))].add((ent, (rel,)))
                    tp_answers[(ent, (rel,))] = small_ent_out[ent][rel]
                    fn_answers[(ent, (rel,))] = ent_out[ent][rel]
                else:
                    num_more_answer += 1

    with open('./data/%s/%s-queries.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(tp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name), 'wb') as f:
        pickle.dump(fp_answers, f)
    print(num_more_answer)


def time_convert(dataset, ent_out, name, is_time, is_time_interval):
    if not is_time_interval:
        print('time_convert Error!!!')
        exit(-1)
    with open('data/' + dataset + '/ts2id.pkl', "rb") as fo:  # read
        ts2id = pickle.load(fo, encoding='bytes')
    with open('data/' + dataset + '/id2ts.pkl', "rb") as fo:  # read
        id2ts = pickle.load(fo, encoding='bytes')

    ent_out_ts = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    for ent in ent_out:
        for rel in ent_out[ent]:
            # 写一个函数，将时间步连续的合并为区间
            left, right, pt = -1, -1, -1
            tmp_set = set()
            tmp_ans = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            for t in ent_out[ent][rel]:
                if left == -1 or right == -1:
                    left, right = t, t
                    pt = ts2id[t]
                    # print(ent_out[ent][rel][t])
                    tmp_set = tmp_set | ent_out[ent][rel][t]
                    continue
                if t == id2ts[pt + 1]:
                    right = t
                    pt = ts2id[t]
                    tmp_set = tmp_set | ent_out[ent][rel][t]
                else:
                    ent_out_ts[ent][rel][(left, right)] = tmp_set
                    tmp_ans[ent][rel][(left, right)] = tmp_set
                    left = t
                    right = t
                    pt = ts2id[t]
                    tmp_set = set()
                    tmp_set = tmp_set | ent_out[ent][rel][t]
            # 2. 对基础区间进行 合并
            while len(tmp_ans[ent][rel]) > 1:
                tmp_keys = sorted(tmp_ans[ent][rel].keys(), key=lambda x: x[0])
                tmp_dic = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
                tmp = set()
                flag = False
                left = tmp_keys[0][0]
                right = tmp_keys[0][1]
                for key1 in tmp_keys:
                    if key1 == tmp_keys[-1] and flag == False:
                        # 如果是最后一个元素 且 是奇数
                        tmp = tmp_ans[ent][rel][key1] | tmp_dic[ent][rel].pop((left, right))
                        right = key1[1]
                        tmp_dic[ent][rel][(left, right)] = tmp
                        continue
                    if flag:
                        right = key1[1]
                        tmp = tmp | tmp_ans[ent][rel][key1]
                        tmp_dic[ent][rel][(left, right)] = tmp
                        tmp = set()
                        flag = not flag
                    else:
                        left = key1[0]
                        right = key1[1]
                        tmp = tmp_ans[ent][rel][key1]
                        flag = not flag
                tmp_ans = tmp_dic
                for key2, value2 in tmp_dic[ent][rel].items():
                    ent_out_ts[ent][rel][key2] = value2

    num_out = 0
    ent_in_ts = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for ent in ent_out_ts:
        for rel in ent_out_ts[ent]:
            num_out = num_out + len(ent_out_ts[ent][rel].keys())
            for ts in ent_out_ts[ent][rel]:
                for ent_in in ent_out_ts[ent][rel][ts]:
                    ent_in_ts[ent_in][rel][ts].add(ent)
    for ent in ent_in_ts:
        if len(ent_in_ts[ent].keys()) < 1:
            print(ent)
            ent_in_ts.pop(ent)
            print(ent_in_ts.get(ent))
    print(len(ent_in_ts[14531].keys()))
    print("ent_out_ts has %s examples" % num_out)

    with open('./data/%s/%s_ent_out_ts.pkl' % (dataset, name), 'wb') as f:
        dill.dump(ent_out_ts, f)

    with open('./data/%s/%s_ent_in_ts.pkl' % (dataset, name), 'wb') as f:
        dill.dump(ent_in_ts, f)

    with open('./data/%s/%s_ent_out_ts.pkl' % (dataset, name), 'rb') as f:
        # 只能用dill来读取数据
        dil = dill.load(f, encoding='bytes')
        # pic = pickle.load(f, encoding='bytes') #出错

    return ent_out_ts, ent_in_ts


def ground_queries(dataset, query_structure, ent_in, ent_out, small_ent_in, small_ent_out, gen_num, max_ans_num,
                   query_name, mode, ent2id, rel2id, is_time, is_time_interval):
    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_no_extra_answer, num_no_extra_negative, num_empty = 0, 0, 0, 0, 0, 0, 0, 0
    tp_ans_num, fp_ans_num, fn_ans_num = [], [], []
    if is_time:
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        s0 = time.time()
        old_num_sampled = -1
        while num_sampled < gen_num and (time.time() - s0) < 5400:
            if num_sampled != 0:
                if num_sampled % (gen_num // 100) == 0 and num_sampled != old_num_sampled:
                    logging.info(
                        '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                            mode,
                            query_structure,
                            num_sampled, gen_num, (time.time() - s0) / num_sampled, num_try, num_repeat,
                            num_more_answer,
                            num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))
                    old_num_sampled = num_sampled
            # if num_sampled % 2 == 0 and num_sampled != 0 and num_sampled != old_num_sampled:
            #     print(
            #         '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
            #             mode,
            #             query_structure,
            #             num_sampled, gen_num, (time.time() - s0) / (num_sampled + 0.001), num_try, num_repeat,
            #             num_more_answer,
            #             num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
            print(
                '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                    mode,
                    query_structure,
                    num_sampled, gen_num, (time.time() - s0) / (num_sampled + 0.001), num_try, num_repeat,
                    num_more_answer,
                    num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
            num_try += 1
            empty_query_structure = deepcopy(query_structure)
            answer = random.sample(ent_in.keys(), 1)[0]
            broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id, is_time,
                                     is_time_interval)
            if broken_flag:
                num_broken += 1
                continue
            query = empty_query_structure
            answer_set = achieve_answer(query, ent_in, ent_out, is_time, is_time_interval)
            small_answer_set = achieve_answer(query, small_ent_in, small_ent_out, is_time, is_time_interval)
            if len(answer_set) == 0:
                num_empty += 1
                continue
            if mode != 'train':
                if len(answer_set - small_answer_set) == 0:
                    num_no_extra_answer += 1
                    continue
                if 'n' in query_name:
                    if len(small_answer_set - answer_set) == 0:
                        num_no_extra_negative += 1
                        continue
            if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > max_ans_num:
                num_more_answer += 1
                continue
            if list2tuple(query) in queries[list2tuple(query_structure)]:
                num_repeat += 1
                continue
            queries[list2tuple(query_structure)].add(list2tuple(query))
            tp_answers[list2tuple(query)] = small_answer_set
            fp_answers[list2tuple(query)] = small_answer_set - answer_set
            fn_answers[list2tuple(query)] = answer_set - small_answer_set
            num_sampled += 1
            tp_ans_num.append(len(tp_answers[list2tuple(query)]))
            fp_ans_num.append(len(fp_answers[list2tuple(query)]))
            fn_ans_num.append(len(fn_answers[list2tuple(query)]))
    else:
        queries = defaultdict(set)
        tp_answers = defaultdict(set)
        fp_answers = defaultdict(set)
        fn_answers = defaultdict(set)
        s0 = time.time()
        old_num_sampled = -1
        while num_sampled < gen_num:
            if num_sampled != 0:
                if num_sampled % (gen_num // 100) == 0 and num_sampled != old_num_sampled:
                    logging.info(
                        '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                            mode,
                            query_structure,
                            num_sampled, gen_num, (time.time() - s0) / num_sampled, num_try, num_repeat,
                            num_more_answer,
                            num_broken, num_no_extra_answer, num_no_extra_negative, num_empty))
                    old_num_sampled = num_sampled
            print(
                '%s %s: [%d/%d], avg time: %s, try: %s, repeat: %s: more_answer: %s, broken: %s, no extra: %s, no negative: %s empty: %s' % (
                    mode,
                    query_structure,
                    num_sampled, gen_num, (time.time() - s0) / (num_sampled + 0.001), num_try, num_repeat,
                    num_more_answer,
                    num_broken, num_no_extra_answer, num_no_extra_negative, num_empty), end='\r')
            num_try += 1
            empty_query_structure = deepcopy(query_structure)
            answer = random.sample(ent_in.keys(), 1)[0]
            if len(ent_in[answer].keys()) < 1:
                print(answer)
            broken_flag = fill_query(empty_query_structure, ent_in, ent_out, answer, ent2id, rel2id)
            if broken_flag:
                num_broken += 1
                continue
            query = empty_query_structure
            answer_set = achieve_answer(query, ent_in, ent_out)
            small_answer_set = achieve_answer(query, small_ent_in, small_ent_out)
            if len(answer_set) == 0:
                num_empty += 1
                continue
            if mode != 'train':
                if len(answer_set - small_answer_set) == 0:
                    num_no_extra_answer += 1
                    continue
                if 'n' in query_name:
                    if len(small_answer_set - answer_set) == 0:
                        num_no_extra_negative += 1
                        continue
            if max(len(answer_set - small_answer_set), len(small_answer_set - answer_set)) > max_ans_num:
                num_more_answer += 1
                continue
            if list2tuple(query) in queries[list2tuple(query_structure)]:
                num_repeat += 1
                continue
            queries[list2tuple(query_structure)].add(list2tuple(query))
            tp_answers[list2tuple(query)] = small_answer_set
            fp_answers[list2tuple(query)] = small_answer_set - answer_set
            fn_answers[list2tuple(query)] = answer_set - small_answer_set
            num_sampled += 1
            tp_ans_num.append(len(tp_answers[list2tuple(query)]))
            fp_ans_num.append(len(fp_answers[list2tuple(query)]))
            fn_ans_num.append(len(fn_answers[list2tuple(query)]))

    print()
    logging.info("{} tp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(tp_ans_num), np.min(tp_ans_num),
                                                                    np.mean(tp_ans_num), np.std(tp_ans_num)))
    logging.info("{} fp max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fp_ans_num), np.min(fp_ans_num),
                                                                    np.mean(fp_ans_num), np.std(fp_ans_num)))
    logging.info("{} fn max: {}, min: {}, mean: {}, std: {}".format(mode, np.max(fn_ans_num), np.min(fn_ans_num),
                                                                    np.mean(fn_ans_num), np.std(fn_ans_num)))

    name_to_save = '%s-%s' % (mode, query_name)
    with open('./data/%s/%s-queries.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(queries, f)
    with open('./data/%s/%s-fp-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(fp_answers, f)
    with open('./data/%s/%s-fn-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(fn_answers, f)
    with open('./data/%s/%s-tp-answers.pkl' % (dataset, name_to_save), 'wb') as f:
        pickle.dump(tp_answers, f)
    return queries, tp_answers, fp_answers, fn_answers


def defaultdict_set():
    return defaultdict(set)


def generate_queries(dataset, query_structures, gen_num, max_ans_num, gen_train, gen_valid, gen_test, query_names,
                     save_name, is_time, is_time_interval, is_time_convert):
    base_path = './data/%s' % dataset
    indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']
    # ent_out_ts = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    if gen_train or gen_valid:
        train_ent_in, train_ent_out = construct_graph(base_path, indexified_files[:1], is_time, is_time_interval)  # ent_in
        if is_time_interval and is_time_convert:
            train_ent_out_ts, train_ent_in_ts = time_convert(dataset, train_ent_out, 'train', is_time, is_time_interval)
        elif is_time_interval:
            train_ent_out_ts = dill.load(open(os.path.join(base_path, "train_ent_out_ts.pkl"), 'rb'))
            train_ent_in_ts = dill.load(open(os.path.join(base_path, "train_ent_in_ts.pkl"), 'rb'))
    if gen_valid or gen_test:
        valid_ent_in, valid_ent_out = construct_graph(base_path, indexified_files[:2], is_time, is_time_interval)
        valid_only_ent_in, valid_only_ent_out = construct_graph(base_path, indexified_files[1:2], is_time,
                                                                is_time_interval)
        if is_time_interval and is_time_convert:
            valid_ent_out_ts, valid_ent_in_ts = time_convert(dataset, valid_ent_out, 'valid', is_time, is_time_interval)
            valid_only_ent_out_ts, valid_only_ent_in_ts = time_convert(dataset, valid_only_ent_out, 'valid_only',
                                                                       is_time, is_time_interval)
        elif is_time_interval:
            valid_ent_out_ts = dill.load(open(os.path.join(base_path, "valid_ent_out_ts.pkl"), 'rb'))
            valid_ent_in_ts = dill.load(open(os.path.join(base_path, "valid_ent_in_ts.pkl"), 'rb'))
            valid_only_ent_out_ts = dill.load(open(os.path.join(base_path, "valid_only_ent_out_ts.pkl"), 'rb'))
            valid_only_ent_in_ts = dill.load(open(os.path.join(base_path, "valid_only_ent_in_ts.pkl"), 'rb'))
    if gen_test:
        test_ent_in, test_ent_out = construct_graph(base_path, indexified_files[:3], is_time, is_time_interval)
        test_only_ent_in, test_only_ent_out = construct_graph(base_path, indexified_files[2:3], is_time,
                                                              is_time_interval)
        if is_time_interval and is_time_convert:
            test_ent_out_ts, test_ent_in_ts = time_convert(dataset, test_ent_out, 'test', is_time, is_time_interval)
            test_only_ent_out_ts, test_only_ent_in_ts = time_convert(dataset, test_only_ent_out, 'test_only',
                                                                     is_time, is_time_interval)
        elif is_time_interval:
            test_ent_out_ts = dill.load(open(os.path.join(base_path, "test_ent_out_ts.pkl"), 'rb'))
            test_ent_in_ts = dill.load(open(os.path.join(base_path, "test_ent_in_ts.pkl"), 'rb'))
            test_only_ent_out_ts = dill.load(open(os.path.join(base_path, "test_only_ent_out_ts.pkl"), 'rb'))
            test_only_ent_in_ts = dill.load(open(os.path.join(base_path, "test_only_ent_in_ts.pkl"), 'rb'))

    ent2id = pickle.load(open(os.path.join(base_path, "ent2id.pkl"), 'rb'))
    rel2id = pickle.load(open(os.path.join(base_path, "rel2id.pkl"), 'rb'))

    if is_time:
        train_queries = defaultdict(set)
        train_tp_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        train_fp_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        train_fn_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        valid_queries = defaultdict(set)
        valid_tp_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        valid_fp_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        valid_fn_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        test_queries = defaultdict(set)
        test_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        test_tp_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        test_fp_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        test_fn_answers = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    else:
        train_queries = defaultdict(set)
        train_tp_answers = defaultdict(set)
        train_fp_answers = defaultdict(set)
        train_fn_answers = defaultdict(set)
        valid_queries = defaultdict(set)
        valid_tp_answers = defaultdict(set)
        valid_fp_answers = defaultdict(set)
        valid_fn_answers = defaultdict(set)
        test_queries = defaultdict(set)
        test_answers = defaultdict(set)
        test_tp_answers = defaultdict(set)
        test_fp_answers = defaultdict(set)
        test_fn_answers = defaultdict(set)

    t1, t2, t3, t4, t5, t6 = 0, 0, 0, 0, 0, 0
    assert len(query_structures) == 1
    idx = 0
    query_structure = query_structures[idx]
    query_name = query_names[idx] if save_name else str(idx)
    print('general structure is', query_structure, "with name", query_name)
    if is_time:
        if query_structure == ['e', ['r', ['t']]]:
            if gen_train:
                """
                (dataset, ent_out, small_ent_out, max_ans_num, name, is_time, is_time_interval)
                """
                write_links(dataset, train_ent_out, defaultdict(lambda: defaultdict(lambda: defaultdict(set))),
                            max_ans_num, 'train-' + query_name, is_time, is_time_interval)
            if gen_valid:
                write_links(dataset, valid_only_ent_out, train_ent_out, max_ans_num, 'valid-' + query_name,
                            is_time, is_time_interval)
            if gen_test:
                write_links(dataset, test_only_ent_out, valid_ent_out, max_ans_num, 'test-' + query_name,
                            is_time, is_time_interval)
            print("link prediction created!")
            exit(-1)
        elif query_structure == ['e', ['r', ['st', 'et']]]:
            if gen_train:
                write_links(dataset, train_ent_out_ts, defaultdict(lambda: defaultdict(lambda: defaultdict(set))),
                            max_ans_num, 'train-' + query_name, is_time, is_time_interval)
            if gen_valid:
                write_links(dataset, valid_only_ent_out_ts, train_ent_out_ts, max_ans_num, 'valid-' + query_name,
                            is_time, is_time_interval)
            if gen_test:
                write_links(dataset, test_only_ent_out_ts, valid_ent_out_ts, max_ans_num, 'test-' + query_name,
                            is_time, is_time_interval)
            print("link prediction created!")
            exit(-1)
    else:
        if query_structure == ['e', ['r']]:
            if gen_train:
                write_links(dataset, train_ent_out, defaultdict(set), max_ans_num, 'train-' + query_name,
                            is_time, is_time_interval)
            if gen_valid:
                write_links(dataset, valid_only_ent_out, train_ent_out, max_ans_num, 'valid-' + query_name,
                            is_time, is_time_interval)
            if gen_test:
                write_links(dataset, test_only_ent_out, valid_ent_out, max_ans_num, 'test-' + query_name,
                            is_time, is_time_interval)
            print("link prediction created!")
            exit(-1)

    name_to_save = query_name
    if gen_train:
        t_v_t = 'train'
    elif gen_valid:
        t_v_t = 'valid'
    else:
        t_v_t = 'test'
    set_logger("./data/{}/{}/".format(dataset, t_v_t), name_to_save)

    num_sampled, num_try, num_repeat, num_more_answer, num_broken, num_empty = 0, 0, 0, 0, 0, 0
    train_ans_num = []
    s0 = time.time()
    if gen_train:
        if is_time_interval:
            train_queries, train_tp_answers, train_fp_answers, train_fn_answers = ground_queries(dataset,
                                                                                                 query_structure,
                                                                                                 train_ent_in_ts,
                                                                                                 train_ent_out_ts,
                                                        defaultdict(lambda: defaultdict(lambda: defaultdict(set))),
                                                        defaultdict(lambda: defaultdict(lambda: defaultdict(set))),
                                                                                                 gen_num[0],
                                                                                                 max_ans_num,
                                                                                                 query_name, 'train',
                                                                                                 ent2id, rel2id,
                                                                                                 is_time,
                                                                                                 is_time_interval)
        else:
            train_queries, train_tp_answers, train_fp_answers, train_fn_answers = ground_queries(dataset,
                                                                                            query_structure,
                                                                                            train_ent_in,
                                                                                            train_ent_out,
                                                                                defaultdict(lambda: defaultdict(set)),
                                                                                defaultdict(lambda: defaultdict(set)),
                                                                                            gen_num[0], max_ans_num,
                                                                                            query_name, 'train',
                                                                                            ent2id, rel2id, is_time,
                                                                                            is_time_interval)
    if gen_valid:
        if is_time_interval:
            valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers = ground_queries(dataset,
                                                                                                 query_structure,
                                                                                                 valid_ent_in_ts,
                                                                                                 valid_ent_out_ts,
                                                                                                 train_ent_in_ts,
                                                                                                 train_ent_out_ts,
                                                                                                 gen_num[1],
                                                                                                 max_ans_num,
                                                                                                 query_name,
                                                                                                 'valid', ent2id,
                                                                                                 rel2id,
                                                                                                 is_time,
                                                                                                 is_time_interval)
        else:
            valid_queries, valid_tp_answers, valid_fp_answers, valid_fn_answers = ground_queries(dataset,
                                                                                                 query_structure,
                                                                                                 valid_ent_in,
                                                                                                 valid_ent_out,
                                                                                                 train_ent_in,
                                                                                                 train_ent_out,
                                                                                                 gen_num[1],
                                                                                                 max_ans_num,
                                                                                                 query_name,
                                                                                                 'valid', ent2id,
                                                                                                 rel2id,
                                                                                                 is_time,
                                                                                                 is_time_interval)
    if gen_test:
        if is_time_interval:
            test_queries, test_tp_answers, test_fp_answers, test_fn_answers = ground_queries(dataset, query_structure,
                                                                                             test_ent_in_ts,
                                                                                             test_ent_out_ts,
                                                                                             valid_ent_in_ts,
                                                                                             valid_ent_out_ts,
                                                                                             gen_num[2], max_ans_num,
                                                                                             query_name, 'test', ent2id,
                                                                                             rel2id, is_time,
                                                                                             is_time_interval)
        else:
            test_queries, test_tp_answers, test_fp_answers, test_fn_answers = ground_queries(dataset, query_structure,
                                                                                             test_ent_in, test_ent_out,
                                                                                             valid_ent_in,
                                                                                             valid_ent_out,
                                                                                             gen_num[2], max_ans_num,
                                                                                             query_name, 'test', ent2id,
                                                                                             rel2id, is_time,
                                                                                             is_time_interval)
    print('%s queries generated with structure %s' % (gen_num, query_structure))


def fill_query(query_structure, ent_in, ent_out, answer, ent2id, rel2id, is_time=False, is_time_interval=False):
    assert type(query_structure[-1]) == list
    all_relation_flag = True
    if is_time:
        for ele in query_structure[-1]:
            if ele not in ['r', 'n', 't', ['st', 'et']]:
                all_relation_flag = False
                break
        if all_relation_flag:
            r = -1
            t_tmp = -1
            for i in range(len(query_structure[-1]))[::-1]:
                if query_structure[-1][i] == 'n':
                    query_structure[-1][i] = -2
                    continue
                if query_structure[-1][i] == ['st', 'et'] or query_structure[-1][i] == 't':
                    ii = i - 1
                    found = False
                    for j in range(40):
                        if len(ent_in[answer].keys()) < 1:
                            print(answer)
                            ent_in.pop(answer)
                            return True
                        r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                        # print(r_tmp // 2, r // 2, r_tmp == r)
                        if r_tmp // 2 != r // 2 or r_tmp == r:  # 防止出现环
                            r = r_tmp
                            found = True
                            break
                    if not found:
                        return True
                    query_structure[-1][ii] = r

                    t_tmp = random.sample(ent_in[answer][r].keys(), 1)[0]
                    query_structure[-1][i] = t_tmp
                    if len(ent_in[answer][r][t_tmp]) >= 1:
                        tmp_flag = False
                        for kk in range(40):
                            tmp_answer = random.sample(ent_in[answer][r][t_tmp], 1)[0]
                            if len(ent_in[tmp_answer].keys()) >= 1:
                                answer = tmp_answer
                                tmp_flag = True
                                break
                            ent_in.pop(tmp_answer)
                            # print("don't have the key %s, len < 1" % tmp_answer)
                            if len(set(ent_in[answer][r][t_tmp])) <= 1:
                                break
                        if tmp_flag == False:
                            return True
                    else:
                        tmp_flag = False
                        for kk in range(40):
                            tmp_answer = random.sample(ent_in[answer][r][t_tmp], 1)[0]
                            if len(ent_in[tmp_answer].keys()) >= 1:
                                answer = tmp_answer
                                tmp_flag = True
                                break
                            ent_in.pop(tmp_answer)
                            # print("don't have the key %s, len < 1" % tmp_answer)
                            if len(set(ent_in[answer][r][t_tmp])) <= 1:
                                break
                        if tmp_flag == False:
                            return True
                    continue
                if query_structure[-1][i] == 'r':
                    found = False
                    for j in range(40):
                        r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                        # print(r_tmp // 2, r // 2, r_tmp == r)
                        if r_tmp // 2 != r // 2 or r_tmp == r:  # 防止出现环
                            r = r_tmp
                            found = True
                            break
                    if not found:
                        return True
                    query_structure[-1][i] = r

            if query_structure[0] == 'e':
                query_structure[0] = answer
            else:
                return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id, is_time,
                                  is_time_interval)
        else:
            same_structure = defaultdict(list)
            for i in range(len(query_structure)):
                same_structure[list2tuple(query_structure[i])].append(i)
            for i in range(len(query_structure)):
                if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                    assert i == len(query_structure) - 1
                    query_structure[i][0] = -1
                    continue
                broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id, is_time,
                                         is_time_interval)
                if broken_flag:
                    return True
            """
            防止两个查询案例一模一样，
            如[[e, [r, [st, et]]], [e, [r, [st, et]]], [e, [r, [st, et], n]]]
            防止前两个生成的查询案例一模一样
            """
            for structure in same_structure:
                if len(same_structure[structure]) != 1:
                    structure_set = set()
                    for i in same_structure[structure]:
                        structure_set.add(list2tuple(query_structure[i]))
                    if len(structure_set) < len(same_structure[structure]):
                        return True
    else:
        for ele in query_structure[-1]:
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:
            r = -1
            for i in range(len(query_structure[-1]))[::-1]:
                if query_structure[-1][i] == 'n':
                    query_structure[-1][i] = -2
                    continue
                found = False
                for j in range(40):
                    r_tmp = random.sample(ent_in[answer].keys(), 1)[0]
                    if r_tmp // 2 != r // 2 or r_tmp == r:  # 防止出现环
                        r = r_tmp
                        found = True
                        break
                if not found:
                    return True
                query_structure[-1][i] = r
                if len(ent_in[answer][r]) >= 1:
                    answer = random.sample(ent_in[answer][r], 1)[0]
                else:
                    answer = random.sample(ent_in[answer][r], 1)[0]
            if query_structure[0] == 'e':
                query_structure[0] = answer
            else:
                return fill_query(query_structure[0], ent_in, ent_out, answer, ent2id, rel2id, is_time,
                                  is_time_interval)
        else:
            same_structure = defaultdict(list)
            for i in range(len(query_structure)):
                same_structure[list2tuple(query_structure[i])].append(i)
            for i in range(len(query_structure)):
                if len(query_structure[i]) == 1 and query_structure[i][0] == 'u':
                    assert i == len(query_structure) - 1
                    query_structure[i][0] = -1
                    continue
                broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, ent2id, rel2id, is_time,
                                  is_time_interval)
                if broken_flag:
                    return True
            for structure in same_structure:
                if len(same_structure[structure]) != 1:
                    structure_set = set()
                    for i in same_structure[structure]:
                        structure_set.add(list2tuple(query_structure[i]))
                    if len(structure_set) < len(same_structure[structure]):
                        return True


def achieve_answer(query, ent_in, ent_out, is_time=False, is_time_interval=False):
    if is_time:
        assert type(query[-1]) == list
        all_relation_flag = True
        for ele in query[-1]:
            if type(ele) == tuple:
                continue
            if (type(ele) != int) or (ele == -1):   # -1 表示 u 并运算
                all_relation_flag = False
                break
        if all_relation_flag:
            if type(query[0]) == int:
                ent_set = set([query[0]])
            else:
                ent_set = achieve_answer(query[0], ent_in, ent_out, is_time, is_time_interval)
            for i in range(len(query[-1])):
                if query[-1][i] == -2:  # -2 表示减
                    # ent_set = set(range(len(ent_in))) - ent_set
                    ent_set = set(ent_in.keys()) - ent_set
                    # ent_set3 = set(range(len(ent_in))) - set(ent_in.keys())
                elif type(query[-1][i]) == tuple:
                    continue
                else:
                    ent_set_traverse = set()
                    for ent in ent_set:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]][query[-1][i+1]])
                    ent_set = ent_set_traverse
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out, is_time, is_time_interval)
            union_flag = False
            if len(query[-1]) == 1 and query[-1][0] == -1:
                union_flag = True
            for i in range(1, len(query)):
                if not union_flag:
                    ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out, is_time, is_time_interval))
                else:
                    if i == len(query) - 1:
                        continue
                    ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out, is_time, is_time_interval))
        return ent_set
    else:
        assert type(query[-1]) == list
        all_relation_flag = True
        for ele in query[-1]:
            if (type(ele) != int) or (ele == -1):
                all_relation_flag = False
                break
        if all_relation_flag:
            if type(query[0]) == int:
                ent_set = set([query[0]])
            else:
                ent_set = achieve_answer(query[0], ent_in, ent_out)
            for i in range(len(query[-1])):
                if query[-1][i] == -2:
                    ent_set = set(range(len(ent_in))) - ent_set
                else:
                    ent_set_traverse = set()
                    for ent in ent_set:
                        ent_set_traverse = ent_set_traverse.union(ent_out[ent][query[-1][i]])
                    ent_set = ent_set_traverse
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out)
            union_flag = False
            if len(query[-1]) == 1 and query[-1][0] == -1:
                union_flag = True
            for i in range(1, len(query)):
                if not union_flag:
                    ent_set = ent_set.intersection(achieve_answer(query[i], ent_in, ent_out))
                else:
                    if i == len(query) - 1:
                        continue
                    ent_set = ent_set.union(achieve_answer(query[i], ent_in, ent_out))
        return ent_set


@click.command()
@click.option('--dataset', default="icews18_small")
@click.option('--seed', default=0)
@click.option('--gen_train_num', default=0)
@click.option('--gen_valid_num', default=0)
@click.option('--gen_test_num', default=0)
@click.option('--max_ans_num', default=1e6)
@click.option('--reindex', is_flag=True, default=False)
@click.option('--gen_train', is_flag=True, default=True)
@click.option('--gen_valid', is_flag=True, default=False)
@click.option('--gen_test', is_flag=True, default=False)
@click.option('--gen_id', default=6)
@click.option('--save_name', is_flag=True, default=True)
@click.option('--index_only', is_flag=True, default=False)
@click.option('--is_time', is_flag=True, default=True)
@click.option('--is_time_interval', is_flag=True, default=True)
@click.option('--is_time_convert', is_flag=True, default=True)
def main(dataset, seed, gen_train_num, gen_valid_num, gen_test_num, max_ans_num, reindex, gen_train, gen_valid,
         gen_test, gen_id, save_name, index_only, is_time, is_time_interval, is_time_convert):
    train_num_dict = {'FB15k': 273710, "FB15k-237": 149689, "NELL": 107982, 'icews18': 290589}
    valid_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000, 'icews18': 8000}
    test_num_dict = {'FB15k': 8000, "FB15k-237": 5000, "NELL": 4000, 'icews18': 8000}
    if gen_train and gen_train_num == 0:
        if 'FB15k-237' in dataset:
            gen_train_num = 149689
        elif 'FB15k' in dataset:
            gen_train_num = 273710
        elif 'NELL' in dataset:
            gen_train_num = 107982
        elif 'icews18' in dataset:
            gen_train_num = 260206
        elif 'icews18_small' in dataset:
            gen_train_num = 1000
            # gen_train_num = 290589
            # gen_train_num = 349689
        else:
            gen_train_num = train_num_dict[dataset]
    if gen_valid and gen_valid_num == 0:
        if 'FB15k-237' in dataset:
            gen_valid_num = 5000
        elif 'FB15k' in dataset:
            gen_valid_num = 8000
        elif 'NELL' in dataset:
            gen_valid_num = 4000
        elif 'icews18' in dataset:
            gen_valid_num = 8000
        elif 'icews18_small' in dataset:
            gen_valid_num = 800
        else:
            gen_valid_num = valid_num_dict[dataset]
    if gen_test and gen_test_num == 0:
        if 'FB15k-237' in dataset:
            gen_test_num = 5000
        elif 'FB15k' in dataset:
            gen_test_num = 8000
        elif 'NELL' in dataset:
            gen_test_num = 4000
        elif 'icews18' in dataset:
            gen_test_num = 8000
        elif 'icews18_small' in dataset:
            gen_test_num = 800
        else:
            gen_test_num = test_num_dict[dataset]
    if index_only:
        index_dataset(dataset, reindex)
        exit(-1)

    e = 'e'
    r = 'r'
    n = 'n'
    u = 'u'
    t = 't'
    st = 'st'
    et = 'et'
    if is_time:
        if is_time_interval:
            # query_structures = [
            #     [e, [[st, et], r]],
            #     [e, [[st, et], r, [st, et], r]],
            #     [e, [[st, et], r, [st, et], r, [st, et], r]],
            #     [[e, [[st, et], r]], [e, [[st, et], r]]],
            #     [[e, [[st, et], r]], [e, [[st, et], r]], [e, [[st, et], r]]],
            #     [[e, [[st, et], r, [st, et], r]], [e, [[st, et], r]]],
            #     [[[e, [[st, et], r]], [e, [[st, et], r]]], [[st, et], r]],
            #     # negation
            #     [[e, [[st, et], r]], [e, [[st, et], r, n]]],
            #     [[e, [[st, et], r]], [e, [[st, et], r]], [e, [[st, et], r, n]]],
            #     [[e, [[st, et], r, [st, et], r]], [e, [[st, et], r, n]]],
            #     [[e, [[st, et], r, [st, et], r, n]], [e, [[st, et], r]]],
            #     [[[e, [[st, et], r]], [e, [[st, et], r, n]]], [[st, et], r]],
            #     # union
            #     [[e, [[st, et], r]], [e, [[st, et], r]], [u]],
            #     [[[e, [[st, et], r]], [e, [[st, et], r]], [u]], [[st, et], r]]
            # ]
            query_structures = [
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
            query_names = ['1pts', '2pts', '3pts', '2its', '3its', 'pits', 'ipts', '2ints', '3ints', 'pints', 'pnits',
                           'inpts', '2uts', 'upts']
        else:
            query_structures = [
                [e, [r, [t]]],
                [e, [r, [t], r, [t]]],
                [e, [r, [t], r, [t], r, [t]]],
                [[e, [r, [t]]], [e, [r, [t]]]],
                [[e, [r, [t]]], [e, [r, [t]]], [e, [r, [t]]]],
                [[e, [r, [t], r, [t]]], [e, [r, [t]]]],
                [[[e, [r, [t]]], [e, [r, [t]]]], [r, [t]]],
                # negation
                [[e, [r, [t]]], [e, [r, [t], n]]],
                [[e, [r, [t]]], [e, [r, [t]]], [e, [r, [t], n]]],
                [[e, [r, [t], r, [t]]], [e, [r, [t], n]]],
                [[e, [r, [t], r, [t], n]], [e, [r, [t]]]],
                [[[e, [r, [t]]], [e, [r, [t], n]]], [r, [t]]],
                # union
                [[e, [r, [t]]], [e, [r, [t]]], [u]],
                [[[e, [r, [t]]], [e, [r, [t]]], [u]], [r, [t]]]
            ]
            query_names = ['1pt', '2pt', '3pt', '2it', '3it', 'pit', 'ipt', '2int', '3int', 'pint', 'pnit', 'inpt',
                           '2ut', 'upt']
    else:
        query_structures = [
            [e, [r]],
            [e, [r, r]],
            [e, [r, r, r]],
            [[e, [r]], [e, [r]]],
            [[e, [r]], [e, [r]], [e, [r]]],
            [[e, [r, r]], [e, [r]]],
            [[[e, [r]], [e, [r]]], [r]],
            # negation
            [[e, [r]], [e, [r, n]]],
            [[e, [r]], [e, [r]], [e, [r, n]]],
            [[e, [r, r]], [e, [r, n]]],
            [[e, [r, r, n]], [e, [r]]],
            [[[e, [r]], [e, [r, n]]], [r]],
            # union
            [[e, [r]], [e, [r]], [u]],
            [[[e, [r]], [e, [r]], [u]], [r]]
        ]
        query_names = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2in', '3in', 'pin', 'pni', 'inp', '2u', 'up']

    generate_queries(dataset, query_structures[gen_id:gen_id + 1], [gen_train_num, gen_valid_num, gen_test_num],
                     max_ans_num, gen_train, gen_valid, gen_test, query_names[gen_id:gen_id + 1], save_name, is_time,
                     is_time_interval, is_time_convert)


if __name__ == '__main__':
    main()
