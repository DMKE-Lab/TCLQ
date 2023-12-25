import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import data, utils
from .TKGDataset import TemporalKnowledgeGraphDataset
from torchdrug.layers import functional
from torchdrug.core import Registry as R

# from .data import Query
from .data_temporal import TemporalQuery


class LogicalQueryDataset(TemporalKnowledgeGraphDataset):
    """Logical query dataset."""
    struct2type = {
        ("e", ("r", ("st", "et"))): "1pts",
        ("e", ("r", ("st", "et"), "r", ("st", "et"))): "2pts",
        ("e", ("r", ("st", "et"), "r", ("st", "et"), "r", ("st", "et"))): "3pts",
        (("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"),))): "2its",
        (("e", ("r", ("st", "et"))), ("e", ("r", ("st", "et"))), ("e", ("r", ("st", "et")))): "3its",
        (("e", ("r", ("st", "et"), "r", ("st", "et"),)), ("e", ("r", ("st", "et"),))): "pits",
        ((("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"),))), ("r", ("st", "et"),)): "ipts",
        (("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"), "n"))): "2ints",
        (("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"), "n"))): "3ints",
        (("e", ("r", ("st", "et"), "r", ("st", "et"),)), ("e", ("r", ("st", "et"), "n"))): "pints",
        (("e", ("r", ("st", "et"), "r", ("st", "et"), "n")), ("e", ("r", ("st", "et"),))): "pnits",
        ((("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"), "n"))), ("r", ("st", "et"),)): "inpts",
        (("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"),)), ("u",)): "2uts-DNF",
        ((("e", ("r", ("st", "et"),)), ("e", ("r", ("st", "et"),)), ("u",)), ("r", ("st", "et"),)): "upts-DNF",
    }

    def load_pickle(self, path, query_types=None, union_type="DNF", verbose=0):
        """
        Load the dataset from pickle dumps (BetaE format).

        Parameters:
            path (str): path to pickle dumps
            query_types (list of str, optional): query types to load.
                By default, load all query types.
            union_type (str, optional): which union type to use, ``DNF`` or ``DM``
            verbose (int, optional): output verbose level
        """
        query_types = query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, union_type)
                elif query_type[query_type.find("-") + 1:] != union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}

        with open(os.path.join(path, "id2ent.pkl"), "rb") as fin:
            entity_vocab = pickle.load(fin)
        with open(os.path.join(path, "id2rel.pkl"), "rb") as fin:
            relation_vocab = pickle.load(fin)
        with open(os.path.join(path, "id2ts.pkl"), "rb") as fin:
            timestamp_vocab = pickle.load(fin)


        quadruples = []
        num_samples = []
        for split in ["train_indexified_tid", "valid_indexified_tid", "test_indexified_tid"]:
            quadruple_file = os.path.join(path, "%s.txt" % split)
            with open(quadruple_file) as fin:
                if verbose:
                    fin = tqdm(fin, "Loading %s" % quadruple_file, utils.get_line_count(quadruple_file))
                num_sample = 0
                for line in fin:
                    e1, rel, e2, t = [int(x) for x in line.split()]
                    quadruples.append((e1, e2, rel, t))
                    num_sample += 1
                num_samples.append(num_sample)
        self.load_quadruples(quadruples, entity_vocab=entity_vocab, relation_vocab=relation_vocab,
                             timestamp_vocab=timestamp_vocab)
        fact_mask = torch.arange(num_samples[0])
        # self.graph: is the full graph without missing edges 是没有缺边的完整图
        # self.fact_graph: is the training graph 是训练图
        self.fact_graph = self.temporal_graph.edge_mask(fact_mask)
        # self.fact_graph2 = self.tgraph.edge_mask(fact_mask)

        "******************************************"
        queries = []
        types = []
        easy_answers = []
        hard_answers = []
        num_samples = []
        max_query_length = 0

        for split in ["train", "valid", "test"]:
            if verbose:
                pbar = tqdm(desc="Loading %s-*.pkl" % split, total=3)
            with open(os.path.join(path, "%s-queries-tid.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            if verbose:
                pbar.update(1)
            type2queries = {self.struct2type[k]: v for k, v in struct2queries.items()}
            type2queries = {k: v for k, v in type2queries.items() if k in self.type2id}
            if split == "train":
                with open(os.path.join(path, "%s-answers-tid.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                query2hard_answers = defaultdict(set)
                if verbose:
                    pbar.update(2)
            else:
                with open(os.path.join(path, "%s-easy-answers-tid.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)
                with open(os.path.join(path, "%s-hard-answers-tid.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
                if verbose:
                    pbar.update(1)
            num_sample = sum([len(q) for t, q in type2queries.items()])
            if verbose:
                pbar = tqdm(desc="Processing %s queries" % split, total=num_sample)
            for type in type2queries:
                struct_queries = sorted(type2queries[type])
                for query in struct_queries:
                    easy_answers.append(query2easy_answers[query])
                    hard_answers.append(query2hard_answers[query])
                    """
                    用到data中的Query
                    从嵌套元组（BetaE格式）构造一个逻辑查询。
                    """
                    if type == '3ints':
                        x = 1
                    if type == 'pints':
                        x = 1
                    if type == 'inpts':
                        x = 1
                    if type == '2uts':
                        x = 1
                    if type == 'upts':
                        x = 1
                    query = TemporalQuery.from_nested(query)
                    queries.append(query)
                    max_query_length = max(max_query_length, len(query))
                    types.append(self.type2id[type])
                    if verbose:
                        pbar.update(1)
            num_samples.append(num_sample)
        self.queries = queries
        self.types = types
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        self.num_samples = num_samples
        self.max_query_length = max_query_length

        # self.num_quadruple = len(queries)

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": functional.as_mask(easy_answer, self.num_entity),
            "hard_answer": functional.as_mask(hard_answer, self.num_entity),
        }

    def __len__(self):
        return len(self.queries)

    def __repr__(self):
        lines = [
            "#entity: %d" % self.num_entity,
            "#relation: %d" % self.num_relation,
            "#timestamp: %d" % self.num_timestamp,
            "#quadruple: %d" % self.num_quadruple,
            "#query: %d" % len(self.queries),
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits

@R.register("dataset_temporal.ICEWS18LogicalQuery")
class ICEWS18LogicalQuery(LogicalQueryDataset):

    # url = "http://snap.stanford.edu/betae/KG_data.zip"
    # md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        # path = '/data/xls/emb_code/TCLQ/kg-datasets'
        path = 'E:\A_Study\TCLQ\kg-datasets'
        self.path = path
        print("path:", path)

        # zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "icews18", "data")

        self.load_pickle(path, query_types, union_type, verbose=verbose)

@R.register("dataset_temporal.FB15kLogicalQuery")
class FB15kLogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)

        # path = '/data/xls/emb_code/GNN-QE-master/kg-datasets'
        path = 'E:\A_Study\Experiment\GNN-QE-master\kg-datasets'
        self.path = path
        print("path:", path)

        # zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "FB15k-betae")
        # if not os.path.exists(path):
        #     utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)


@R.register("dataset_temporal.FB15k237LogicalQuery")
class FB15k237LogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        # path = '/data/xls/emb_code/GNN-QE-master/kg-datasets'
        path = 'E:\A_Study\Experiment\GNN-QE-master\kg-datasets'
        self.path = path
        print("path:", path)

        # zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "FB15k-237-betae")

        self.load_pickle(path, query_types, union_type, verbose=verbose)


@R.register("dataset_temporal.NELL995LogicalQuery")
class NELL995LogicalQuery(LogicalQueryDataset):

    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"

    def __init__(self, path, query_types=None, union_type="DNF", verbose=1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5=self.md5)
        path = os.path.join(path, "NELL-betae")
        if not os.path.exists(path):
            utils.extract(zip_file)

        self.load_pickle(path, query_types, union_type, verbose=verbose)
