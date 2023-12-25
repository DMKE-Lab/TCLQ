import os
import csv
import math
import lmdb
import pickle
import logging
import warnings
from collections import defaultdict, Sequence

from tqdm import tqdm

import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch.utils import data as torch_data

from torchdrug import core, data, utils
from .TemporalGraph import TemporalGraph
from .TGraph import TGraph


logger = logging.getLogger(__name__)


class TemporalKnowledgeGraphDataset(torch_data.Dataset, core.Configurable):
    """
    Knowledge graph dataset.

    The whole dataset contains one knowledge graph.
    """

    def load_quadruples(self, quadruples, entity_vocab=None, relation_vocab=None, timestamp_vocab=None,
                        inv_entity_vocab=None, inv_relation_vocab=None, inv_timestamp_vocab=None):
        """
        Load the dataset from triplets.
        The mapping between indexes and tokens is specified through either vocabularies or inverse vocabularies.

        Parameters:
            quadruples (array_like): triplets of shape :math:`(n, 4)`
            entity_vocab (dict of str, optional): maps entity indexes to tokens
            relation_vocab (dict of str, optional): maps relation indexes to tokens
            inv_entity_vocab (dict of str, optional): maps tokens to entity indexes
            inv_relation_vocab (dict of str, optional): maps tokens to relation indexes
        """
        entity_vocab, inv_entity_vocab = self._standarize_vocab(entity_vocab, inv_entity_vocab)
        relation_vocab, inv_relation_vocab = self._standarize_vocab(relation_vocab, inv_relation_vocab)
        timestamp_vocab, inv_timestamp_vocab = self._standarize_vocab(timestamp_vocab, inv_timestamp_vocab)

        num_node = len(entity_vocab) if entity_vocab else None
        num_relation = len(relation_vocab) if relation_vocab else None
        num_timestamp = len(timestamp_vocab) if timestamp_vocab else None
        # num_timestamp = (max(timestamp_vocab)-min(timestamp_vocab)+1) if timestamp_vocab else None
        # self.graph = data.Graph(triplets, num_node=num_node, num_relation=num_relation)
        self.temporal_graph = TemporalGraph(quadruples, num_node=num_node, num_relation=num_relation,
                                            num_timestamp=num_timestamp, timestamp_vocab=timestamp_vocab,
                                            tid_vocab=list(inv_timestamp_vocab.values()))
        self.temporal_graph.set_timestamp_vocab(timestamp_vocab)
        self.temporal_graph.set_tid_vocab(list(inv_timestamp_vocab.values()))
        # self.tgraph = TGraph(quadruples, num_node=num_node, num_relation=num_relation,
        #                                     num_timestamp=num_timestamp)
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.timestamp_vocab = timestamp_vocab
        self.inv_entity_vocab = inv_entity_vocab
        self.inv_relation_vocab = inv_relation_vocab
        self.inv_timestamp_vocab = inv_timestamp_vocab
        self.tid_vocab = list(inv_timestamp_vocab.values())

    def load_tsv(self, tsv_file, verbose=0):
        """
        Load the dataset from a tsv file.

        Parameters:
            tsv_file (str): file name
            verbose (int, optional): output verbose level
        """
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        inv_timestamp_vocab = {}
        quadruples = []

        with open(tsv_file, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            if verbose:
                reader = tqdm(reader, "Loading %s" % tsv_file)
            for tokens in reader:
                h_token, r_token, t_token, time_token = tokens
                if h_token not in inv_entity_vocab:
                    inv_entity_vocab[h_token] = len(inv_entity_vocab)
                h = inv_entity_vocab[h_token]
                if r_token not in inv_relation_vocab:
                    inv_relation_vocab[r_token] = len(inv_relation_vocab)
                r = inv_relation_vocab[r_token]
                if t_token not in inv_entity_vocab:
                    inv_entity_vocab[t_token] = len(inv_entity_vocab)
                t = inv_entity_vocab[t_token]
                if time_token not in inv_timestamp_vocab:
                    inv_timestamp_vocab[time_token] = len(inv_timestamp_vocab)
                time = inv_timestamp_vocab[time_token]
                quadruples.append((h, t, r, time))

        self.load_quadruples(quadruples, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab,
                          inv_timestamp_vocab=inv_timestamp_vocab)

    def load_tsvs(self, tsv_files, verbose=0):
        """
        Load the dataset from multiple tsv files.

        Parameters:
            tsv_files (list of str): list of file names
            verbose (int, optional): output verbose level
        """
        inv_entity_vocab = {}
        inv_relation_vocab = {}
        inv_timestamp_vocab = {}
        quadruples = []
        num_samples = []

        for tsv_file in tsv_files:
            with open(tsv_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                if verbose:
                    reader = tqdm(reader, "Loading %s" % tsv_file, utils.get_line_count(tsv_file))

                num_sample = 0
                for tokens in reader:
                    h_token, r_token, t_token, time_token = tokens
                    if h_token not in inv_entity_vocab:
                        inv_entity_vocab[h_token] = len(inv_entity_vocab)
                    h = inv_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_entity_vocab:
                        inv_entity_vocab[t_token] = len(inv_entity_vocab)
                    t = inv_entity_vocab[t_token]
                    if time_token not in inv_timestamp_vocab:
                        inv_timestamp_vocab[time_token] = len(inv_timestamp_vocab)
                    time = inv_timestamp_vocab[time_token]
                    quadruples.append((h, t, r, time))
                    num_sample += 1
            num_samples.append(num_sample)

        self.load_quadruples(quadruples, inv_entity_vocab=inv_entity_vocab, inv_relation_vocab=inv_relation_vocab,
                          inv_timestamp_vocab=inv_timestamp_vocab)
        self.num_samples = num_samples

    def _standarize_vocab(self, vocab, inverse_vocab):
        if vocab is not None:
            if isinstance(vocab, dict):
                assert set(vocab.keys()) == set(range(len(vocab))), "Vocab keys should be consecutive numbers"
                vocab = [vocab[k] for k in range(len(vocab))]
            if inverse_vocab is None:
                inverse_vocab = {v: i for i, v in enumerate(vocab)}
        if inverse_vocab is not None:
            assert set(inverse_vocab.values()) == set(range(len(inverse_vocab))), \
                "Inverse vocab values should be consecutive numbers"
            if vocab is None:
                vocab = sorted(inverse_vocab, key=lambda k: inverse_vocab[k])
        return vocab, inverse_vocab

    @property
    def num_entity(self):
        """Number of entities."""
        return self.temporal_graph.num_node

    @property
    def num_quadruple(self):
        """Number of triplets."""
        return self.temporal_graph.num_edge

    @property
    def num_relation(self):
        """Number of relations."""
        return self.temporal_graph.num_relation

    @property
    def num_timestamp(self):
        """Number of timestamp."""
        return self.temporal_graph.num_timestamp

    def __getitem__(self, index):
        return self.temporal_graph.edge_list[index]

    def __len__(self):
        return self.temporal_graph.num_edge

    def __repr__(self):
        lines = [
            "#entity: %d" % self.num_entity,
            "#relation: %d" % self.num_relation,
            "#triplet: %d" % self.num_quadruple,
            "#timestamp: %d" % self.num_timestamp
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
