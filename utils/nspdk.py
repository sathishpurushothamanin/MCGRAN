import os
import numpy as np
import networkx as nx


#nas-101 related imports
import tensorflow as tf
from scipy.sparse import csr_matrix
#from nasbench.lib import model_metrics_pb2
#import base64
import json

from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize



if __name__ == '__main__':
    data_dir = 'data/NAS-101'
    filename = 'nasbench_1000_80_7.tfrecord'
    graphs = get_data(data_dir, filename, max_num_nodes=7)
    train_graphs = graphs[:500]
    test_graphs = graphs[500:]
    print(len(graphs))
    print(graphs[0].nodes.data())
    print(graphs[0].edges.data())

    print('MMD distance of NSPDK ', compute_mmd_nspdk(train_graphs, test_graphs, compute_pairwise_kernel))