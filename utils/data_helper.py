###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F

#nas-101 related imports
import tensorflow as tf
from scipy.sparse import csr_matrix
from nasbench.lib import model_metrics_pb2
import base64
import json
#import dgl

__all__ = [
    'save_graph_list', 'load_graph_list',
    'preprocess_graph_list', 'create_graphs'
]


# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G


def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  # print('Loading graph dataset: ' + str(name))
  # G = nx.Graph()
  # # load data
  # path = os.path.join(data_dir, name)
  # data_adj = np.loadtxt(
      # os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  # if node_attributes:
    # data_node_att = np.loadtxt(
        # os.path.join(path, '{}_node_attributes.txt'.format(name)),
        # delimiter=',')
  # data_node_label = np.loadtxt(
      # os.path.join(path, '{}_node_labels.txt'.format(name)),
      # delimiter=',').astype(int)
  # data_graph_indicator = np.loadtxt(
      # os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      # delimiter=',').astype(int)
  # if graph_labels:
    # data_graph_labels = np.loadtxt(
        # os.path.join(path, '{}_graph_labels.txt'.format(name)),
        # delimiter=',').astype(int)

  # data_tuple = list(map(tuple, data_adj))
  # # print(len(data_tuple))
  # # print(data_tuple[0])

  # # add edges
  # G.add_edges_from(data_tuple)
  # # add node attributes
  # for i in range(data_node_label.shape[0]):
    # if node_attributes:
      # G.add_node(i + 1, feature=data_node_att[i])
    # G.add_node(i + 1, label=int(data_node_label[i])-1)
  # G.remove_nodes_from(list(nx.isolates(G)))

  # # remove self-loop
  # G.remove_edges_from(nx.selfloop_edges(G))

  # # print(G.number_of_nodes())
  # # print(G.number_of_edges())

  # # split into graphs
  # graph_num = data_graph_indicator.max()
  # node_list = np.arange(data_graph_indicator.shape[0]) + 1
  # graphs = []
  # max_nodes = 0
  # for i in range(graph_num):
    # # find the nodes for each graph
    # nodes = node_list[data_graph_indicator == i + 1]
    # G_sub = G.subgraph(nodes)
    # if graph_labels:
      # G_sub.graph['label'] = int(data_graph_labels[i]) - 1  #index start from 0 for easier processing
    # # print('nodes', G_sub.number_of_nodes())
    # # print('edges', G_sub.number_of_edges())
    # # print('label', G_sub.graph)
    # graphs.append(G_sub)
    # # if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    # # ) <= max_num_nodes:
      # # graphs.append(G_sub)
      # # if G_sub.number_of_nodes() > max_nodes:
        # # max_nodes = G_sub.number_of_nodes()
      # # print(G_sub.number_of_nodes(), 'i', i)
      # # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  # print('Loaded')
  graphs = list()
  dataset = dgl.data.TUDataset('DD')
    
  for data, label in dataset:
    # print(data.adj())
    # print(type(data.ndata['node_labels']))
    # for item in data.ndata['node_labels']:
        # print(item)
    data.ndata['label'] = data.ndata['node_labels']
    # print(data.ndata)
    # print(data.edata)
    # print(label)
    
    # print('DGL to networkx')
    graph = dgl.to_networkx(data, node_attrs=['label'])
    graph.graph['label'] = label.tolist()[0]
    if data.ndata['node_labels'].shape[0] >= min_num_nodes and data.ndata['node_labels'].shape[0] <= max_num_nodes:
        graphs.append(nx.Graph(graph))
    # print(graph.graph['label'])
    # print(graph.nodes(data=True))
    # print(graph.edges(data=True))
  
  return graphs


def create_graphs(graph_type, data_dir='data', max_num_samples=1000, min_test_accuracy=80, max_test_accuracy=100, max_num_nodes=7):
  # npr = np.random.RandomState(seed)
  ### load datasets
#  data_dir = '/content/drive/MyDrive/GRAN-NEW/GRAN-NAS-Pipeline/data/NAS-101'

  graphs = []
  filename = f'nasbench_{max_num_samples}_{min_test_accuracy}_{max_test_accuracy}_{max_num_nodes}.tfrecord'
  if graph_type == 'nas':
      #add node labels to the dataset
        nas_dataset = list()
        graphs = []
        tf.compat.v1.enable_eager_execution()
        count = 1
#        print(os.path.join(data_dir, filename), os.path.exists(os.path.join(data_dir, filename)))
        if os.path.exists(os.path.join(data_dir, filename)):
            dataset = tf.data.TFRecordDataset(os.path.join(data_dir, filename))
            operations = {'input':0, 'conv3x3-bn-relu':1, 'conv1x1-bn-relu':2, 'maxpool3x3':3, 'output':4}
            for serialized_row in dataset:
              module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = json.loads(serialized_row.numpy().decode('utf-8'))
              metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))
              dim = int(np.sqrt(len(raw_adjacency)))
              adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
              adjacency = np.reshape(adjacency, (dim, dim))
              graph = nx.Graph(csr_matrix(adjacency))
              graph_ops = raw_operations.split(',')
              for node_idx in range(graph.number_of_nodes()):
                  graph.add_node(node_idx,
                   label=operations[graph_ops[node_idx]])
#              print(graph.number_of_nodes())
              if graph.number_of_nodes() == max_num_nodes:
                  data_item = dict()
                  data_item['graph'] = graph
                  data_item['total_parameters'] = metrics.trainable_parameters
                  data_item['total_training_time'] = metrics.evaluation_data[2].training_time
                  data_item['test_accuracy'] = metrics.evaluation_data[2].test_accuracy
                  data_item['hash'] = module_hash
#                  edges = dict()
#                  graph = nx.DiGraph(csr_matrix(adjacency))
#                  for node_idx in range(max_num_nodes):
#                    node_incoming_edge = list(graph.predecessors(node_idx))
#                    if len(node_incoming_edge) == 0:
#                        edges[node_idx] = 0
#                    else:
#                        edges[node_idx] = node_incoming_edge
#                  data_item['node_in_edges'] = edges
                  nas_dataset.append(data_item)

        else:
          dataset = tf.compat.v1.python_io.tf_record_iterator(os.path.join(data_dir, 'nasbench_only108.tfrecord'))
          operations = {'input':0, 'conv3x3-bn-relu':1, 'conv1x1-bn-relu':2, 'maxpool3x3':3, 'output':4}
          min_test_accuracy = min_test_accuracy/100.0
          max_test_accuracy = max_test_accuracy/100.0
          with tf.io.TFRecordWriter(os.path.join(data_dir, filename)) as writer:
              for serialized_row in dataset:
                  module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = json.loads(serialized_row.decode('utf-8'))
                  metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))
                  if (metrics.evaluation_data[2].test_accuracy > min_test_accuracy
                    and metrics.evaluation_data[2].test_accuracy <= max_test_accuracy):
                      writer.write(serialized_row)
                      dim = int(np.sqrt(len(raw_adjacency)))
                      adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
                      adjacency = np.reshape(adjacency, (dim, dim))
                      graph = nx.Graph(csr_matrix(adjacency))
                      graph_ops = raw_operations.split(',')

                      for node_idx in range(graph.number_of_nodes()):
                          graph.add_node(node_idx,
                           label=operations[graph_ops[node_idx]])

                      if graph.number_of_nodes() == max_num_nodes:
                          data_item = dict()
                          data_item['graph'] = graph
                          data_item['total_parameters'] = metrics.trainable_parameters
                          data_item['total_training_time'] = metrics.evaluation_data[2].training_time
                          data_item['test_accuracy'] = metrics.evaluation_data[2].test_accuracy
                          data_item['hash'] = module_hash
#                          edges = dict()
#                          graph = nx.DiGraph(csr_matrix(adjacency))
#                          for node_idx in range(max_num_nodes):
#                            node_incoming_edge = list(graph.predecessors(node_idx))
#                            if len(node_incoming_edge) == 0:
#                                edges[node_idx] = 0
#                            else:
#                                edges[node_idx] = node_incoming_edge
#                          data_item['node_in_edges'] = edges
                          nas_dataset.append(data_item)
                          count += 1

                      if count > max_num_samples:
                          break

        graphs = nas_dataset             
  elif graph_type == 'DD':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
  elif graph_type == 'NWSG':

    dataset = list()
    for j in range(2, max_num_nodes):
      for i in np.linspace(0,1,100):
#        graph = nx.newman_watts_strogatz_graph(max_num_nodes, j, i, seed=1234)
        graph = nx.watts_strogatz_graph(max_num_nodes, j, i, seed=1234)
        data_item = dict()
        data_item['graph'] = graph
        data_item['k'] = j/(max_num_nodes*1.0) #normalization
        data_item['p'] = i
        dataset.append(data_item)

    graphs = dataset
    
  elif graph_type == 'EBAG':
    dataset = list()
    for rand_seed in range(20):
        for j in range(2, max_num_nodes):
            graph = nx.barabasi_albert_graph(max_num_nodes, j, seed=rand_seed)
            data_item = dict()
            data_item['type'] = [0, 1, 0, 0]
            data_item['graph'] = graph
            data_item['m'] = j/(max_num_nodes*1.0)  #normalization
            data_item['p'] = 0
            data_item['q'] = 0
            dataset.append(data_item)
    graphs = dataset#[:1000]

  elif graph_type == 'PCG':
    dataset = list()
    for j in range(2, max_num_nodes):
        for i in np.linspace(0,1,100):
            graph = nx.powerlaw_cluster_graph(max_num_nodes, j, i, seed=1234)
            data_item = dict()
            data_item['graph'] = graph
            data_item['k'] = j/(max_num_nodes*1.0)  #normalization
            data_item['p'] = i
            dataset.append(data_item)
            
    graphs = dataset

  elif graph_type == 'ERG':
    dataset = list()
    for k in np.linspace(0,1,1000):
        graph = nx.gnp_random_graph(max_num_nodes, k, seed=1234)
        data_item = dict()
        data_item['graph'] = graph
        data_item['k'] = 0 #normalization
        data_item['p'] = k
        dataset.append(data_item)
        
    graphs = dataset
  elif graph_type == 'ALL':
    dataset = list()
    for k in np.linspace(0,1,1000):
        graph = nx.gnp_random_graph(max_num_nodes, k, seed=1234)
        data_item = dict()
        data_item['type'] = [0, 0, 0, 1]
        data_item['graph'] = graph
        data_item['m'] = 0 #normalization
        data_item['p'] = k
#        data_item['q'] = 0
        dataset.append(data_item)

    for j in range(2, max_num_nodes):
        for i in np.linspace(0,1,100):
            graph = nx.powerlaw_cluster_graph(max_num_nodes, j, i, seed=1234)
            data_item = dict()
            data_item['type'] = [0, 0, 1, 0]
            data_item['graph'] = graph
            data_item['m'] = j/(max_num_nodes*1.0) #normalization
            data_item['p'] = i
#            data_item['q'] = 0
            dataset.append(data_item)

    for rand_seed in np.random.randint(1, 10000000, size=20).tolist():
        for j in range(2, max_num_nodes):
            graph = nx.barabasi_albert_graph(max_num_nodes, j, seed=rand_seed)
            data_item = dict()
            data_item['type'] = [0, 1, 0, 0]
            data_item['graph'] = graph
            data_item['m'] = j/(max_num_nodes*1.0)  #normalization
            data_item['p'] = 0
#            data_item['q'] = 0
            dataset.append(data_item)
            
    for j in range(2, max_num_nodes):
      for i in np.linspace(0,1,100):
        graph = nx.newman_watts_strogatz_graph(max_num_nodes, j, i, seed=1234)
        data_item = dict()
        data_item['type'] = [1, 0, 0, 0]
        data_item['graph'] = graph
        data_item['m'] = j/(max_num_nodes*1.0) #normalization
        data_item['p'] = i
#        data_item['q'] = 0
        dataset.append(data_item)

    npr = np.random.RandomState(1234)
    npr.shuffle(dataset)
    graphs = dataset#[:10000]
    # args.max_prev_node = 230
#  num_nodes = [gg.number_of_nodes() for gg in graphs]
#  num_edges = [gg.number_of_edges() for gg in graphs]
#  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
#  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))

  return graphs