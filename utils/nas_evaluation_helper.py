import networkx as nx
import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle

from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize

def display_embedding(hidden_features, graph_label):

    # pca = PCA(n_components=len(graph_label))
    pca = PCA(n_components=50)
    # print(hidden_features)
    # new_shape = (hidden_features.shape[0], hidden_features.shape[2])
    # hidden_features = np.reshape(hidden_features, new_shape)
    # print(hidden_features.shape)
    pca_result = pca.fit_transform(hidden_features)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
    ##Variance PCA: 0.993621154832802

    #Run T-SNE on the PCA features.
    tsne = TSNE(n_components=2, verbose = 1)
    tsne_results = tsne.fit_transform(pca_result)


    y_test_cat = F.one_hot(torch.Tensor(graph_label).long(), num_classes = 2).cpu().numpy()
    print(y_test_cat)
    color_map = np.argmax(y_test_cat, axis=1)
    print(color_map)
    plt.figure(figsize=(10,10))
    for cl in range(2):
        indices = np.where(color_map==cl)
        indices = indices[0]
        print(indices)
        print(tsne_results[indices,0], tsne_results[indices, 1])
        plt.scatter(tsne_results[indices,0], tsne_results[indices, 1], label=cl)
    plt.legend()

    pickle.dump(tsne_results, open('tsne_results.p', 'wb'))
    # print(tsne_results)
    # plt.show()
    #
    # plt.figure(figsize=(10,10))
    # plt.scatter(tsne_results[:,0], tsne_results[:, 1])
    # plt.legend()
    plt.savefig('tsne_graph_embedding.png')
    # plt.show()


def validate_nn_structure(graphs_nas):
    num_invalid_graphs = 0

    for graph in graphs_nas:
        graph = nx.to_numpy_matrix(graph)
        num_vertices = np.shape(graph)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
          top = frontier.pop()
          for v in range(top + 1, num_vertices):
            if graph[top, v] and v not in visited_from_input:
              visited_from_input.add(v)
              frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
          top = frontier.pop()
          for v in range(0, top):
            if graph[v, top] and v not in visited_from_output:
              visited_from_output.add(v)
              frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        if len(extraneous) > 0:
            num_invalid_graphs = num_invalid_graphs + 1

    return num_invalid_graphs

def validate_self_loops(graphs_nas):
    num_graphs_with_self_loops = 0
    for graph in graphs_nas:
        num_graphs_with_self_loops += 1 if nx.number_of_selfloops(graph) > 0 else 0
    return num_graphs_with_self_loops

def validate_isolated_nodes(graphs_nas):
    num_graphs_with_isolates = 0
    for graph in graphs_nas:
        num_graphs_with_isolates += 1 if nx.number_of_isolates(graph) > 0 else 0

    return num_graphs_with_isolates

def validate_uniqueness(graphs_nas):
    sum_minimum_graph_edit_distance = list()
    for i in range(len(graphs_nas)):
        for j in range(i+1, len(graphs_nas)):
            sum_minimum_graph_edit_distance.append(int(min(
            nx.optimize_graph_edit_distance(graphs_nas[i],
            graphs_nas[j]))))

    freq_distribution = dict()
    for item in sum_minimum_graph_edit_distance:
        if item in freq_distribution.keys():
            freq_distribution[item] += 1
        else:
            freq_distribution[item] = 1

    total_min_graph_edit_distance = sum(freq_distribution.values())
    normalized_freq_distribution = dict()
    for key in freq_distribution:
        normalized_freq_distribution[key] = freq_distribution[key]/total_min_graph_edit_distance*100
    return normalized_freq_distribution

def validate_novelty(training_graphs, graphs_nas):
    sum_minimum_graph_edit_distance = list()
    for i in range(len(training_graphs)):
        for j in range(len(graphs_nas)):
            sum_minimum_graph_edit_distance.append(int(min(nx.optimize_graph_edit_distance(training_graphs[i], graphs_nas[j]))))

    freq_distribution = dict()
    for item in sum_minimum_graph_edit_distance:
        if item in freq_distribution.keys():
            freq_distribution[item] += 1
        else:
            freq_distribution[item] = 1

    total_min_graph_edit_distance = sum(freq_distribution.values())
    normalized_freq_distribution = dict()
    for key in freq_distribution:
        normalized_freq_distribution[key] = freq_distribution[key]/total_min_graph_edit_distance*100

    return normalized_freq_distribution


def compute_pairwise_kernel(X, Y=None):
    X = vectorize(X, complexity=4, discrete=True)

    if Y is not None:
        Y = vectorize(Y, complexity=4, discrete=True)
    
    return pairwise_kernels(X, Y, metric='linear')

def compute_mmd_nspdk(train_graphs, pred_graphs=None, compute_pairwise_kernel=None):
    actual = compute_pairwise_kernel(train_graphs)
    pred = compute_pairwise_kernel(pred_graphs)
    actual_pred = compute_pairwise_kernel(train_graphs, Y=pred_graphs)

    mmd =  np.average(actual) + np.average(pred) - 2 * np.average(actual_pred)
    return mmd

def add_labels(graph, node_label, edge_label='-'):
    for node_idx in range(graph.number_of_nodes()):
        graph.add_node(node_idx,
            label=str(node_label[node_idx]))
        labels = '-'
        nx.set_edge_attributes(graph, labels, "label")
    return graph