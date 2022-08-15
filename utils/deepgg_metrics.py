# -*- coding: utf-8 -*-
"""
Source: https://github.com/innvariant/deepgg
Author: Julian Stier
"""

import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy

def count_graphs_degree(graph_list: list):
    degree_count = collections.Counter()

    for graph in graph_list:
        degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
        degree_count.update(degree_sequence)
    return degree_count

#def count_cs_degree(ds_construction_sequences: list):
#    return count_graphs_degree([construction_sequence_to_graph(seq) for seq in ds_construction_sequences])

def graph_degree_counter_to_histogram(counter_degree: collections.Counter, degrees_sorted: list=None):
    if degrees_sorted is None:
        degrees_sorted = sorted([d for d in sorted(counter_degree)])
        
    return [counter_degree.get(d) if counter_degree.get(d) is not None else 0 for d in degrees_sorted]

def get_degree_histogram(graph_list: list, degrees_sorted: list=None):
    counter_degree = count_graphs_degree(graph_list)

    return graph_degree_counter_to_histogram(counter_degree(graph_list), degrees_sorted)

def plot_histogram(count: collections.Counter, title='', label=''):
    deg, cnt = zip(*count.items())
    fig, ax = plt.subplots()
    plt.figure(figsize= (10, 10))
    plt.bar(deg, cnt, width=0.8, color='b')
    plt.title('Histogram %s' % title)
    plt.ylabel('Count')
    plt.xlabel(label)
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

def kl_divergence(p, q):    
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(q != 0, p * np.log(p / q), 0))

def compute_avg_shortest_path(graphs: list):
    return np.array([nx.average_shortest_path_length(G.subgraph(max(nx.connected_components(G), key=len))) for G in graphs if len(G) > 0])


def display_boxplot_of_distributions(actual_dist, pred_dist, graph_type=None):
    sns.set(style="whitegrid")
    sns.set(font_scale=2)
    
    distribution_dataset, distribution_generated = actual_dist, pred_dist
    
    plt.figure(figsize=(20, 1))
    ax = sns.boxplot(distribution_dataset)
    ax.set_title('{name}'.format(name=graph_type[0]))
    ax.set(xlim=(1, 4))
    plt.show()
    
    plt.figure(figsize=(20, 1))
    ax = sns.boxplot(distribution_generated)
    ax.set_title('{name}'.format(name=graph_type[1]))
    ax.set(xlim=(1, 4))
    plt.show()


def display_deepgg_metrics(actual_dist, pred_dist):

    num_generated = len(pred_dist)
    num_dataset = len(actual_dist)
    print('Generated graphs:', num_generated)
    print('Dataset graphs:', num_dataset)
    
    pred_degree_dist = count_graphs_degree(pred_dist)
    actual_degree_dist = count_graphs_degree(actual_dist)
#    print(pred_degree_dist)
#    
#    min_x = min(pred_degree_dist)
#    max_x = max(pred_degree_dist)
#
#    pred_degree_dist = [(x - min_x)/(max_x - min_x) for x in pred_degree_dist]
#
#    min_x = min(actual_degree_dist)
#    max_x = max(actual_degree_dist)
#    actual_degree_dist = [(x - min_x)/(max_x - min_x) for x in actual_degree_dist]
    
    plot_histogram(actual_degree_dist)
    plot_histogram(pred_degree_dist)
    
#    degrees = set([d for d in sorted(pred_degree_dist)] + [d for d in sorted(actual_degree_dist)])
#    sorted_degrees = sorted(degrees)
#    hist_list_generated = np.array([pred_degree_dist.get(d) if pred_degree_dist.get(d) is not None else 0 for d in sorted_degrees])
#    hist_list_dataset = np.array([actual_degree_dist.get(d) if actual_degree_dist.get(d) is not None else 0 for d in sorted_degrees])
#    
#    h2 = hist_list_dataset/num_dataset
#    h1 = hist_list_generated/num_generated
#    print(h1)
#    print(h2)
#    print('KL-divergence', kl_divergence(h2, h1))
#    degree_entropy = entropy(pk=h2, qk=h1)
#    print('Entropy', degree_entropy)
#    print('Degree entropy sufficiently small?', degree_entropy < 0.001)