# Copyright 2019 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and limitations under the License.

# Modified version of original benchmarks
# https://github.com/google-research/nasbench/blob/master/NASBench.ipynb

# Standard imports
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import networkx as nx
from utils.nas_evaluation_helper import *
from utils.runner_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.nasbench_dataset import *
from nasbench import api

# Use nasbench_full.tfrecord for full dataset (run download command above).
#nasbench = api.NASBench('data/nas-101/nasbench_only108.tfrecord')

# Useful constants
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix,

def create_labeled_graph(matrix, ops):
    operations = {'input':0, 'conv3x3-bn-relu':1, 'conv1x1-bn-relu':2, 'maxpool3x3':3, 'output':4}
    graph = nx.from_numpy_matrix(matrix)
    for node_idx in range(graph.number_of_nodes()):
        graph.add_node(node_idx,
        label=operations[ops[node_idx]])
    nx.set_edge_attributes(graph, '-', "label")
    
    return graph

def random_spec():
  """Returns a random valid spec."""
  searched_graphs = list()
  while True:
    matrix = np.random.choice(ALLOWED_EDGES, size=(NUM_VERTICES, NUM_VERTICES))
    matrix = np.triu(matrix, 1)
    ops = np.random.choice(ALLOWED_OPS, size=(NUM_VERTICES)).tolist()
    ops[0] = INPUT
    ops[-1] = OUTPUT
    spec = api.ModelSpec(matrix=matrix, ops=ops)
    searched_graphs.append(create_labeled_graph(matrix, ops))
    if nasbench.is_valid(spec):
      if len(spec.ops) == 7:
        return spec, searched_graphs

def mutate_spec(old_spec, mutation_rate=1.0):
  """Computes a valid mutated spec from the old_spec."""
  searched_graphs = list()
  while True:
    new_matrix = copy.deepcopy(old_spec.original_matrix)
    new_ops = copy.deepcopy(old_spec.original_ops)

    # In expectation, V edges flipped (note that most end up being pruned).
    edge_mutation_prob = mutation_rate / NUM_VERTICES
    for src in range(0, NUM_VERTICES - 1):
      for dst in range(src + 1, NUM_VERTICES):
        if random.random() < edge_mutation_prob:
          new_matrix[src, dst] = 1 - new_matrix[src, dst]
          
    # In expectation, one op is resampled.
    op_mutation_prob = mutation_rate / OP_SPOTS
    for ind in range(1, NUM_VERTICES - 1):
      if random.random() < op_mutation_prob:
        available = [o for o in nasbench.config['available_ops'] if o != new_ops[ind]]
        new_ops[ind] = random.choice(available)
        
    new_spec = api.ModelSpec(new_matrix, new_ops)
    searched_graphs.append(create_labeled_graph(new_matrix, new_ops))
    if nasbench.is_valid(new_spec):
      if len(new_spec.ops) == 7:
        return new_spec, searched_graphs

def random_combination(iterable, sample_size):
  """Random selection from itertools.combinations(iterable, r)."""
  pool = tuple(iterable)
  n = len(pool)
  indices = sorted(random.sample(range(n), sample_size))
  return tuple(pool[i] for i in indices)

def run_random_search(max_num_searches=100, max_time_budget=5e6):
  """Run a single roll-out of random search to a fixed time budget."""
  nasbench.reset_budget_counters()
  times, best_valids, best_tests = [0.0], [0.0], [0.0]
  
  total_searched_graphs = list()
  valid_graphs_test_accuracy = list()
  while True:
    spec, searched_graphs = random_spec()
    data = nasbench.query(spec)
    total_searched_graphs.extend(searched_graphs)
    # It's important to select models only based on validation accuracy, test
    # accuracy is used only for comparing different search trajectories.
    
    valid_graphs_test_accuracy.append(data['test_accuracy'])
    if data['test_accuracy'] > best_tests[-1]:
      best_valids.append(data['validation_accuracy'])
      best_tests.append(data['test_accuracy'])
    else:
      best_valids.append(best_valids[-1])
      best_tests.append(best_tests[-1])

    time_spent, _ = nasbench.get_budget_counters()
    times.append(time_spent)

    if len(total_searched_graphs) >= 1000:
      break
    if time_spent > max_time_budget:
      # Break the first time we exceed the budget.
      break

  # print(f'{len(searched_graphs)} number of graphs searched in this iteration')
  return times, best_valids, best_tests, total_searched_graphs, valid_graphs_test_accuracy #return also the saved graphs

def run_evolution_search(max_num_searches=100,
                            max_time_budget=5e6,
                            population_size=800,
                            tournament_size=10,
                            mutation_rate=1.0):
  """Run a single roll-out of regularized evolution to a fixed time budget."""
  nasbench.reset_budget_counters()
  times, best_valids, best_tests = [0.0], [0.0], [0.0]
  

  population = []   # (validation, spec) tuples

  # For the first population_size individuals, seed the population with randomly
  # generated cells.
  
  for _ in range(population_size):
    spec, _ = random_spec()
    data = nasbench.query(spec)
    time_spent, _ = nasbench.get_budget_counters()
    times.append(time_spent)
    population.append((data['test_accuracy'], spec))

    if data['test_accuracy'] > best_tests[-1]:
      best_valids.append(data['validation_accuracy'])
      best_tests.append(data['test_accuracy'])
    else:
      best_valids.append(best_valids[-1])
      best_tests.append(best_tests[-1])

    if time_spent > max_time_budget:
      break

  total_searched_graphs = list()
  valid_graphs_test_accuracy = list()
  # After the population is seeded, proceed with evolving the population.
  while True:
    sample = random_combination(population, tournament_size)
    best_spec = sorted(sample, key=lambda i:i[0])[-1][1]
    new_spec, searched_graphs = mutate_spec(best_spec, mutation_rate)
    total_searched_graphs.extend(searched_graphs)
    data = nasbench.query(new_spec)
    time_spent, _ = nasbench.get_budget_counters()
    times.append(time_spent)

    # In regularized evolution, we kill the oldest individual in the population.
    population.append((data['test_accuracy'], new_spec))
    population.pop(0)

    valid_graphs_test_accuracy.append(data['test_accuracy'])
    if data['test_accuracy'] > best_tests[-1]:
      best_valids.append(data['validation_accuracy'])
      best_tests.append(data['test_accuracy'])
    else:
      best_valids.append(best_valids[-1])
      best_tests.append(best_tests[-1])

    if len(total_searched_graphs) >= 1000:
      break
    if time_spent > max_time_budget:
      break

  # print(f'{len(searched_graphs)} number of graphs searched in this iteration')
  return times, best_valids, best_tests, total_searched_graphs, valid_graphs_test_accuracy

def benchmark(save_dir):
     
    # Run random search and evolution search 10 times each. This should take a few
    # minutes to run. Note that each run would have taken days of compute to
    # actually train and evaluate if the dataset were not precomputed.
    random_data = {'times': list(), 'best_valid' : list(), 'best_test' : list(), 'worst_test' : list(), 'searched_graphs' : list(), 'num_valid_graphs': list(), 'valid_graphs_test_accuracy': list()}
    evolution_data = {'times' : list(), 'best_valid' : list(), 'best_test' : list(), 'worst_test' : list(), 'searched_graphs' : list(), 'num_valid_graphs': list(), 'valid_graphs_test_accuracy': list()}
    for repeat in range(8):
        print('Running iteration %d' % (repeat + 1))
        np.random.seed(12345670 + repeat)
        times, best_valid, best_test, random_searched_graphs, valid_graphs_test_accuracy = run_random_search()
        
        random_data['times'].append(np.mean(times))
        random_data['best_valid'].append(np.mean(best_valid))
        random_data['best_test'].append(np.max(valid_graphs_test_accuracy))
        random_data['worst_test'].append(np.min(valid_graphs_test_accuracy))
        random_data['searched_graphs'].append(len(random_searched_graphs))
        random_data['valid_graphs_test_accuracy'].append(np.mean(valid_graphs_test_accuracy))
        random_data['num_valid_graphs'].append(len(valid_graphs_test_accuracy))
        

        times, best_valid, best_test, evolution_searched_graphs, valid_graphs_test_accuracy = run_evolution_search()
        evolution_data['times'].append(np.mean(times))
        evolution_data['best_valid'].append(np.mean(best_valid))
        evolution_data['best_test'].append(np.max(valid_graphs_test_accuracy))
        evolution_data['worst_test'].append(np.min(valid_graphs_test_accuracy))
        evolution_data['searched_graphs'].append(len(evolution_searched_graphs))
        evolution_data['valid_graphs_test_accuracy'].append(np.mean(valid_graphs_test_accuracy))
        evolution_data['num_valid_graphs'].append(len(valid_graphs_test_accuracy))

    pickle.dump(random_data, open(os.path.join(save_dir, 'random_search.p'), 'wb'))
    pickle.dump(evolution_data, open(os.path.join(save_dir, 'evolutionary_search.p'), 'wb'))

    print('Random Search')
    print('Test Set - Test Accuracy Statistics')
    print('Mean ', np.mean(random_data['valid_graphs_test_accuracy']))
    print('Standard Deviation ', np.std(random_data['valid_graphs_test_accuracy']))
    print('Number of valid graphs ', np.mean(random_data['num_valid_graphs']))
    print('Number of graphs searched ', np.mean(random_data['searched_graphs']))
    print('Best Test Accuracy ', np.max(random_data['best_test']))
    print('Worst Test Accuracy ', np.min(random_data['worst_test']))

    print('Evolutionary Search')
    print('Test Set - Test Accuracy Statistics')
    print('Mean ', np.mean(evolution_data['valid_graphs_test_accuracy']))
    print('Standard Deviation ', np.std(evolution_data['valid_graphs_test_accuracy']))
    print('Number of valid graphs ', np.mean(evolution_data['num_valid_graphs']))
    print('Number of graphs searched ', np.mean(evolution_data['searched_graphs']))
    print('Best Test Accuracy ', np.max(evolution_data['best_test']))
    print('Worst Test Accuracy ', np.min(evolution_data['worst_test']))
