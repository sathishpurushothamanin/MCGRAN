import cma
import gc
import networkx as nx
import pickle
import torch
import numpy as np
from nasbench import api
from nasbench.lib.model_spec import is_upper_triangular
from utils.eval_helper import degree_stats, orbit_stats_all, clustering_stats, spectral_stats
from utils.deepgg_metrics import display_deepgg_metrics
from utils.logger import get_logger
from utils.vis_helper import draw_neural_architecture
from sklearn import metrics
from utils.dist_helper import compute_mmd, gaussian_emd
import csv
from utils.dist_helper import gaussian, emd, gaussian_tv

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_kernels
from eden.graph import vectorize

#nas-101 related imports
import tensorflow as tf
from scipy.sparse import csr_matrix
from nasbench.lib import model_metrics_pb2
import base64
import json

import os
#from utils.nas_evaluation_helper import display_embedding
#from utils.nas_evaluation_helper import validate_self_loops
#from utils.nas_evaluation_helper import validate_isolated_nodes
#from utils.nas_evaluation_helper import validate_nn_structure
#from utils.nas_evaluation_helper import validate_novelty
#from utils.nas_evaluation_helper import validate_uniqueness
#from utils.nas_evaluation_helper import compute_pairwise_kernel
#from utils.nas_evaluation_helper import compute_mmd_nspdk
#from utils.nas_evaluation_helper import add_labels


#logger = get_logger('exp_logger')

def validation_metrics(pred_label, actual_label):
    train_metrics = {'f1_score': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
    
    all_prediction = pred_label
    all_actual = actual_label
#    
#    all_prediction = list()
#    all_actual = list()
#    print(pred_label)
#    for index in range(len(pred_node_labels)):
#        all_prediction.extend(pred_node_labels[index])
#        all_actual.extend(actual_node_labels[index])
#    
#    print(all_prediction)
    train_metrics['f1_score'] = metrics.f1_score(all_actual, all_prediction, average='macro', zero_division=0)
    train_metrics['accuracy'] = metrics.accuracy_score(all_actual, all_prediction)
    train_metrics['precision'] = metrics.precision_score(all_actual, all_prediction, average='macro', zero_division=0)
    train_metrics['recall'] = metrics.recall_score(all_actual, all_prediction, average='macro', zero_division=0)
    
    return train_metrics

def validate_structural_metrics(graphs_gen):
    structure_evaluation_metrics = dict()
    structure_evaluation_metrics['self_loops'] = validate_self_loops(graphs_gen)
    structure_evaluation_metrics['isolated_nodes'] = validate_isolated_nodes(graphs_gen)
    structure_evaluation_metrics['invalid_nn'] = validate_nn_structure(graphs_gen)
    
    #dump the results
    return structure_evaluation_metrics

def validate_graph_quality_metrics(graphs_train, graphs_gen):
    structure_evaluation_metrics = dict()
    structure_evaluation_metrics['novelty'] = validate_novelty(graphs_train, graphs_gen)
    structure_evaluation_metrics['uniqueness'] = validate_uniqueness(graphs_gen)
    
    #dump the results
    return structure_evaluation_metrics

def validate_mmd_metrics(graphs_gen, graphs_train, graphs_dev, graphs_test):
    structure_evaluation_metrics = {'test': dict(), 'dev': dict()}
    num_nodes_gen = [gg.number_of_nodes() for gg in graphs_gen]
    
    #Compared with Validation Set
    num_nodes_dev = [len(gg.nodes) for gg in graphs_dev]  # shape B X 1
    
    mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(graphs_dev, graphs_gen, degree_only=False)
    mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)
    
    # # Compared with Test Set
    num_nodes_test = [len(gg.nodes) for gg in graphs_test]  # shape B X 1
    mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(graphs_test, graphs_gen, degree_only=False)
    mmd_num_nodes_test = compute_mmd([np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)
    
    structure_evaluation_metrics['test']['num_nodes'] = mmd_num_nodes_test
    structure_evaluation_metrics['test']['node_degree'] = mmd_degree_test
    structure_evaluation_metrics['test']['node_clustering'] = mmd_clustering_test
    structure_evaluation_metrics['test']['graph_4orbits'] = mmd_4orbits_test
    structure_evaluation_metrics['test']['graph_spectral'] = mmd_spectral_test
    
    structure_evaluation_metrics['dev']['num_nodes'] = mmd_num_nodes_dev
    structure_evaluation_metrics['dev']['node_degree'] = mmd_degree_dev
    structure_evaluation_metrics['dev']['node_clustering'] = mmd_clustering_dev
    structure_evaluation_metrics['dev']['graph_4orbits'] = mmd_4orbits_dev
    structure_evaluation_metrics['dev']['graph_spectral'] = mmd_spectral_dev
    
    #used in notebook analysis
#    pickle.dump(valid_vs_gen, open(os.path.join(config.save_dir, 'Validation_MMD_Scores_Validation_vs_Generated_Graphs.p'), 'wb'))
#    pickle.dump(test_vs_gen, open(os.path.join(config.save_dir, 'Validation_MMD_Scores_Test_vs_Generated_Graphs.p'), 'wb'))
    
#    logger.info("Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev))
#    logger.info("Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test))
    labeled_training_graphs = []
    for gg in graphs_train:
        graph = add_labels(gg, [str(gg.nodes.data()[node_index]['label']) for node_index in range(gg.number_of_nodes())])
        labeled_training_graphs.append(graph)

    labeled_test_graphs = []
    for gg in graphs_test:
        graph = add_labels(gg, [str(gg.nodes.data()[node_index]['label']) for node_index in range(gg.number_of_nodes())])
        labeled_test_graphs.append(graph)
    
    structure_evaluation_metrics['dev']['NSPDK'] = compute_mmd_nspdk(labeled_training_graphs, graphs_gen, compute_pairwise_kernel)
    structure_evaluation_metrics['test']['NSPDK'] = compute_mmd_nspdk(labeled_test_graphs, graphs_gen, compute_pairwise_kernel)
    
    #dump the results
    return structure_evaluation_metrics

def evaluate_metrics(config, A_pred, node_label_list, graphs_train, graphs_dev, graphs_test):
    
    assert len(A_pred) > 0
    
    graphs_gen = [nx.from_numpy_matrix(aa) for aa in A_pred]
    
    
    # Use nasbench_full.tfrecord for full dataset (run download command above).
    filepath = os.path.join(config.dataset.data_path, 'nasbench_only108.tfrecord')
    nasbench = api.NASBench(filepath, seed = config.seed)
    
    tag = 'test'
    node_label_vis = node_label_list[0]
    valid_graphs = []
    accuracy_list = []
    trainable_parameters_list = []
    training_time_list = []
    graph_label = 'valid'

    
    for index in range(config.test.batch_size):
        matrix = np.triu(np.transpose(A_pred[index].astype(np.int32)))
        graph = nx.Graph(csr_matrix(matrix))
        operations = {0:'input', 1:'conv3x3-bn-relu', 2:'conv1x1-bn-relu',
            3:'maxpool3x3', 4:'output'}
        additional_details = []
        ops = [operations[node_index] for node_index in node_label_vis[index].cpu().numpy().astype(np.int32)]
        # try:
        spec = api.ModelSpec(matrix=matrix, ops=ops)
    
        if nasbench.is_valid(spec):
            graph = nx.Graph(csr_matrix(spec.matrix))
            ops = spec.ops
            # fixed_stats, computed_stats = nasbench.get_metrics_from_spec(spec)
            data = nasbench.query(spec, epochs=108)
            additional_details.append(f"Trainable parameters {data['trainable_parameters']}")
            additional_details.append(f"Training time {data['training_time']}")
            additional_details.append(f"Accuracy {data['test_accuracy']}")
            accuracy_list.append(data['test_accuracy'])
            trainable_parameters_list.append(data['trainable_parameters'])
            training_time_list.append(data['training_time'])
            # additional_details.append(f"NTK Condition Number {get_ntk(spec)[0]}")
            valid_graphs.append(add_labels(graph, ops))
            graph_label = 'valid'
        else:
            graph_label = 'invalid'
            # try:
                # # data = nasbench.evaluate(spec, 'model')
                # # additional_details.append(f"Trainable parameters {data['trainable_parameters']}")
                # # additional_details.append(f"Accuracy {data['test_accuracy']}")
                # # accuracy_list.append(data['test_accuracy'])
                # additional_details.append(f"NTK Condition Number {get_ntk(spec)[0]}")
                # valid_graphs.append(graph)
            # except:
                # graph_label.append(1)
        draw_neural_architecture(graph, ops,
            file_name=os.path.join(config.save_dir,
            f"{graph_label}_architecture_{index}.svg"), additional_details=additional_details)
    
        tmp_path = os.path.join(config.save_dir, '{}_{}_a.p'.format(tag, index))
        tmp_path_label = os.path.join(config.save_dir, '{}_{}_l.p'.format(tag, index))
        pickle.dump(matrix, open(tmp_path, 'wb'))
        pickle.dump(node_label_vis[index].cpu().numpy().astype(np.int32), open(tmp_path_label, 'wb'))
        # except:
            # print("Invalid graph")
    
    
    # GRAN_Embedding = pickle.load(open('GRAN_Embedding.p', "rb"))
    # display_embedding(GRAN_Embedding, graph_label)
    
    # print(graph_label)
    # logger.info("Mean Accuracy: {}".format(sum(accuracy_list)/len(accuracy_list)))
    # pickle.dump(accuracy_list, open(os.path.join(config.save_dir, 'generated_graph_test_accuracy_list.p'), 'wb'))
    
    
    # generated_graph_filepath = os.path.join(config.save_dir, 'generated_graph_list.p')
    # save_graph_list(valid_graphs, generated_graph_filepath)
    
    print('Generated Graphs - Test Accuracy Statistics')
    print('Mean ', np.mean(accuracy_list))
    print('Standard Deviation ', np.std(accuracy_list))
    print('Maximum ', np.max(accuracy_list))
    print('Minimum ', np.min(accuracy_list))
    print('Number of valid graphs ', len(accuracy_list))

    print('Generated Graphs - Model Trainable Parameters Statistics')
    print('Mean ', np.mean(trainable_parameters_list))
    print('Standard Deviation ', np.std(trainable_parameters_list))
    print('Maximum ', np.max(trainable_parameters_list))
    print('Minimum ', np.min(trainable_parameters_list))

    print('Generated Graphs - Model Training Time Statistics')
    print('Mean ', np.mean(training_time_list))
    print('Standard Deviation ', np.std(training_time_list))
    print('Maximum ', np.max(training_time_list))
    print('Minimum ', np.min(training_time_list))
    
    accuracy_list = list()
    operations = {0:'input', 1:'conv3x3-bn-relu', 2:'conv1x1-bn-relu',
            3:'maxpool3x3', 4:'output'}
    
    for gg in graphs_train:
        ops = [operations[gg.nodes.data()[node_index]['label']] for node_index in range(gg.number_of_nodes())]
        spec = api.ModelSpec(matrix=np.triu(nx.to_numpy_matrix(gg, dtype=np.int32)), ops=ops)
        # fixed_stats, computed_stats = nasbench.get_metrics_from_spec(spec)
        data = nasbench.query(spec, epochs=108)
        accuracy_list.append(data['test_accuracy'])
       
    print('Training Graphs - Test Accuracy Statistics')
    print('Mean ', np.mean(accuracy_list))
    print('Standard Deviation ', np.std(accuracy_list))
    print('Maximum ', np.max(accuracy_list))
    print('Minimum ', np.min(accuracy_list))
    print('Number of valid graphs ', len(accuracy_list))



    graphs_gen = list()
    for index in range(config.test.batch_size):
        matrix = np.triu(np.transpose(A_pred[index].astype(np.int32)))
        graph = nx.Graph(csr_matrix(matrix))
    
        ops = [operations[node_index] for node_index in node_label_vis[index].cpu().numpy().astype(np.int32)]
        graphs_gen.append(add_labels(graph, ops))
    
    structure_evaluation_metrics = validate_mmd_metrics(graphs_gen, graphs_train, graphs_dev, graphs_test)
    metrics = validate_structural_metrics(graphs_gen)

    structure_evaluation_metrics['self_loops'] = metrics['self_loops']
    structure_evaluation_metrics['isolated_nodes'] = metrics['isolated_nodes']
    structure_evaluation_metrics['invalid_nn'] = metrics['invalid_nn']

    metrics = validate_graph_quality_metrics(graphs_train, graphs_gen)

    structure_evaluation_metrics['novelty'] = metrics['novelty']
    structure_evaluation_metrics['uniqueness'] = metrics['uniqueness']
 
    return structure_evaluation_metrics

def print_info(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
#    print('Inside ' + __class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    # print(output)
    # tmp_path = os.path.join(config.s)
    pickle.dump(output.data.squeeze().cpu().numpy(), open('GRAN_Embedding.p', 'wb'))

def free_up_cache():
    gc.collect()
    torch.cuda.empty_cache()

def compute_edge_ratio(G_list):
  num_edges_max, num_edges = .0, .0
  print(len(G_list))
  for gg in G_list:
    num_nodes = gg.number_of_nodes()
    num_edges += gg.number_of_edges()
    num_edges_max += num_nodes**2

  ratio = (num_edges_max - num_edges) / num_edges
  return ratio


def get_graph(adj):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  adj = adj[~np.all(adj == 0, axis=1)]
  adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_matrix(adj)
  return G


def evaluate(graph_gt, graph_pred, degree_only=True):
  mmd_degree = degree_stats(graph_gt, graph_pred)

  if degree_only:
    mmd_4orbits = 0.0
    mmd_clustering = 0.0
    mmd_spectral = 0.0
  else:
    mmd_4orbits = orbit_stats_all(graph_gt, graph_pred)
    mmd_clustering = clustering_stats(graph_gt, graph_pred)
    mmd_spectral = spectral_stats(graph_gt, graph_pred)


  return mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral

def display_model(model):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
      print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def evolution(model, optimization_layer_name, input_dict, config):
    seed = config.seed
    # Use nasbench_full.tfrecord for full dataset (run download command above).
    filepath = os.path.join(config.dataset.data_path, 'nasbench_only108.tfrecord')
    nasbench = api.NASBench(filepath, seed = seed)

    old_state = model.state_dict()
    new_state = old_state
    model.eval()

    output_latent_space  = model.state_dict()[optimization_layer_name]
    shape = output_latent_space.shape
    output_latent_space = output_latent_space.view(-1).cpu()

    # es = cma.CMAEvolutionStrategy(output_theta_latent_space, 0.5, {})
    es = cma.CMAEvolutionStrategy(output_latent_space, 0.1, {'seed':seed, 'maxiter':100})
    count = 100
    while es.best.f > 0.07:
        solutions = es.ask()
        performances = []
        for solution in solutions:
            new_state[optimization_layer_name] = torch.tensor(solution).reshape(shape)

            model.load_state_dict(new_state)
            A_tmp, node_label_list = model(input_dict)
            A_pred = [aa.data.long().cpu().numpy() for aa in A_tmp]
            graphs_gen = [nx.from_numpy_matrix(aa) for aa in A_pred]
            no_degenerate_graphs = True
            for graph_index in range(len(graphs_gen)):
                if graphs_gen[graph_index].number_of_nodes() < 5:
                    no_degenerate_graphs = False
            if no_degenerate_graphs:
                # performances.append(get_performance(A_pred, node_label_list, 'ntk'))
                performances.append(get_performance(A_pred, node_label_list, nasbench, metric = 'acc'))
            else:
                performances.append(1)
        es.tell(solutions, performances)
        es.logger.add()  # write data to disc to be plotted
        es.disp()
        if count < 0:
            break
        else:
            count = count - 1

    new_state[optimization_layer_name] = torch.tensor(es.best.__dict__['x']).reshape(shape)

    return new_state

def get_performance(A_pred, node_label_list, nasbench, metric = 'acc'):

    best_accuracy = 0
    operations = {0:'input', 1:'conv3x3-bn-relu', 2:'conv1x1-bn-relu',
        3:'maxpool3x3', 4:'output'}

    valid_graphs = 0
    ntk = 0
    for graph_index in range(node_label_list.shape[0]):
        if is_upper_triangular(np.transpose(A_pred[graph_index].astype(np.int32))):
            ops = [operations[node_index] for node_index in node_label_list[graph_index].cpu().numpy().astype(np.int32)]
            spec = api.ModelSpec(matrix=np.triu(np.transpose(A_pred[graph_index].astype(np.int32))), ops=ops)
            if nasbench.is_valid(spec):
                if metric == 'acc':
                    try:
                        fixed_stats, computed_stats = nasbench.get_metrics_from_spec(spec)
                        best_accuracy += computed_stats[108][0]['final_test_accuracy']
                        # print(graph_index, computed_stats[108][0]['final_test_accuracy'])
                        valid_graphs += 1
                    except:
                        print("Error in processing")
#                elif metric == 'ntk':
#                    ntk += sum(get_ntk(spec))
                else:
                    print("Unknown metric specified")

    # print("\nTotal number of valid graphs is ", valid_graphs)
    if metric == 'acc':
        performance = 1 if valid_graphs < node_label_list.shape[0]/2 else  (1 - best_accuracy/valid_graphs)
    elif metric == 'ntk':
        performance = 1 if valid_graphs == 0 else  (1 - ntk/valid_graphs)
    else:
        performance = None
    return performance  #idea is to increase the accuracy of the architecture as optimization will reduce w.r.t objective function


def compute_pairwise_kernel(X, Y=None):
    X = vectorize(X, complexity=4, discrete=True)

    if Y is not None:
        Y = vectorize(Y, complexity=4, discrete=True)
    
    return pairwise_kernels(X, Y, metric='linear')

def compute_mmd_nspdk(train_graphs, pred_graphs, compute_pairwise_kernel):
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
    
    
def get_data(data_dir, filename, max_num_nodes=7):
    graphs = []
    #tf.compat.v1.enable_eager_execution()
    #count = 1
    if os.path.exists(os.path.join(data_dir, filename)):
        # dataset = tf.compat.v1.python_io.tf_record_iterator(os.path.join(data_dir, filename))
        dataset = tf.data.TFRecordDataset(os.path.join(data_dir, filename))
        operations = {'input':0, 'conv3x3-bn-relu':1, 'conv1x1-bn-relu':2, 'maxpool3x3':3, 'output':4}
        for serialized_row in dataset:
            module_hash, epochs, raw_adjacency, raw_operations, raw_metrics = json.loads(serialized_row.numpy().decode('utf-8'))
#            metrics = model_metrics_pb2.ModelMetrics.FromString(base64.b64decode(raw_metrics))
            dim = int(np.sqrt(len(raw_adjacency)))
            adjacency = np.array([int(e) for e in list(raw_adjacency)], dtype=np.int8)
            adjacency = np.reshape(adjacency, (dim, dim))
            graph = nx.Graph(csr_matrix(adjacency))
            graph_ops = raw_operations.split(',')
            graph = add_labels(graph, [operations[graph_ops[node_idx]] for node_idx in range(graph.number_of_nodes())])
            # for node_idx in range(graph.number_of_nodes()):
                # graph.add_node(node_idx,
                # label=str(operations[graph_ops[node_idx]]))
            # labels = '-'
            # nx.set_edge_attributes(graph, labels, "label")
            if graph.number_of_nodes() == max_num_nodes:
              graphs.append(graph)
    else:
        print('file does not exist')
    return graphs

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
    
    if len(normalized_freq_distribution.keys()) == 0:
        normalized_freq_distribution[0] = 100.0

    return normalized_freq_distribution


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
    
    if len(normalized_freq_distribution.keys()) == 0:
        normalized_freq_distribution[0] = 100.0

    return normalized_freq_distribution


def output_dataset_csv(data, column_names, file_name):
    with open(file_name,'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        writer.writeheader()
        for index in range(len(data)):
            dataitem = dict()
            for column_name in column_names:
                dataitem[column_name] = data[index][column_name]
            writer.writerow(dataitem)

def output_search_stats_csv(data, column_names, file_name):
    with open(file_name,'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_names)
        writer.writeheader()
        for index in range(len(data)):
            dataitem = dict()
            for column_index, column_name in enumerate(column_names):
                dataitem[column_name] = data[index][column_index]
            writer.writerow(dataitem)