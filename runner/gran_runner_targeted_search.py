from __future__ import (division, print_function)
import os
import time

import numpy as np
import networkx as nx
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

import tensorflow as tf
from scipy.sparse import csr_matrix
from nasbench.lib import model_metrics_pb2
import base64
import json
from nasbench import api
from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.vis_helper import draw_neural_architecture, visualize_graphs
from utils.vis_helper import draw_graph_list, draw_graph_list_separate 


from utils.data_parallel import DataParallel
#from scipy.sparse import csr_matrix
from utils.nas_evaluation_helper import *

#from nas_101_api.nas101_model import *
#from nas_101_api.nas101_model import *
# from runner.geometric_processing import *

from utils.runner_helper import print_info, free_up_cache
from utils.runner_helper import compute_edge_ratio, get_graph
from utils.runner_helper import evaluate_metrics, evaluate, validation_metrics
from utils.runner_helper import evolution
from utils.runner_helper import display_model, add_labels
from utils.runner_helper import output_dataset_csv, output_search_stats_csv

from nasbench.api import OutOfDomainError
from nasbench.lib import config
#from nasbench.lib.model_spec import is_upper_triangular

#from sklearn import metrics
from sklearn import preprocessing
try:
  ###
  # workaround for solving the issue of multi-worker
  # https://github.com/pytorch/pytorch/issues/973
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
  ###
except:
  pass


# Use nasbench_full.tfrecord for full dataset (run download command above).
filepath = os.path.join('data/nas-101', 'nasbench_only108.tfrecord')
nasbench = api.NASBench(filepath, seed = 1234)
nasbench_config = config.build_config()
    
logger = get_logger('exp_logger')
__all__ = ['GranRunner_Targeted_Search', 'compute_edge_ratio', 'get_graph', 'evaluate']

class GranRunner_Targeted_Search(object):

  def __init__(self, config):
    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
#    self.writer = SummaryWriter(config.save_dir)
    self.is_vis = config.test.is_vis
    self.better_vis = config.test.better_vis
    self.num_vis = config.test.num_vis
    self.vis_num_row = config.test.vis_num_row
    self.is_single_plot = config.test.is_single_plot
    self.num_gpus = len(self.gpus)
    self.is_shuffle = False
    
    self.config.num_categories = config.model.node_categories
    self.config.model.tree_level = 0
    
    assert self.use_gpu == True

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    ### load graphs
    self.dataset = create_graphs(config.dataset.name,
        data_dir=config.dataset.data_path,
        max_num_samples=config.dataset.max_num_samples,
        min_test_accuracy=config.dataset.min_test_accuracy,
        max_test_accuracy=config.dataset.max_test_accuracy,
        max_num_nodes=config.model.max_num_nodes)

    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride
    self.num_graphs = len(self.dataset)
   
#    self.all_datasets
    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.dataset)
    
    self.graphs = list()
    self.total_parameters = list()
    self.total_training_time = list()
    self.test_accuracy = list()
    self.hash_list = set()
    for data_item in self.dataset:
        self.graphs.append(data_item['graph'])
        self.total_parameters.append(data_item['total_parameters'])
        self.total_training_time.append(data_item['total_training_time'])
        self.test_accuracy.append(data_item['test_accuracy'])
        self.hash_list.add(data_item['hash'])
    
    self.graphs_train = self.graphs
    self.total_parameters_train = self.total_parameters
    self.total_training_time_train = self.total_training_time
    self.test_accuracy_train = self.test_accuracy
    

    logger.info('Training Set')
    logger.info('Test Accuracy Statistics Mean: {:.4f}; SD: {:.4f}; Max: {:.4f}; Min: {:.4f}; Median: {:.4f}'.format(np.mean(self.test_accuracy_train), np.std(self.test_accuracy_train), np.max(self.test_accuracy_train), np.min(self.test_accuracy_train), np.median(self.test_accuracy_train)))
        
    self.search_train_stats = list()
    self.search_gen_stats = list()

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
#    self.max_num_nodes = len(self.num_nodes_pmf_train)
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()


  def train(self):
    ### create data loader
    
    self.search_train_stats.append([np.mean(self.test_accuracy_train),
                                    np.std(self.test_accuracy_train),
                                    np.max(self.test_accuracy_train),
                                    np.min(self.test_accuracy_train),
                                    np.median(self.test_accuracy_train)])
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, self.total_parameters_train, self.total_training_time_train, self.test_accuracy_train, tag='train_{}'.format(self.config.model.max_num_nodes))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    if self.use_gpu:
      model = DataParallel(model, device_ids=self.gpus).to(self.device)

    #display model details
    if self.config.model.display_detailed_model:
        display_model(model)
        
    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

#    early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_epoch,
        gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

#    best_f1_score = 0
    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.config.save_dir, 
                                "model_snapshot_search_{}_{}.pth".format(self.config.model.max_num_nodes, self.config.model.tree_level))
      load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer,
          scheduler=lr_scheduler)
      resume_epoch = self.train_conf.resume_epoch
    
    # Training Loop
    iter_count = 0
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()
      train_iterator = train_loader.__iter__()

      pred_label_list = list()
      actual_label_list = list()
      for inner_iter in range(len(train_loader) // self.num_gpus):
        optimizer.zero_grad()

        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = train_iterator.next()
            batch_data.append(data)
            iter_count += 1


        avg_train_loss = .0
#        avg_train_acc = .0
        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = []

          if self.use_gpu:
            for dd, gpu_id in enumerate(self.gpus):
              data = {}
              data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              data['edge_list'] = batch_data[dd][ff]['edge_list']
              data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_labels'] = batch_data[dd][ff]['node_labels'].pin_memory().to(gpu_id, non_blocking=True)
              data['conditional_data'] = batch_data[dd][ff]['conditional_data'].pin_memory().to(gpu_id, non_blocking=True)
              batch_fwd.append((data,))
          if batch_fwd:
            train_loss, pred_label, actual_label = model(*batch_fwd)
            pred_label = pred_label.cpu().tolist()
            actual_label = actual_label.cpu().tolist()
            for index in range(len(pred_label)):
              pred_label_list.extend(pred_label[index])
              actual_label_list.extend(actual_label[index])


            avg_train_loss += train_loss
#            avg_train_acc += train_acc

            # assign gradient
            train_loss.backward()

        clip_grad_norm_(model.parameters(), 5.0e-0)
        optimizer.step()
        lr_scheduler.step()
        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)

        # reduce
        train_loss = float(avg_train_loss.data.cpu().numpy())
      classifier_train_metrics = validation_metrics(pred_label = pred_label_list, actual_label = actual_label_list)
      if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
        logger.info("Training Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, train_loss, classifier_train_metrics['accuracy'], classifier_train_metrics['precision'], classifier_train_metrics['recall'], classifier_train_metrics['f1_score']))

    
      results['train_loss'] += [train_loss]
      results['train_acc'] += [classifier_train_metrics['accuracy']]
      results['train_f1_score'] += [classifier_train_metrics['f1_score']]
      results['train_step'] += [iter_count]
     
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, tag="search_{}_{}".format(self.config.model.max_num_nodes, self.config.model.tree_level), scheduler=lr_scheduler)

    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
#    self.writer.close()

    return 1

  def test(self):

    ### load model
    model = eval(self.model_conf.name)(self.config)
    model_file = os.path.join(self.config.save_dir, "model_snapshot_search_{}_{}.pth".format(self.config.model.max_num_nodes, self.config.model.tree_level))

    load_model(model, model_file, self.device)


    if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

    #display model details
    if self.config.model.display_detailed_model:
        display_model(model)
    
    model.eval()

    ### Generate Graphs
    self.A_pred = []
    num_nodes_pred = []
    self.test_conf.batch_size

    #high performance architecture search - keeping the last term always one
    conditional_data = np.concatenate((np.float32(np.random.choice([0, 1], 
                                           size=(self.test_conf.batch_size, 2))), 
                                           np.ones((self.test_conf.batch_size,1))), axis=1)
    
    conditional_data = torch.from_numpy(conditional_data).float()
    
    gen_run_time = []
    self.graphs_gen_nodes = []

    with torch.no_grad():
      start_time = time.time()
      input_dict = {}
      input_dict['is_sampling']=True
      input_dict['batch_size']=self.test_conf.batch_size
      input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
      input_dict['conditional_data'] = conditional_data
      input_dict['expanded_search'] = False
      A_tmp, graphs_node_list = model(input_dict)
      gen_run_time += [time.time() - start_time]
      self.A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
      num_nodes_pred += [aa.shape[0] for aa in A_tmp]
      self.graphs_gen_nodes.extend(graphs_node_list)

    logger.info('Average test time per mini-batch = {}'.format(
      np.mean(gen_run_time)))
 

    node_label_list = self.graphs_gen_nodes
    median_accuracy_list = np.median(self.test_accuracy_train)
    self.valid_graphs = []
    self.accuracy_list = []
    self.trainable_parameters_list = []
    self.training_time_list = []
    for index in range(self.test_conf.batch_size):
        
        matrix = np.triu(np.transpose(self.A_pred[index].astype(np.int32)))
        graph = nx.Graph(csr_matrix(matrix))
        operations_encode = {'input':0, 'conv3x3-bn-relu':1, 'conv1x1-bn-relu':2, 'maxpool3x3':3, 'output':4}
        operations_decode = {0:'input', 1:'conv3x3-bn-relu', 2:'conv1x1-bn-relu',
            3:'maxpool3x3', 4:'output'}
        ops = [operations_decode[node_index] for node_index in node_label_list[index].cpu().numpy().astype(np.int32)]
        try:
            spec = api.ModelSpec(matrix=matrix, ops=ops)
        
            if nasbench.is_valid(spec):
                graph = nx.Graph(csr_matrix(spec.matrix))
                ops = spec.ops
                # fixed_stats, computed_stats = nasbench.get_metrics_from_spec(spec)
                data = nasbench.query(spec, epochs=108)
                if graph.number_of_nodes() == self.config.model.max_num_nodes:
                    for node_idx in range(graph.number_of_nodes()):
                          graph.add_node(node_idx,
                           label=operations_encode[ops[node_idx]])
                    graph_hash = spec.hash_spec(nasbench_config['available_ops'])
                    if graph_hash not in self.hash_list and data['test_accuracy'] > median_accuracy_list:
                        self.valid_graphs.append(graph)
                        self.accuracy_list.append(data['test_accuracy'])
                        self.trainable_parameters_list.append(data['trainable_parameters'])
                        self.training_time_list.append(data['training_time'])
                        self.hash_list.add(graph_hash)
        except OutOfDomainError as OOD:
            #Todo
            #save invalid matrix and investigate
            print(OOD)
        except ValueError:
            pass
    if len(self.accuracy_list) > 0:
        logger.info('Generated Graphs')
        logger.info('Number of valid graphs {:04d}'.format(len(self.accuracy_list)))
        logger.info('Test Accuracy Statistics Mean: {:.4f}; SD: {:.4f}; Max: {:.4f}; Min: {:.4f}; Median: {:.4f}'.format(np.mean(self.accuracy_list), np.std(self.accuracy_list), np.max(self.accuracy_list), np.min(self.accuracy_list), np.median(self.accuracy_list)))
        logger.info('Model Trainable Parameters Statistics Mean: {:.4f}; SD: {:.4f}; Max: {:.4f}; Min: {:.4f}; Median: {:.4f}'.format(np.mean(self.trainable_parameters_list), np.std(self.trainable_parameters_list), np.max(self.trainable_parameters_list), np.min(self.trainable_parameters_list), np.median(self.trainable_parameters_list)))
        logger.info('Model Training Time Statistics Mean: {:.4f}; SD: {:.4f}; Max: {:.4f}; Min: {:.4f}; Median: {:.4f}'.format(np.mean(self.training_time_list), np.std(self.training_time_list), np.max(self.training_time_list), np.min(self.training_time_list), np.median(self.training_time_list)))
        self.search_gen_stats.append([np.mean(self.accuracy_list),
                                        np.std(self.accuracy_list),
                                        np.max(self.accuracy_list),
                                        np.min(self.accuracy_list),
                                        np.median(self.accuracy_list)])

  def targeted_search(self):

    #set model and data selection dynamically

    for _ in range(250):

        self.train()

        self.test()

        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
        
        self.update_dataset()
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5
    
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}.p'.format(self.config.model.max_num_nodes))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}.p'.format(self.config.model.max_num_nodes))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}.p'.format(self.config.model.max_num_nodes))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))


  def update_dataset(self):
    
    self.graphs = list()
    self.total_parameters = list()
    self.total_training_time = list()
    self.test_accuracy = list()
    
    for index in range(len(self.accuracy_list)):
        
        data_item = dict()
        data_item['graph'] = self.valid_graphs[index]
        data_item['total_parameters'] = self.trainable_parameters_list[index]
        data_item['total_training_time'] = self.training_time_list[index]
        data_item['test_accuracy'] = self.accuracy_list[index]
        self.dataset.append(data_item)

    for data_item in self.dataset[len(self.dataset)-20:]:
        self.graphs.append(data_item['graph'])
        self.total_parameters.append(data_item['total_parameters'])
        self.total_training_time.append(data_item['total_training_time'])
        self.test_accuracy.append(data_item['test_accuracy'])
 
    self.graphs_train = self.graphs
    self.total_parameters_train = self.total_parameters
    self.total_training_time_train = self.total_training_time
    self.test_accuracy_train = self.test_accuracy

    logger.info('Training Set')
    logger.info('Test Accuracy Statistics Mean: {:.4f}; SD: {:.4f}; Max: {:.4f}; Min: {:.4f}; Median: {:.4f}'.format(np.mean(self.test_accuracy_train), np.std(self.test_accuracy_train), np.max(self.test_accuracy_train), np.min(self.test_accuracy_train), np.median(self.test_accuracy_train)))
    
    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()