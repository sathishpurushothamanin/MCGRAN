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
    
logger = get_logger('exp_logger')
__all__ = ['GranRunner_Higher_Order_Search', 'compute_edge_ratio', 'get_graph', 'evaluate']

class GranRunner_Higher_Order_Search(object):

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
    
    self.config.num_categories = 10
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
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))
   
#    self.all_datasets
    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.dataset)
    
    self.graphs = list()
    self.total_parameters = list()
    self.total_training_time = list()
    self.test_accuracy = list()
    
    for data_item in self.dataset:
        self.graphs.append(data_item['graph'])
        self.total_parameters.append(data_item['total_parameters'])
        self.total_training_time.append(data_item['total_training_time'])
        self.test_accuracy.append(data_item['test_accuracy'])
    
    self.all_datasets = {self.config.model.tree_level: {self.config.model.max_num_nodes: self.dataset}}
    
    estimator = preprocessing.KBinsDiscretizer(n_bins=self.config.num_categories, 
                                                 encode='onehot', strategy='uniform',
                                                 dtype=np.float32)
    
    self.supernode_categories = estimator.fit_transform(np.array(self.test_accuracy).reshape(-1, 1))
    self.supernode_categories = csr_matrix(self.supernode_categories).todense().tolist()
    
    self.expanded_dataset = list()
    
    self.graphs_train = self.graphs[:self.num_train]
    self.total_parameters_train = self.total_parameters[:self.num_train]
    self.total_training_time_train = self.total_training_time[:self.num_train]
    self.test_accuracy_train = self.test_accuracy[:self.num_train]
    self.supernode_categories_train =  self.supernode_categories[:self.num_train]

    self.graphs_dev = self.graphs[:self.num_dev]
    self.total_parameters_dev = self.total_parameters[:self.num_dev]
    self.total_training_time_dev = self.total_training_time[:self.num_dev]
    self.test_accuracy_dev = self.test_accuracy[:self.num_dev]
    self.supernode_categories_dev =  self.supernode_categories[:self.num_dev]
    
    self.graphs_test = self.graphs[self.num_train:]
    self.total_parameters_test = self.total_parameters[self.num_train:]
    self.total_training_time_test = self.total_training_time[self.num_train:]
    self.test_accuracy_test = self.test_accuracy[self.num_train:]
    self.supernode_categories_test =  self.supernode_categories[self.num_train:]
    
    print('Training Set - Test Accuracy Statistics')
    print('Mean ', np.mean(self.test_accuracy_train))
    print('Standard Deviation ', np.std(self.test_accuracy_train))
    print('Maximum ', np.max(self.test_accuracy_train))
    print('Minimum ', np.min(self.test_accuracy_train))
    print('Number of valid graphs ', len(self.test_accuracy_train))
    
    self.search_train_stats = list()
    self.search_gen_stats = list()

#    training_ds_operations_freq = dict()
#    for G in self.graphs_train:
#        for layer in [node['label'] for idx, node in G.nodes.data()]:
#            if layer in training_ds_operations_freq.keys():
#                training_ds_operations_freq[layer] +=1
#            else:
#                training_ds_operations_freq[layer] = 1
#    for index in range(5):
#        if index not in training_ds_operations_freq.keys():
#            training_ds_operations_freq[index] = 0.0001
#    self.class_weights = [training_ds_operations_freq[0],
#        training_ds_operations_freq[1],
#        training_ds_operations_freq[2],
#        training_ds_operations_freq[3],
#        training_ds_operations_freq[4]]
#
#    self.config.class_weights = self.class_weights



    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
#    self.max_num_nodes = len(self.num_nodes_pmf_train)
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()

    ### save split for benchmarking
    if config.dataset.is_save_split:
      base_path = os.path.join(config.dataset.data_path, 'save_split')
      if not os.path.exists(base_path):
        os.makedirs(base_path)

      save_graph_list(
          self.graphs_train,
          os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_dev,
          os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_test,
          os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))


  def train(self):
    ### create data loader
    
    self.search_train_stats.append([np.mean(self.test_accuracy_train),
                                    np.std(self.test_accuracy_train),
                                    np.max(self.test_accuracy_train),
                                    np.min(self.test_accuracy_train)])
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, self.total_parameters_train, self.total_training_time_train, self.test_accuracy_train, self.supernode_categories_train, tag='train_{}'.format(self.config.model.max_num_nodes))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)
#
#    val_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_dev, self.total_parameters_dev, self.total_training_time_dev, self.test_accuracy_dev, tag='dev_{}'.format(self.config.model.max_num_nodes))
#    val_loader = torch.utils.data.DataLoader(
#        val_dataset,
#        batch_size=self.train_conf.batch_size,
#        shuffle=self.train_conf.shuffle,
#        num_workers=self.train_conf.num_workers,
#        collate_fn=train_dataset.collate_fn,
#        drop_last=False)

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
      lr_scheduler.step()
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
              data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_labels'] = batch_data[dd][ff]['node_labels'].pin_memory().to(gpu_id, non_blocking=True)
              data['pseudo_coordinates'] = batch_data[dd][ff]['pseudo_coordinates'].pin_memory().to(gpu_id, non_blocking=True)
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
        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)
#        avg_train_acc /= float(self.dataset_conf.num_fwd_pass)
#        if epoch == 10:
#          print(train_loss.testing)
        # reduce
        train_loss = float(avg_train_loss.data.cpu().numpy())
      classifier_train_metrics = validation_metrics(pred_label = pred_label_list, actual_label = actual_label_list)
      if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
        logger.info("Training Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, train_loss, classifier_train_metrics['accuracy'], classifier_train_metrics['precision'], classifier_train_metrics['recall'], classifier_train_metrics['f1_score']))
#      if epoch % 1 == 0:
#        model.eval()
#        with torch.no_grad():
#          val_iterator = val_loader.__iter__()
#          pred_label_list = list()
#          actual_label_list = list()
#          for inner_iter in range(len(val_loader) // self.num_gpus):
#            batch_data = []
#            if self.use_gpu:
#              for _ in self.gpus:
#                data = val_iterator.next()
#                batch_data.append(data)
#                iter_count += 1
#
#
#            avg_val_loss = .0
##            avg_val_acc = .0
#            for ff in range(self.dataset_conf.num_fwd_pass):
#              batch_fwd = []
#
#              if self.use_gpu:
#                for dd, gpu_id in enumerate(self.gpus):
#                  data = {}
#                  data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['node_labels'] = batch_data[dd][ff]['node_labels'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['pseudo_coordinates'] = batch_data[dd][ff]['pseudo_coordinates'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['conditional_data'] = batch_data[dd][ff]['conditional_data'].pin_memory().to(gpu_id, non_blocking=True)
#                  batch_fwd.append((data,))
#              if batch_fwd:
#                val_loss, pred_label, actual_label = model(*batch_fwd)
#                pred_label = pred_label.cpu().tolist()
#                actual_label = actual_label.cpu().tolist()
#                for index in range(len(pred_label)):
#                  pred_label_list.extend(pred_label[index])
#                  actual_label_list.extend(actual_label[index])
#
#                avg_val_loss += val_loss
##                avg_val_acc += val_acc
#
#                # assign gradient
#            avg_val_loss /= float(self.dataset_conf.num_fwd_pass)
##            avg_val_acc /= float(self.dataset_conf.num_fwd_pass)
#
#            # reduce
#            val_loss = float(avg_val_loss.data.cpu().numpy())
##            val_acc = float(avg_val_acc.data.cpu().numpy())
#
#        classifier_validation_metrics = validation_metrics(pred_label = pred_label_list, actual_label = actual_label_list)
#        if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
#          logger.info("Validation Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, val_loss, classifier_validation_metrics['accuracy'], classifier_validation_metrics['precision'], classifier_validation_metrics['recall'], classifier_validation_metrics['f1_score']))

#      self.writer.add_scalar('train_loss', train_loss, epoch)
#      self.writer.add_scalar('train_acc', classifier_train_metrics['accuracy'], epoch)
#      self.writer.add_scalar('train_f1_score', classifier_train_metrics['f1_score'], epoch)
#      self.writer.add_scalar('val_loss', val_loss, epoch)
#      self.writer.add_scalar('val_acc', classifier_validation_metrics['accuracy'], epoch)
#      self.writer.add_scalar('val_f1_score', classifier_validation_metrics['f1_score'], epoch)
    
      results['train_loss'] += [train_loss]
      results['train_acc'] += [classifier_train_metrics['accuracy']]
      results['train_f1_score'] += [classifier_train_metrics['f1_score']]
#      results['val_loss'] += [val_loss]
#      results['val_acc'] += [classifier_validation_metrics['accuracy']]
#      results['val_f1_score'] = [classifier_validation_metrics['f1_score']]
      results['train_step'] += [iter_count]
      
      print("Training Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, train_loss, classifier_train_metrics['accuracy'], classifier_train_metrics['precision'], classifier_train_metrics['recall'], classifier_train_metrics['f1_score']))
#      print("Validation Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, val_loss, classifier_validation_metrics['accuracy'], classifier_validation_metrics['precision'], classifier_validation_metrics['recall'], classifier_validation_metrics['f1_score']))
#      if epoch > 250 and classifier_validation_metrics['f1_score'] > best_f1_score:
#          best_f1_score = classifier_validation_metrics['f1_score']
#          snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, tag="search_{}".format(self.config.model.max_num_nodes), scheduler=lr_scheduler)

      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        #to do modify model name
#        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, tag="search_{}".format(self.config.model.max_num_nodes), scheduler=lr_scheduler)
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, tag="search_{}_{}".format(self.config.model.max_num_nodes, self.config.model.tree_level), scheduler=lr_scheduler)

    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
#    self.writer.close()

    return 1

  def test(self):

    ### load model
    model = eval(self.model_conf.name)(self.config)
    model_file = os.path.join(self.config.save_dir, "model_snapshot_search_{}_{}.pth".format(self.config.model.max_num_nodes, self.config.model.tree_level))
    #to do
    #modify test model name
#    model_file = os.path.join(self.test_conf.test_model_dir, self.test_conf.test_model_name)
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
    num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

    if self.config.model.tree_level ==  0:
        conditional_data = np.eye(self.config.num_categories)
        conditional_data = conditional_data[np.random.choice(range(self.config.num_categories), size=(self.test_conf.batch_size))]
    else:
        conditional_data = np.float32(np.random.choice([0, 1], 
                                           size=(self.test_conf.batch_size, self.config.num_categories)))
#        conditional_data = np.concatenate((np.float32(np.random.choice([0, 1], 
#                                           size=(self.test_conf.batch_size, 2))), 
#                                           np.ones((self.test_conf.batch_size,1))), axis=1)
    
    conditional_data = torch.from_numpy(conditional_data).float()
    
    gen_run_time = []
    self.graphs_gen_nodes = []

    for ii in tqdm(range(num_test_batch)):
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
    
#    conditional_data = list()
#    test_dataset = eval(self.dataset_conf.loader_name)(self.config, 
#                       self.graphs_test[:self.num_test_gen], 
#                       self.total_parameters_test[:self.num_test_gen], 
#                       self.total_training_time_test[:self.num_test_gen], 
#                       self.test_accuracy_test[:self.num_test_gen], 
#                       self.supernode_categories_test[:self.num_test_gen], 
#                       tag='test')
#    test_loader = torch.utils.data.DataLoader(
#        test_dataset,
#        batch_size=self.test_conf.batch_size,
#        shuffle=False,
#        num_workers=self.test_conf.num_workers,
#        collate_fn=test_dataset.collate_fn,
#        drop_last=False)
#    
#    gen_run_time = []
#    self.graphs_gen_nodes = []
#    for ii in tqdm(range(num_test_batch)):
#        test_iterator = test_loader.__iter__()
#        batch_data = list()
#        for inner_iter in range(len(test_loader) // self.num_gpus):
#            batch_data = []
#            if self.use_gpu:
#              for _ in self.gpus:
#                data = test_iterator.next()
#                batch_data.append(data)
#        
#        for ff in range(self.dataset_conf.num_fwd_pass):
#            batch_fwd = []
#
#            if self.use_gpu:
#                for dd, gpu_id in enumerate(self.gpus):
#                  data = {}
#                  data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['node_labels'] = batch_data[dd][ff]['node_labels'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['pseudo_coordinates'] = batch_data[dd][ff]['pseudo_coordinates'].pin_memory().to(gpu_id, non_blocking=True)
#                  data['conditional_data'] = batch_data[dd][ff]['conditional_data'].pin_memory().to(gpu_id, non_blocking=True)
#                  batch_fwd.append((data,))
#        start_time = time.time()
#        input_dict = {}
#        input_dict['is_sampling']=True
#        input_dict['batch_size']=self.test_conf.batch_size
#        input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
#        input_dict['conditional_data'] = data['conditional_data']
#        input_dict['expanded_search'] = False
#        conditional_data.extend(data['conditional_data'].cpu().numpy().tolist())
#        A_tmp, graphs_node_list = model(input_dict)
##        gen_run_time += [time.time() - start_time]
##        self.A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
##        num_nodes_pred += [aa.shape[0] for aa in A_tmp]
#        gen_run_time += [time.time() - start_time]
#        self.A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
#        num_nodes_pred += [aa.shape[0] for aa in A_tmp]
#        self.graphs_gen_nodes.extend(graphs_node_list)
#
#    logger.info('Average test time per mini-batch = {}'.format(
#      np.mean(gen_run_time)))

    node_label_list = self.graphs_gen_nodes
    self.valid_graphs = []
    self.accuracy_list = []
    self.trainable_parameters_list = []
    self.training_time_list = []
    for index in range(self.num_test_gen):
        
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
                additional_details = []
                if graph.number_of_nodes() == self.config.model.max_num_nodes:
                    for node_idx in range(graph.number_of_nodes()):
                          graph.add_node(node_idx,
                           label=operations_encode[ops[node_idx]])
                    self.valid_graphs.append(graph)
                    self.accuracy_list.append(data['test_accuracy'])
                    self.trainable_parameters_list.append(data['trainable_parameters'])
                    self.training_time_list.append(data['training_time'])
                    additional_details.append(f"Trainable parameters {data['trainable_parameters']}")
                    additional_details.append(f"Training time {data['training_time']}")
                    additional_details.append(f"Accuracy {data['test_accuracy']}")
#                    draw_neural_architecture(graph, ops,
#                        file_name=os.path.join(self.config.save_dir,
#                        f"architecture_{index}.svg"), additional_details=additional_details)
        except OutOfDomainError as OOD:
            #Todo
            #save invalid matrix and investigate
            print(OOD)
        except ValueError:
            pass
    if len(self.accuracy_list) > 0:
        print('Generated Graphs - Test Accuracy Statistics')
        print('Mean ', np.mean(self.accuracy_list))
        print('Standard Deviation ', np.std(self.accuracy_list))
        print('Maximum ', np.max(self.accuracy_list))
        print('Minimum ', np.min(self.accuracy_list))
        print('Number of valid graphs ', len(self.accuracy_list))
    
        self.search_gen_stats.append([np.mean(self.accuracy_list),
                                        np.std(self.accuracy_list),
                                        np.max(self.accuracy_list),
                                        np.min(self.accuracy_list)])
    
        print('Generated Graphs - Model Trainable Parameters Statistics')
        print('Mean ', np.mean(self.trainable_parameters_list))
        print('Standard Deviation ', np.std(self.trainable_parameters_list))
        print('Maximum ', np.max(self.trainable_parameters_list))
        print('Minimum ', np.min(self.trainable_parameters_list))
    
        print('Generated Graphs - Model Training Time Statistics')
        print('Mean ', np.mean(self.training_time_list))
        print('Standard Deviation ', np.std(self.training_time_list))
        print('Maximum ', np.max(self.training_time_list))
        print('Minimum ', np.min(self.training_time_list))
    ### Evaluate Generated Graphs
#    structure_evaluation_metrics = evaluate_metrics(self.config, A_pred, graphs_gen_nodes, self.graphs_train, self.graphs_dev, self.graphs_test)
#    
#    logger.info("Test MMD scores of #nodes/degree/clustering/4orbits/spectral/NSPDK are = {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(structure_evaluation_metrics['test']['num_nodes'], 
#                structure_evaluation_metrics['test']['node_degree'], 
#                structure_evaluation_metrics['test']['node_clustering'], 
#                structure_evaluation_metrics['test']['graph_4orbits'],
#                structure_evaluation_metrics['test']['graph_spectral'],
#                structure_evaluation_metrics['test']['NSPDK']))
#    
#    logger.info("Dev MMD scores of #nodes/degree/clustering/4orbits/spectral/NSPDK are = {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(structure_evaluation_metrics['dev']['num_nodes'], 
#                structure_evaluation_metrics['dev']['node_degree'], 
#                structure_evaluation_metrics['dev']['node_clustering'], 
#                structure_evaluation_metrics['dev']['graph_4orbits'],
#                structure_evaluation_metrics['dev']['graph_spectral'],
#                structure_evaluation_metrics['test']['NSPDK']))
#    
#    logger.info("Generated Graphs #self-loops/isolated_nodes/invalid_nn are = {}/{}/{}".format(structure_evaluation_metrics['self_loops'],
#                structure_evaluation_metrics['isolated_nodes'],
#                structure_evaluation_metrics['invalid_nn']))
#    
#    logger.info("The uniqueness of the generated graphs:")
#    for key in sorted(structure_evaluation_metrics['uniqueness']):
#        logger.info("{}: {}".format(key, round(structure_evaluation_metrics['uniqueness'][key], 2)))
#    
#    logger.info("The novelty of the generated graphs:")
#    for key in sorted(structure_evaluation_metrics['novelty']):
#        logger.info("{}: {}".format(key, round(structure_evaluation_metrics['novelty'][key], 2)))
#        
#    pickle.dump(structure_evaluation_metrics, open(os.path.join(self.config.save_dir, 'structure_evaluation_metrics.p'), 'wb'))
#    ### Visualize Generated Graphs

#    visualize_graphs(self.config, self.A_pred, self.graphs_train)

  def targeted_search(self):
    
    #total_elapsed_time
    #loop
    #to do
    #set model and data selection dynamically
    for _ in range(100):
        #exploit
        #start time
        #expolitation
        self.train()
        #end time
        #explore
        #start time
        #targeted search with fixed number of nodes
        self.test()
        #targeted search with more number of nodes
        self.expanded_search()
        #end time
        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
  
        #self.get_new_architectures()
        
        self.update_dataset()
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5

    #for experiment analysis
    dataset_column_names = list(self.dataset[0].keys())[1:]
    stats_column_names  = ['Mean', 'Standard_Deviation', 'Max', 'Min']
    
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_gen_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_train_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.dataset, dataset_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.expanded_dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.expanded_dataset, dataset_column_names, file_name)
    
    self.search_gen_stats = list()
    self.search_train_stats = list()
    self.accuracy_list = list()
#    free_up_cache()
#    print(self.config.model.max_num_nodes)
#    print(self.config.model.max_num_nodes.test)
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes+1: self.expanded_dataset})
    self.dataset = self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes+1]
    self.expanded_dataset = list()
    self.config.model.max_num_nodes = self.config.model.max_num_nodes+1
#    self.max_num_nodes = self.config.model.max_num_nodes   
    self.train_conf.is_resume = False
    self.train_conf.max_epoch = 100
    
    for _ in range(100):
        #exploit
        #start time
        #expolitation
        self.update_dataset()
        self.train()
        #end time
        #explore
        #start time
        #targeted search with fixed number of nodes
        self.test()
        #targeted search with more number of nodes
        self.expanded_search()
        #end time
        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
  
        #self.get_new_architectures()
        
        #self.update_dataset()
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5
        

    #for experiment analysis
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_gen_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_train_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.dataset, dataset_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.expanded_dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.expanded_dataset, dataset_column_names, file_name)

    self.search_gen_stats = list()
    self.search_train_stats = list()
    self.accuracy_list = list()
#    free_up_cache()
#    print(self.config.model.max_num_nodes)
#    print(self.config.model.max_num_nodes.test)
    
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes+1: self.expanded_dataset})
    self.dataset = self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes+1]
#    self.dataset = self.expanded_dataset
    self.expanded_dataset = list()
    self.config.model.max_num_nodes = self.config.model.max_num_nodes+1
#    self.max_num_nodes = self.config.model.max_num_nodes   
    self.train_conf.is_resume = False
    self.train_conf.max_epoch = 100
    for _ in range(100):
        #exploit
        #start time
        #expolitation
        self.update_dataset()
        self.train()
        #end time
        #explore
        #start time
        #targeted search with fixed number of nodes
        self.test()
        #targeted search with more number of nodes
       # self.expanded_search()
        #end time
        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
  
        #self.get_new_architectures()
        
        #self.update_dataset()
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5
        

    #for experiment analysis
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_gen_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_train_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.dataset, dataset_column_names, file_name)
    
    print(self.config.model.max_num_nodes)
#    print(self.config.model.max_num_nodes.test)
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})
    
    self.config.model.tree_level  += 1
    self.config.num_categories = 3
    
    
    
    self.search_gen_stats = list()
    self.search_train_stats = list()
    self.accuracy_list = list()
    
    self.expanded_dataset = list()
    self.config.model.max_num_nodes = 5
#    self.max_num_nodes = self.config.model.max_num_nodes
    self.train_conf.is_resume = False
    self.train_conf.max_epoch = 100
    
#    print(self.all_datasets.keys())
#    print(self.all_datasets[0].keys())
    
    self.all_datasets.update({self.config.model.tree_level: self.all_datasets[self.config.model.tree_level-1]})
    self.dataset = self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes]

#    print(self.all_datasets.keys())
#    print(self.all_datasets[1].keys())
#    
#    print(self.all_datasets.testing)
    
    for _ in range(100):
        #exploit
        #start time
        #expolitation
        self.update_dataset()
        self.train()
        #end time
        #explore
        #start time
        #targeted search with fixed number of nodes
        self.test()
        #targeted search with more number of nodes
        self.expanded_search()
        #end time
        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
  
        #self.get_new_architectures()
        
        
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5

    #for experiment analysis
    dataset_column_names = list(self.dataset[0].keys())[1:]
    stats_column_names  = ['Mean', 'Standard_Deviation', 'Max', 'Min']
    
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_gen_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_train_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.dataset, dataset_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.expanded_dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.expanded_dataset, dataset_column_names, file_name)
    
    self.search_gen_stats = list()
    self.search_train_stats = list()
    self.accuracy_list = list()
#    free_up_cache()
    print(self.config.model.max_num_nodes)
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})
    self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes+1].extend(self.expanded_dataset)
    self.dataset = self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes+1]
    print(self.dataset[0]['graph'].number_of_nodes())
    print(self.all_datasets.keys())
    print(self.all_datasets[1].keys())
    self.expanded_dataset = list()
    self.config.model.max_num_nodes = self.config.model.max_num_nodes+1
#    self.max_num_nodes = self.config.model.max_num_nodes   
    self.train_conf.is_resume = False
    self.train_conf.max_epoch = 100
    
    for _ in range(100):
        #exploit
        #start time
        #expolitation
        self.update_dataset()
        self.config.model.max_num_nodes = 6
#        self.max_num_nodes = 6
        self.train()
        #end time
        #explore
        #start time
        #targeted search with fixed number of nodes
        self.test()
        #targeted search with more number of nodes
        self.expanded_search()
        #end time
        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
  
        #self.get_new_architectures()
        
        #self.update_dataset()
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5
        

    #for experiment analysis
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_gen_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_train_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.dataset, dataset_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.expanded_dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_expanded_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.expanded_dataset, dataset_column_names, file_name)

    self.search_gen_stats = list()
    self.search_train_stats = list()
    self.accuracy_list = list()
#    free_up_cache()
    print(self.config.model.max_num_nodes)
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})
    self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes+1].extend(self.expanded_dataset)
    self.dataset = self.all_datasets[self.config.model.tree_level][self.config.model.max_num_nodes+1]
    
    self.dataset = self.expanded_dataset
    self.expanded_dataset = list()
    self.config.model.max_num_nodes = self.config.model.max_num_nodes+1
#    self.max_num_nodes = self.config.model.max_num_nodes   
    self.train_conf.is_resume = False
    self.train_conf.max_epoch = 100
    for _ in range(100):
        #exploit
        #start time
        #expolitation
        self.update_dataset()
        self.config.model.max_num_nodes = 7
#        self.max_num_nodes = 7
        self.train()
        #end time
        #explore
        #start time
        #targeted search with fixed number of nodes
        self.test()
        #targeted search with more number of nodes
       # self.expanded_search()
        #end time
        
        np.random.seed(self.train_conf.max_epoch)
        torch.manual_seed(self.train_conf.max_epoch)
        torch.cuda.manual_seed_all(self.train_conf.max_epoch)
  
        #self.get_new_architectures()
        
        #self.update_dataset()
        
        self.train_conf.is_resume = True
        self.train_conf.resume_epoch = self.train_conf.max_epoch
        self.train_conf.max_epoch = self.train_conf.resume_epoch + 5
        

    #for experiment analysis
    tmp_path = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_gen_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_gen_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_gen_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.search_train_stats, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'search_train_stats_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_search_stats_csv(self.search_train_stats, stats_column_names, file_name)
    
    tmp_path = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.p'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    pickle.dump(self.dataset, open(tmp_path, 'wb'))
    file_name = os.path.join(self.config.save_dir, 'final_dataset_{}_{}.csv'.format(self.config.model.max_num_nodes, self.config.model.tree_level))
    output_dataset_csv(self.dataset, dataset_column_names, file_name)
    
    
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})

  def expanded_search(self):

    ### load model
    model = eval(self.model_conf.name)(self.config)
    model_file = os.path.join(self.config.save_dir, "model_snapshot_search_{}_{}.pth".format(self.config.model.max_num_nodes, self.config.model.tree_level))
    load_model(model, model_file, self.device)

    if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)
    
    model.eval()

    ### Generate Graphs
    A_pred = []

    num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))


#    conditional_data = np.concatenate((np.float32(np.random.choice([0, 1], 
#                                       size=(self.test_conf.batch_size, 2))), 
#                                       np.ones((self.test_conf.batch_size,1))), axis=1)
    if self.config.model.tree_level ==  0:
        conditional_data = np.eye(self.config.num_categories)
        conditional_data = conditional_data[np.random.choice(range(self.config.num_categories), size=(self.test_conf.batch_size))]
    else:
        conditional_data = np.float32(np.random.choice([0, 1], 
                                           size=(self.test_conf.batch_size, self.config.num_categories)))
#        conditional_data = np.concatenate((np.float32(np.random.choice([0, 1], 
#                                           size=(self.test_conf.batch_size, 2))), 
#                                           np.ones((self.test_conf.batch_size,1))), axis=1)
    
    conditional_data = torch.from_numpy(conditional_data).float()
    
#    conditional_data = np.eye(self.num_supernode_categories)
#    conditional_data = conditional_data[np.random.choice(range(self.num_supernode_categories), size=(self.test_conf.batch_size))]
#    conditional_data = torch.from_numpy(conditional_data).float()
    gen_run_time = []
    graphs_gen_nodes = []
    for ii in tqdm(range(num_test_batch)):
        with torch.no_grad():
          start_time = time.time()
          input_dict = {}
          input_dict['is_sampling']=True
          input_dict['batch_size']=self.test_conf.batch_size
          input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
          input_dict['conditional_data'] = conditional_data
          input_dict['expanded_search'] = True
          A_tmp, graphs_node_list = model(input_dict)
          gen_run_time += [time.time() - start_time]
          A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
          graphs_gen_nodes.append(graphs_node_list)

    logger.info('Average test time per mini-batch = {}'.format(
      np.mean(gen_run_time)))
    
    
    insert_node_pos = np.random.choice(range(2, self.config.model.max_num_nodes-1))
    A_pred_expanded  = list()
    #random local replication strategy
    for aa in A_pred:
        new_matrix = np.insert(aa, insert_node_pos, 0, axis=1)
        new_matrix = np.insert(new_matrix, insert_node_pos, 0, axis=0)
        new_matrix[insert_node_pos, :] = new_matrix[insert_node_pos+1, :]
        new_matrix[:, insert_node_pos] = new_matrix[:, insert_node_pos+1]
        A_pred_expanded.append(new_matrix)

#    new_graphs_gen = [nx.from_numpy_matrix(aa) for aa in A_pred_expanded]
        
    node_label_list = [graphs_gen_nodes[index].cpu().numpy().astype(np.int32).tolist() for index in range(len(graphs_gen_nodes))]

    for index in range(len(node_label_list[0])):
        node_label_list[0][index].insert(insert_node_pos, np.random.choice(range(1, 4)))
        

    new_graphs_gen_nodes = node_label_list[0]
#    print(new_graphs_gen_nodes)
    #Engergy density or influencer replication strategy
    all_node_degree_list = list()
    for aa in A_pred:
        graph = nx.from_numpy_matrix(aa)
        node_degree_list = [(n, d) for n, d in graph.degree()]
        upper = 0
        lower = 0
        upper_index = 0
        lower_index = 0
        for index in range(self.config.model.max_num_nodes):
            if index < insert_node_pos:
                if node_degree_list[index][1] >= upper:
                    upper_index = node_degree_list[index][0]
                    upper = node_degree_list[index][1]
            else:
                if node_degree_list[index][1] >= lower:
                    lower_index = node_degree_list[index][0] + 1
                    lower = node_degree_list[index][1]
        new_matrix = np.insert(aa, insert_node_pos, 0, axis = 1)
        new_matrix = np.insert(new_matrix, insert_node_pos, 0, axis = 0)
        new_matrix[insert_node_pos, upper_index] = 1.0
        new_matrix[lower_index, insert_node_pos] = 1.0
        A_pred_expanded.append(new_matrix)
        all_node_degree_list.append(node_degree_list)
        
    new_graphs_gen = [nx.from_numpy_matrix(aa) for aa in A_pred_expanded]
        
    node_label_list = [graphs_gen_nodes[index].cpu().numpy().astype(np.int32).tolist() for index in range(len(graphs_gen_nodes))]

    for index in range(len(node_label_list[0])):
        node_degree = [d for n, d in all_node_degree_list[index]]
        node_label_list[0][index].insert(insert_node_pos, node_label_list[0][index][np.argmax(node_degree)])
#        node_label_list[0][index].insert(insert_node_pos, np.random.choice(range(1, 4)))
        
#    print(node_label_list[0])
    new_graphs_gen_nodes.extend(node_label_list[0])
#    print(new_graphs_gen_nodes)
    
    

    #random rearrangement of nodes except for input and output
    #preserve structure
    #Did not work
#    node_label_list = [graphs_gen_nodes[index].cpu().numpy().astype(np.int32).tolist() for index in range(len(graphs_gen_nodes))]
#
#    for index in range(len(node_label_list[0])):
#        for node_index in range(1, self.max_num_nodes-1):
#            node_label_list[0][index].insert(node_index, np.random.choice(range(1, 4)))
#    new_graphs_gen_nodes = node_label_list[0]
    
    
    
    accuracy_list = []
    trainable_parameters_list = []
    training_time_list = []
    for index in range(len(new_graphs_gen)):
        
        graph = new_graphs_gen[index]
        matrix = np.triu(nx.to_numpy_array(graph).astype(np.int32))

        operations_encode = {'input':0, 'conv3x3-bn-relu':1, 'conv1x1-bn-relu':2, 'maxpool3x3':3, 'output':4}
        operations_decode = {0:'input', 1:'conv3x3-bn-relu', 2:'conv1x1-bn-relu',
            3:'maxpool3x3', 4:'output'}

        ops = [operations_decode[node_index] for node_index in new_graphs_gen_nodes[index]]
        try:
            spec = api.ModelSpec(matrix=matrix, ops=ops)
        
            if nasbench.is_valid(spec):
                graph = nx.Graph(csr_matrix(spec.matrix))
                ops = spec.ops
                data = nasbench.query(spec, epochs=108)
                if graph.number_of_nodes() == self.config.model.max_num_nodes+1:
                    new_expanded_sample = dict()
                    for node_idx in range(graph.number_of_nodes()):
                          graph.add_node(node_idx,
                           label=operations_encode[ops[node_idx]])
                          
                    new_expanded_sample['graph'] = graph
                    new_expanded_sample['test_accuracy'] = data['test_accuracy']
                    new_expanded_sample['total_parameters'] = data['trainable_parameters']
                    new_expanded_sample['total_training_time'] = data['training_time']
                    self.expanded_dataset.append(new_expanded_sample)
                    accuracy_list.append(data['test_accuracy'])
                    trainable_parameters_list.append(data['trainable_parameters'])
                    training_time_list.append(data['training_time'])
        except:
            #todo
            #save invalid matrix and investigate
            print('invalid architecture')
#    self.all_datasets = {self.config.model.tree_level: {self.config.model.max_num_nodes+1: self.expanded_dataset}}
#    print(new_graphs_gen_nodes.testing)
    if len(accuracy_list) > 0:
        print('############################################')
        print('         Expanded Targeted Search           ')
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
        print('############################################')
#    print(accuracy_list.testing)
  def update_dataset(self):
    
        #dataset modification
        #parameterize dataset creation
        #automate the thresholding based on new architectures discovered
        #slice the dataset into new range of high and low - accuracy
        #slice the dataset into new range of high and low - parameters
        #add new graphs with new architectures
        #remove old graphs with low performing architectures
        #test set conditioned for high performing architectures
    
    #estimate threshold
    #remove old data
    #add new data
    
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
#        self.dataset.pop(index)

    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.dataset)
      print('shuffled')

    self.npr = np.random.RandomState(self.seed)
    self.npr.shuffle(self.dataset)
    
    self.num_graphs = len(self.dataset)
    if self.num_graphs < 100:
        self.num_train = 20
        self.num_dev = 5
    else:
        self.num_train = 50
        self.num_dev = 10
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = self.config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))
    
    print('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))
    for data_item in self.dataset:
        self.graphs.append(data_item['graph'])
        self.total_parameters.append(data_item['total_parameters'])
        self.total_training_time.append(data_item['total_training_time'])
        self.test_accuracy.append(data_item['test_accuracy'])
    
    self.all_datasets[self.config.model.tree_level].update({self.config.model.max_num_nodes: self.dataset})
    
    estimator = preprocessing.KBinsDiscretizer(n_bins=self.config.num_categories, 
                                                 encode='onehot', strategy='uniform',
                                                 dtype=np.float32)
    
    self.supernode_categories = estimator.fit_transform(np.array(self.test_accuracy).reshape(-1, 1))
    self.supernode_categories = csr_matrix(self.supernode_categories).todense().tolist()
    
    self.graphs_train = self.graphs[:self.num_train]
    self.total_parameters_train = self.total_parameters[:self.num_train]
    self.total_training_time_train = self.total_training_time[:self.num_train]
    self.test_accuracy_train = self.test_accuracy[:self.num_train]
    self.supernode_categories_train =  self.supernode_categories[:self.num_train]

    self.graphs_dev = self.graphs[:self.num_dev]
    self.total_parameters_dev = self.total_parameters[:self.num_dev]
    self.total_training_time_dev = self.total_training_time[:self.num_dev]
    self.test_accuracy_dev = self.test_accuracy[:self.num_dev]
    self.supernode_categories_dev =  self.supernode_categories[:self.num_dev]
    
    self.graphs_test = self.graphs[self.num_train:]
    self.total_parameters_test = self.total_parameters[self.num_train:]
    self.total_training_time_test = self.total_training_time[self.num_train:]
    self.test_accuracy_test = self.test_accuracy[self.num_train:]
    self.supernode_categories_test =  self.supernode_categories[self.num_train:]

    print('Training Set - Test Accuracy Statistics')
    print('Mean ', np.mean(self.test_accuracy_train))
    print('Standard Deviation ', np.std(self.test_accuracy_train))
    print('Maximum ', np.max(self.test_accuracy_train))
    print('Minimum ', np.min(self.test_accuracy_train))
    print('Number of valid graphs ', len(self.test_accuracy_train))
    
    

#    training_ds_operations_freq = dict()
#    for G in self.graphs_train:
#        for layer in [node['label'] for idx, node in G.nodes.data()]:
#            if layer in training_ds_operations_freq.keys():
#                training_ds_operations_freq[layer] +=1
#            else:
#                training_ds_operations_freq[layer] = 1
#    
#    for index in range(5):
#        if index not in training_ds_operations_freq.keys():
#            training_ds_operations_freq[index] = 0.0001
#    self.class_weights = [training_ds_operations_freq[0],
#        training_ds_operations_freq[1],
#        training_ds_operations_freq[2],
#        training_ds_operations_freq[3],
#        training_ds_operations_freq[4]]
#
#    self.config.class_weights = self.class_weights



    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
#    self.max_num_nodes = len(self.num_nodes_pmf_train)
#    self.config.max
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()

  def optimize(self):    
    # nasbench.config['num_repeats'] = 1

    # freeup cache
    # free_up_cache()

    ### load model
    model = eval(self.model_conf.name)(self.config)
    model_file = os.path.join(self.test_conf.test_model_dir, self.test_conf.test_model_name)
    load_model(model, model_file, self.device)
    


    if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

    model.eval()

    #display model details
    if self.config.model.display_detailed_model:
        display_model(model)

    #edge_predictor optimization
    input_dict = {}
    input_dict['is_sampling']=True
    input_dict['batch_size']=self.test_conf.batch_size
    input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
    edge_predictor_layer_name = 'module.output_theta.4.weight'
    model.load_state_dict(evolution(model, edge_predictor_layer_name, input_dict, self.config))

#    snapshot(model.module if self.use_gpu else model, model.optimizer, self.config, 0, scheduler=model.scheduler)


    ### Generate Graphs
    A_pred = []
    num_nodes_pred = []
    num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

    gen_run_time = []
    graphs_gen_nodes = []
    for ii in tqdm(range(num_test_batch)):
        with torch.no_grad():
          start_time = time.time()
          A_tmp, graphs_node_list = model(input_dict)
          gen_run_time += [time.time() - start_time]
          A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
          num_nodes_pred += [aa.shape[0] for aa in A_tmp]
          graphs_gen_nodes.append(graphs_node_list)

    logger.info('Average test time per mini-batch = {}'.format(
      np.mean(gen_run_time)))

    ### Evaluate Generated Graphs
    evaluate_metrics(self.config, A_pred, graphs_gen_nodes, self.graphs_train, self.graphs_dev, self.graphs_test)
    
    ### Visualize Generated Graphs
    visualize_graphs(self.config, A_pred, self.graphs_train)
    