from __future__ import (division, print_function)
import os
import time

import numpy as np

import pickle
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from nasbench.lib import model_metrics_pb2
import base64
import json
from nasbench import api


import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.vis_helper import draw_neural_architecture, visualize_graphs
from utils.vis_helper import draw_graph_list, draw_graph_list_separate 


from utils.data_parallel import DataParallel
from scipy.sparse import csr_matrix
from utils.nas_evaluation_helper import *

from nas_101_api.nas101_model import *
from nas_101_api.nas101_model import *
# from runner.geometric_processing import *

#from utils.runner_helper import print_info, free_up_cache
from utils.runner_helper import compute_edge_ratio, get_graph
from utils.runner_helper import evaluate_metrics, evaluate, validation_metrics
from utils.runner_helper import evolution
from utils.runner_helper import display_model
from utils.dist_helper import compute_mmd, gaussian_emd
from nasbench import api
from nasbench.lib.model_spec import is_upper_triangular


from sklearn import metrics
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

logger = get_logger('exp_logger')
__all__ = ['GranRunner_SplineCNN', 'compute_edge_ratio', 'get_graph', 'evaluate']

class GranRunner_SplineCNN(object):

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
    
    torch.autograd.set_detect_anomaly(True)
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


    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.dataset)
    
    self.graphs = list()
    self.total_parameters = list()
    self.total_training_time = list()
    self.test_accuracy = list()
#    self.node_in_edges = list()
    
    for data_item in self.dataset:
        self.graphs.append(data_item['graph'])
        self.total_parameters.append(data_item['total_parameters'])
        self.total_training_time.append(data_item['total_training_time'])
        self.test_accuracy.append(data_item['test_accuracy'])
#        self.node_in_edges.append(data_item['node_in_edges'])

    self.graphs_train = self.graphs[:self.num_train]
    self.total_parameters_train = self.total_parameters[:self.num_train]
    self.total_training_time_train = self.total_training_time[:self.num_train]
    self.test_accuracy_train = self.test_accuracy[:self.num_train]
#    self.node_in_edges_train = self.node_in_edges[:self.num_train]
    
    self.graphs_dev = self.graphs[:self.num_dev]
    self.total_parameters_dev = self.total_parameters[:self.num_dev]
    self.total_training_time_dev = self.total_training_time[:self.num_dev]
    self.test_accuracy_dev = self.test_accuracy[:self.num_dev]
#    self.node_in_edges_dev = self.node_in_edges[:self.num_dev]
    
    self.graphs_test = self.graphs[self.num_train:]
    self.total_parameters_test = self.total_parameters[self.num_train:]
    self.total_training_time_test = self.total_training_time[self.num_train:]
    self.test_accuracy_test = self.test_accuracy[self.num_train:]
#    self.node_in_edges_test = self.node_in_edges[self.num_train:]

    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
    self.max_num_nodes = len(self.num_nodes_pmf_train)
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
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, self.total_parameters_train, self.total_training_time_train, self.test_accuracy_train, tag='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

#    val_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_dev, self.total_parameters_dev, self.total_training_time_dev, self.test_accuracy_dev, tag='train')
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

    best_f1_score = 0
    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
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

#        clip_grad_norm_(model.parameters(), 5.0e-0)
        optimizer.step()
        lr_scheduler.step()
        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)
#        avg_train_acc /= float(self.dataset_conf.num_fwd_pass)

        # reduce
        train_loss = float(avg_train_loss.data.cpu().numpy())
      classifier_train_metrics = validation_metrics(pred_label = pred_label_list, actual_label = actual_label_list)
#      if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
#        logger.info("Training Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, train_loss, classifier_train_metrics['accuracy'], classifier_train_metrics['precision'], classifier_train_metrics['recall'], classifier_train_metrics['f1_score']))
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
    
#      results['train_loss'] += [train_loss]
#      results['train_acc'] += [classifier_train_metrics['accuracy']]
#      results['train_f1_score'] += [classifier_train_metrics['f1_score']]
#      results['val_loss'] += [val_loss]
#      results['val_acc'] += [classifier_validation_metrics['accuracy']]
#      results['val_f1_score'] = [classifier_validation_metrics['f1_score']]
#      results['train_step'] += [iter_count]
      print("Training Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, train_loss, classifier_train_metrics['accuracy'], classifier_train_metrics['precision'], classifier_train_metrics['recall'], classifier_train_metrics['f1_score']))
#      print("Validation Loss @ epoch {:04d}; Loss: {:.4f}; Accuracy: {:.4f}; Precision: {:.4f}; Recall: {:.4f}; F1 Score: {:.4f}".format(epoch + 1, val_loss, classifier_validation_metrics['accuracy'], classifier_validation_metrics['precision'], classifier_validation_metrics['recall'], classifier_validation_metrics['f1_score']))
      if epoch > 250 and classifier_train_metrics['f1_score'] > best_f1_score:
          best_f1_score = classifier_train_metrics['f1_score']
          snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
#      if epoch > 250 and classifier_validation_metrics['f1_score'] > best_f1_score:
#          best_f1_score = classifier_validation_metrics['f1_score']
#          snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)

      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
      
#      if avg_val_loss < 0.1:
#        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
#        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
#    self.writer.close()

    return 1

  def test(self):

    ### load model
    model = eval(self.model_conf.name)(self.config)
    model_file = os.path.join(self.test_conf.test_model_dir, self.test_conf.test_model_name)
    load_model(model, model_file, self.device)
#
#    # model.output_theta.register_forward_hook(print_info)
#    # model.decoder_input.register_forward_hook(print_info)
#
    if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

#    #display model details
    if self.config.model.display_detailed_model:
        display_model(model)

    model.eval()

    ### Generate Graphs
    A_pred = []
    num_nodes_pred = []
    num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

    
    conditional_data = np.concatenate((np.float32(np.random.choice([0, 1], 
                                       size=(self.test_conf.batch_size, 2))), 
                                       np.ones((self.test_conf.batch_size,1))), axis=1)
    conditional_data = torch.from_numpy(conditional_data).float()
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
          A_tmp, graphs_node_list = model(input_dict)
          gen_run_time += [time.time() - start_time]
          A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
          num_nodes_pred += [aa.shape[0] for aa in A_tmp]
          graphs_gen_nodes.append(graphs_node_list)

    logger.info('Average test time per mini-batch = {}'.format(
      np.mean(gen_run_time)))
    
    ## Evaluate Generated Graphs
    structure_evaluation_metrics = evaluate_metrics(self.config, A_pred, graphs_gen_nodes, self.graphs_train, self.graphs_dev, self.graphs_test)
    
    logger.info("Test MMD scores of #nodes/degree/clustering/4orbits/spectral/NSPDK are = {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(structure_evaluation_metrics['test']['num_nodes'], 
                structure_evaluation_metrics['test']['node_degree'], 
                structure_evaluation_metrics['test']['node_clustering'], 
                structure_evaluation_metrics['test']['graph_4orbits'],
                structure_evaluation_metrics['test']['graph_spectral'],
                structure_evaluation_metrics['test']['NSPDK']))
    
    logger.info("Dev MMD scores of #nodes/degree/clustering/4orbits/spectral/NSPDK are = {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(structure_evaluation_metrics['dev']['num_nodes'], 
                structure_evaluation_metrics['dev']['node_degree'], 
                structure_evaluation_metrics['dev']['node_clustering'], 
                structure_evaluation_metrics['dev']['graph_4orbits'],
                structure_evaluation_metrics['dev']['graph_spectral'],
                structure_evaluation_metrics['test']['NSPDK']))
    
    logger.info("Generated Graphs #self-loops/isolated_nodes/invalid_nn are = {}/{}/{}".format(structure_evaluation_metrics['self_loops'],
                structure_evaluation_metrics['isolated_nodes'],
                structure_evaluation_metrics['invalid_nn']))
    
    logger.info("The uniqueness of the generated graphs:")
    for key in sorted(structure_evaluation_metrics['uniqueness']):
        logger.info("{}: {}".format(key, round(structure_evaluation_metrics['uniqueness'][key], 2)))
    
    logger.info("The novelty of the generated graphs:")
    for key in sorted(structure_evaluation_metrics['novelty']):
        logger.info("{}: {}".format(key, round(structure_evaluation_metrics['novelty'][key], 2)))
        
    pickle.dump(structure_evaluation_metrics, open(os.path.join(self.config.save_dir, 'structure_evaluation_metrics.p'), 'wb'))
    ### Visualize Generated Graphs
    visualize_graphs(self.config, A_pred, self.graphs_train)
