import os
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

#import pydot

def visualize_graphs(config, A_pred, graphs_train): 
    graphs_gen = A_pred #[nx.from_numpy_matrix(aa) for aa in A_pred]
    is_vis = config.test.is_vis
    better_vis = config.test.better_vis
    num_vis = config.test.num_vis
    vis_num_row = config.test.vis_num_row
    is_single_plot = config.test.is_single_plot
    test_conf = config.test
    block_size = config.model.block_size
    stride = config.model.sample_stride
    
    if is_vis:
      num_col = vis_num_row
      num_row = int(np.ceil(num_vis / num_col))
      test_epoch = test_conf.test_model_name
      test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
      save_name = os.path.join(config.save_dir, '{}_gen_graphs_epoch_{}_block_{}_stride_{}.png'.format(config.test.test_model_name[:-4], test_epoch, block_size, stride))

      # remove isolated nodes for better visulization
      graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:num_vis]]

      if better_vis:
        for gg in graphs_pred_vis:
          gg.remove_nodes_from(list(nx.isolates(gg)))

      # display the largest connected component for better visualization
      vis_graphs = []
      for gg in graphs_pred_vis:
        CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        vis_graphs += [CGs[0]]

      if is_single_plot:
        draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
      else:
        draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')

      save_name = os.path.join(config.save_dir, 'train_graphs.png')

      if is_single_plot:
        draw_graph_list(
            graphs_train[:num_vis],
            num_row,
            num_col,
            fname=save_name,
            layout='spring')
      else:
        draw_graph_list_separate(
            graphs_train[:num_vis],
            fname=save_name[:-4],
            is_single=True,
            layout='spring')

def draw_neural_architecture(graph, labels_list, file_name='architecture.svg', additional_details=[]):
    dot_graph = pydot.Dot(graph_type='digraph')

    edge_list = list(graph.edges)
#    operations = {0:input, 1:'conv3x3-bn-relu', 2:'conv1x1-bn-relu',
#        3:'maxpool3x3', 4:'output'}

    def make_node(name):
        cur_node = pydot.Node(name)
        cur_node.set_shape('box')
        dot_graph.add_node(cur_node)
        return cur_node

    def make_link(a_node, b_node, label = None, width = 1, style='vee'):
        cur_edge = pydot.Edge(a_node,b_node)
        cur_edge.set_penwidth(width)
        cur_edge.set_style(style)
        if label is not None: cur_edge.set_label(label)
        dot_graph.add_edge(cur_edge)
        return cur_edge

    nodes = dict()
    for i in range(len(labels_list)):
        nodes[i] = make_node(f"Layer {i} {labels_list[i]}")

    for i in range(len(additional_details)):
        make_node(f"{additional_details[i]}")

    for edge in edge_list:
        make_link(nodes[edge[0]], nodes[edge[1]])

    dot_graph.write_svg(file_name, prog = 'dot')

def draw_graph_list(G_list,
                    row,
                    col,
                    fname='exp/gen_graph.png',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):
  plt.switch_backend('agg')
  for i, G in enumerate(G_list):
    plt.subplot(row, col, i + 1)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # plt.axis("off")

    # turn off axis label
    plt.xticks([])
    plt.yticks([])

    if layout == 'spring':
      pos = nx.spring_layout(
          G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
      pos = nx.spectral_layout(G)

    if is_single:
      # node_size default 60, edge_width default 1.5
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=node_size,
          node_color='#336699',
          alpha=1,
          linewidths=0,
          font_size=0)
      nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
    else:
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=1.5,
          node_color='#336699',
          alpha=1,
          linewidths=0.2,
          font_size=1.5)
      nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

  plt.tight_layout()
  plt.savefig(fname, dpi=300)
  plt.close()


def draw_graph_list_separate(G_list,
                    fname='exp/gen_graph',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):

  for i, G in enumerate(G_list):
    plt.switch_backend('agg')

    plt.axis("off")

    # turn off axis label
    # plt.xticks([])
    # plt.yticks([])

    if layout == 'spring':
      pos = nx.spring_layout(
          G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
      pos = nx.spectral_layout(G)

    if is_single:
      # node_size default 60, edge_width default 1.5
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=node_size,
          node_color='#336699',
          alpha=1,
          linewidths=0,
          # font_size=0 not available in all libraries
          )
      nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
    else:
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=1.5,
          node_color='#336699',
          alpha=1,
          linewidths=0.2,
          # font_size=1.5
          )
      nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.draw()
    plt.tight_layout()
    plt.savefig(fname+'_{:03d}.png'.format(i), dpi=300)
    plt.close()
