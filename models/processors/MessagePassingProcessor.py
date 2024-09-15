import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, FaceToEdge, Cartesian, Distance

from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

from ..encoders.MLPEncoder import MLP
from .utils.graph_process import decompose_graph, copy_geometric_data
from .utils.blocks import EdgeBlock, NodeBlock

class MessagePassingProcessor(MessagePassing):
    def __init__(self, mlp_config, message_passing_steps=8, shared_mlp=True):
        # Initialize MessagePassing with the 'mean' aggregation scheme
        super(MessagePassingProcessor, self).__init__(aggr='mean')  
        self.message_passing_steps = message_passing_steps
        self.shared_mlp = shared_mlp
        
        # Initialize shared or step-wise MLPs for node and edge updates
        if shared_mlp:
            self.node_mlp = MLP(**mlp_config)
            self.edge_mlp = MLP(128,64)
        else:
            self.node_mlps = nn.ModuleList([MLP(**mlp_config) for _ in range(message_passing_steps)])
            self.edge_mlps = nn.ModuleList([MLP(128,64) for _ in range(message_passing_steps)])
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Add self-loops to ensure each node has at least one message to pass
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        for i in range(self.message_passing_steps):
            if self.shared_mlp:
                # Propagate messages through the graph
                x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            else:
                # Use step-specific MLPs for propagation
                x = self.propagate(edge_index, x=x, edge_attr=edge_attr, mlp_step=i)
        
        data.x = x
        return data

    def message(self, x_j, edge_attr, mlp_step=None):
        """Construct the message from neighboring nodes and edge attributes."""
        if edge_attr is not None:
            msg = torch.cat([x_j, edge_attr], dim=-1)
        else:
            msg = x_j
        
        if not self.shared_mlp and mlp_step is not None:
            msg = self.edge_mlps[mlp_step](msg)
        else:
            msg = self.edge_mlp(msg)
        
        return msg
    
    def update(self, aggr_out, x):
        """Node update function with residual connection."""
        return x + aggr_out

    def edge_update(self, edge_index, x, edge_attr, mlp_step=None):
        """Edge update function that updates edge features based on neighboring nodes."""
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)  # Concatenate features of the two endpoints of each edge

        if not self.shared_mlp and mlp_step is not None:
            edge_attr = self.edge_mlps[mlp_step](edge_features)
        else:
            edge_attr = self.edge_mlp(edge_features)
        
        return edge_attr

# -------------------------------------------------------------------------------------------------
# original implementation for meshgraphnet

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module

class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)

class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()


        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)

class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, output_size = 2, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)

    def forward(self, graph):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded
# -------------------------------------------------------------------------------------------------
