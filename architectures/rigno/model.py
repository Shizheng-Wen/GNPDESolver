import torch 
import torch.nn as nn  
import torch.nn.functional as F 
from typing import Optional, Tuple, Union,TypeVar, Any
from dataclasses import dataclass, asdict, replace, field, is_dataclass, fields
from .graph import Graph,RegionInteractionGraph
from .pair import make_pair, force_make_pair, is_pair

def shallow_asdict(obj:Any)->dict:
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    return obj






def activation_fn(name:str, activation_kwargs:dict = dict())->nn.Module:
    if name == "swish":
        return nn.SiLU()
    elif hasattr(F, name):
        return getattr(F, name)
    else:
        raise ValueError(f"Activation function {name} not found")
        

class MLP(nn.Module):
    def __init__(self, 
                 input_size:int,
                 output_size:int,
                 hidden_size:int,
                 num_layers:int = 3,
                 activation:str="swish"):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
        ])
        for _  in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.act  = activation_fn(activation)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.act(x)
        x = self.layers[-1](x)
        return x
        
    pass

class ConditionedNorm(nn.Module):
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 hidden_size:int):
        super().__init__()
        self.mlp_scale = MLP(input_size, 
                             output_size,
                             hidden_size,
                             num_layers=2,
                             activation="sigmoid")
        self.mlp_bias = MLP(input_size,
                            output_size,
                            hidden_size,
                            num_layers=2,
                            activation="sigmoid")
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.mlp_scale.layers:
            nn.init.normal_(layer.weight, std=0.01)
        for layer in self.mlp_bias.layers:
            nn.init.normal_(layer.weight, std=0.01)
    def forward(self, c, x):
        scale = 1 + c * self.mlp_scale(c)
        bias = c * self.mlp_bias(c)
        x    = x * scale + bias 
        return x

@dataclass
class AugmentedMLPConfig:
    hidden_size:int = 64
    num_layers:int = 3
    activation:str = "swish"
    use_layer_norm:bool = True
    use_conditional_norm:bool = False
    cond_norm_hidden_size:int = 4

class AugmentedMLP(nn.Module):
    mlp:MLP 
    norm:Optional[nn.LayerNorm]
    correction:Optional[ConditionedNorm]
    input_size:int 
    output_size:int
    hidden_size:int

    def __init__(self, 
                 input_size:int,
                 output_size:int,
                 hidden_size:int = 64,
                 num_layers:int = 3,
                 activation:str="swish",
                 use_layer_norm:bool = True,
                 use_conditional_norm:bool = False,
                 cond_norm_hidden_size:int = 4):
        super().__init__()

        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size 
        self.mlp = MLP(input_size, output_size, hidden_size, num_layers, activation)
        
        if use_layer_norm:
            self.norm = nn.LayerNorm(output_size)
        else:
            self.norm = None

        if use_conditional_norm:
            self.correction = ConditionedNorm(
                output_size,
                output_size, 
                cond_norm_hidden_size
            )
        else:
            self.correction = None
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, 
                x:torch.Tensor, 
                condition:Optional[float]=None
                )->torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            ND Tensor of shape [..., input_size]
        condition: Optional[float]

        Returns
        -------
        x: torch.Tensor
            ND Tensor of shape [..., output_size]
        """
        
        x = self.mlp(x)

        if self.norm is not None:
            x = self.norm(x)

        if self.correction is not None:
            assert condition is not None, f"Conditional input c should be provided"
            x = self.correction(c=condition, x=x)
        
        return x

    @classmethod 
    def from_config(cls, 
                    input_size:int,
                    output_size:int, 
                    config:AugmentedMLPConfig):
        return cls(input_size, output_size, **asdict(config))

####################
# Message Passing
####################
@dataclass
class MessagePassingLayerConfig:
    edge_fn_config:AugmentedMLPConfig = AugmentedMLPConfig()
    node_fn_config:AugmentedMLPConfig = AugmentedMLPConfig()
    aggregate:str = "mean"
    aggregate_normalization:Optional[float] = None


class MessagePassingLayer(nn.Module):
    edge_fn:AugmentedMLP
    node_fn:Union[AugmentedMLP, nn.ModuleDict]
    aggregate:str
    aggregate_normalization:Optional[float]

    node_input_size:int 
    edge_input_size:int
    def __init__(self,
                 node_input_size:Optional[Union[int,Tuple[int,int]]], 
                 edge_input_size:int,
                 node_output_size:Optional[Union[int,Tuple[int,int]]],
                 edge_output_size:int,
                 aggregate:str = "mean",
                 aggregate_normalization:Optional[float] = None,
                 edge_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                 node_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                ):
        """
        Parameters
        ----------
        node_input_size:Optional[Union[int,Tuple[int,int]]]
            Input size of node features
            if None, there is no node update 
            elif Tuple[int,int], it is bipartite
            elif int, it is homogenous
        edge_input_size:int
            Input size of edge features
        node_output_size:Optional[Union[int,Tuple[int,int]]]
            Output size of node features
            if None, there is no node update
            elif Tuple[int,int], it is bipartite
            elif int, it is homogenous
        edge_output_size:int
            Output size of edge features
        aggregate:str
            Aggregation function
        aggregate_normalization:Optional[float]
            Normalization factor for aggregation
        edge_fn_config:AugmentedMLPConfig
            Configuration for edge function
        node_fn_config:AugmentedMLPConfig
            Configuration for node function
        
        """
        
        super().__init__()


        
        if edge_input_size is not None:
            assert edge_output_size is not None, "edge_output_size should be provided"
            assert node_input_size is not None, "node_input_size should be provided"
            self.edge_fn = AugmentedMLP.from_config(
                input_size = edge_input_size + sum(make_pair(node_input_size)),
                output_size = edge_output_size,
                config = edge_fn_config
            )
        else:
            self.edge_fn = None

        if node_input_size is not None:
            if is_pair(node_input_size):
                self.node_fn = nn.ModuleList([
                    AugmentedMLP.from_config(
                        input_size  = edge_output_size + node_input_size[0],
                        output_size = make_pair(node_output_size)[0],
                        config      = node_fn_config
                    ),
                    AugmentedMLP.from_config(
                        input_size  = edge_output_size + node_input_size[1],
                        output_size = make_pair(node_output_size)[1],
                        config      = node_fn_config
                    )
                ])
            else:
                self.node_fn = AugmentedMLP.from_config(
                    input_size  = edge_output_size * 2 + node_input_size,
                    output_size = node_output_size,
                    config      = node_fn_config
                )
            
        else:
            self.node_fn = None
    
        self.aggregate = aggregate
        self.aggregate_normalization = aggregate_normalization
        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size

    @property 
    def is_bipartite(self):
        return isinstance(self.node_input_size, (list, tuple))

    def forward(self, 
                graph:Graph, 
                ndata:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], 
                edata:torch.Tensor,
                **kwargs
                )->Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Parameters
        ----------
        graph:Graph
            homogenous or bipartite graph
        ndata:torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            ND tensor of shape [..., n_nodes, node_input_size]
        edata:torch.Tensor
            ND tensor of shape [..., n_edges, edge_input_size]

        Returns
        -------
        if is_bipartite:
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            ((src_ndata, dst_ndata), edata)
            src_ndata: [..., n_src_nodes, node_output_size]
            dst_ndata: [..., n_dst_nodes, node_output_size]
            edata: [..., n_edges, edge_output_size]
        else:
            Tuple[torch.Tensor, torch.Tensor]
            (ndata, edata)
            ndata: [..., n_nodes, node_output_size]
            edata: [..., n_edges, edge_output_size]
        """
        assert graph.is_bipartite == self.is_bipartite, f"graph.is_bipartite({graph.is_bipartite}) != self.is_bipartite({self.is_bipartite})"
        assert make_pair(graph.num_nodes) == tuple(x.shape[-2] for x in make_pair(ndata)), \
            f"graph.num_nodes({make_pair(graph.num_nodes)}) != ndata.shape[-2]({tuple(x.shape[0] for x in make_pair(ndata))})"
        assert make_pair(self.node_input_size) == tuple(x.shape[-1] for x in make_pair(ndata)), \
            f"self.node_input_size({self.node_input_size}) != ndata.shape[-2]({make_pair(self.node_input_size)})"
        assert graph.num_edges      == edata.shape[-2], f"graph.num_edges({graph.num_edges}) != edata.shape[-2]({edata.shape})"
        assert self.edge_input_size == edata.shape[-1], f"self.edge_input_size({self.edge_input_size}) != edata.shape[-1]({edata.shape[1]})"
        
        if self.edge_fn is not None:

            def message_fn(src_ndata, dst_ndata, edata):
                x = torch.cat([src_ndata, dst_ndata, edata], -1)
                return self.edge_fn(x, **kwargs)
            edata = graph.message(message_fn, ndata=ndata, edata=edata)
        
        if self.node_fn is not None:

            if self.is_bipartite:

                def src_aggregate_fn(edata, ndata):
                    x = torch.cat([edata, ndata], -1)
                    return self.node_fn[0](x, **kwargs)

                def dst_aggregate_fn(edata, ndata):
                    x = torch.cat([edata, ndata], -1)
                    return self.node_fn[1](x, **kwargs)

                src_ndata, dst_ndata = graph.aggregate((src_aggregate_fn, dst_aggregate_fn), 
                                                      edata=edata, 
                                                      ndata=ndata,
                                                      reduce=self.aggregate)

                if self.aggregate_normalization is not None:
                    src_ndata = src_ndata / self.aggregate_normalization
                    dst_ndata = dst_ndata / self.aggregate_normalization

                ndata = (src_ndata, dst_ndata)

            else:

                def aggregate_fn(src_edata, dst_edata, ndata):
                    x = torch.cat([src_edata, dst_edata, ndata], -1)
                    return self.node_fn(x, **kwargs)
                
                ndata = graph.aggregate(aggregate_fn, edata=edata, ndata=ndata, reduce=self.aggregate)

                if self.aggregate_normalization is not None:
                    ndata = ndata / self.aggregate_normalization

        return ndata, edata

    @classmethod 
    def from_config(cls, 
                    node_input_size:Optional[Union[int, Tuple[int,int]]] = None,
                    edge_input_size:Optional[int] = None,
                    node_output_size:Optional[Union[int, Tuple[int,int]]] = None,
                    edge_output_size:Optional[int] = None,
                    config:MessagePassingLayerConfig = MessagePassingLayerConfig()):

        return cls(node_input_size, 
                   edge_input_size, 
                   node_output_size, 
                   edge_output_size, 
                   **shallow_asdict(config))

####################
# Feature Encoder
####################

class FeatureEncoderLayer(nn.Module):
    edge_fn:Optional[AugmentedMLP]
    node_fn:Optional[AugmentedMLP]
    def __init__(self,
                 node_input_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_input_size:Optional[int] = None,
                 node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_output_size:Optional[int] = None,
                 edge_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                 node_fn_config:AugmentedMLPConfig = AugmentedMLPConfig(),
                 ):
        super().__init__()
        
        if edge_input_size is not None:
            assert edge_output_size is not None, "edge_output_size should be provided"
            self.edge_fn = AugmentedMLP.from_config(
                input_size = edge_input_size,
                output_size = edge_output_size,
                config = edge_fn_config
            ) 
            
        else:
            self.edge_fn = None 

        if node_input_size is not None:
            if is_pair(node_input_size):
                self.node_fn = nn.ModuleList([
                    AugmentedMLP.from_config(
                        input_size = node_input_size[0],
                        output_size = make_pair(node_output_size)[0],
                        config = node_fn_config
                    ),
                    AugmentedMLP.from_config(
                        input_size = node_input_size[1],
                        output_size = make_pair(node_output_size)[1],
                        config = node_fn_config
                    )
                ])
            else:
                assert not is_pair(node_output_size), f"node_output_size:{node_output_size} should not be pair"
                self.node_fn = AugmentedMLP.from_config(
                    input_size = node_input_size,
                    output_size = node_output_size,
                    config = node_fn_config
                )
        else:
            self.node_fn = None


    def forward(self, ndata, edata, **kwargs
                )->Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Parameters
        ----------
        ndata:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            if is_bipartite:
                (src_ndata, dst_ndata)
                src_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_src_nodes, node_input_size]
                dst_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_dst_nodes, node_input_size]
            else:
                ndata:torch.Tensor
                    torch.Tensor of shape [..., n_nodes, node_input_size]
            Input node features
        edata:torch.Tensor
            Input edge features
            torch.Tensor of shape [..., n_edges, edge_input_size]
        kwargs:dict
            Additional arguments
            
        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Output node features
            if is_bipartite:
                (src_ndata, dst_ndata)
                src_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_src_nodes, node_output_size]
                dst_ndata:torch.Tensor
                    torch.Tensor of shape [..., n_dst_nodes, node_output_size]
            else:
                ndata:torch.Tensor
                    torch.Tensor of shape [..., n_nodes, node_output_size]
        torch.Tensor
            Output edge features
        """
        if self.edge_fn is not None:
            edata = self.edge_fn(edata, **kwargs)
        if self.node_fn is not None:
            if isinstance(self.node_fn, nn.ModuleList):
                assert is_pair(ndata), f"ndata should be a pair, got {ndata}"
                # is_bipartite
                src_ndata, dst_ndata = ndata
              
                src_ndata = self.node_fn[0](src_ndata, **kwargs)
                dst_ndata = self.node_fn[1](dst_ndata, **kwargs)
    
                ndata = (src_ndata, dst_ndata)
            else:
                assert not is_pair(ndata), f"ndata should not be a pair, got {ndata}"
                # not is_bipartite
                ndata = self.node_fn(ndata, **kwargs)
        return ndata, edata

    @classmethod
    def from_config(cls,
                      node_input_size:Optional[Union[int, Tuple[int, int]]],
                      edge_input_size:int,
                      node_output_size:Optional[Union[int, Tuple[int, int]]],
                      edge_output_size:int,
                      config:MessagePassingLayerConfig):
    
        kwargs = shallow_asdict(config)
        kwargs.pop("aggregate")
        kwargs.pop("aggregate_normalization")

        return cls(node_input_size, 
                    edge_input_size, 
                    node_output_size, 
                    edge_output_size, 
                    **kwargs)
    
@dataclass
class DeepGraphNNConfig:
    node_latent_size:Union[int, Tuple[int,int]] = 64
    edge_latent_size:int = 64
    num_message_passing_steps:int = 3
    use_node_encode:bool = True
    use_edge_encode:bool = True
    use_node_decode:bool = True 
    use_edge_decode:bool = True
    mpconfig:MessagePassingLayerConfig = MessagePassingLayerConfig()

class DeepGraphNN(nn.Module):
    
    encoder:Optional[FeatureEncoderLayer]
    processor:nn.ModuleList # nn.ModuleList[MessagePassingLayer]
    decoder:Optional[FeatureEncoderLayer]

    node_input_size:int 
    edge_input_size:int
    node_output_size:int
    edge_output_size:int
    def __init__(self, 
                 node_input_size:Union[int,Tuple[int,int]], 
                 edge_input_size:int,   
                 node_latent_size:Union[int,Tuple[int,int]],
                 edge_latent_size:int,
                 node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                 edge_output_size:Optional[int] = None,
                 num_message_passing_steps:int = 3,
                 use_node_encode:Union[bool,Tuple[bool,bool]] = True,
                 use_edge_encode:bool = True,
                 use_node_decode:Union[bool,Tuple[bool,bool]] = True, 
                 use_edge_decode:bool = True,
                 mpconfig:MessagePassingLayerConfig = MessagePassingLayerConfig()):
        """
        Parameters
        ----------
        node_input_size:Union[int,Tuple[int,int]]
            When it's bipartite, it should be a pair of int
        edge_input_size:int
            The size of edge features
        node_latent_size:Union[int,Tuple[int,int]]
            When it's bipartite, it's not required to be a pair, it can broadcast automatically
        edge_latent_size:int
            The latent size of edge features
        node_output_size:Optional[Union[int,Tuple[int,int]]]
            When it's bipartite, it's not required to be a pair, it can broadcast automatically
            if use_node_decode if False, it's not required to be provided, it will be automatically set to node_latent_size
        edge_output_size:Optional[int]
            When it's bipartite, it's not required to be a pair, it can broadcast automatically
            if use_edge_decode if False, it's not required to be provided, it will be automatically set to edge_latent_size
        num_message_passing_steps:int
            The number of message passing steps, default to 3
        use_node_encode:Union[bool,Tuple[bool,bool]]
            Whether to use node encoder
            if bipartite, it can be a pair of bool, it can be broadcasted automatically
            else, it should be a single bool
        use_edge_encode:bool
            Whether to use edge encoder
        use_node_decode:Union[bool,Tuple[bool,bool]]
            Whether to use node decoder
            if bipartite, it can be a pair of bool, it can be broadcasted automatically
            else, it should be a single bool
        use_edge_decode:bool
            Whether to use edge decoder
        mpconfig:MessagePassingLayerConfig
            The config of message passing layers

        """
        super().__init__()
        assert num_message_passing_steps > 0, f"num_message_passing_steps should be greater than 0, got {num_message_passing_steps}"

        is_bipartite = is_pair(node_input_size)

        # FeatureEncoderLayerCls = FeatureEncoderLayer if not is_bipartite else FeatureEncoderBipartiteLayer
        # MessagePassingLayerCls = MessagePassingLayer if not is_bipartite else MessagePassingBipartiteLayer
        broadcast_fn = make_pair if is_bipartite else lambda x:x

        if any(make_pair(use_node_encode)) or any(make_pair(use_edge_encode)):
            # any use_node_encode or use_edge_encode is True
            self.encoder = FeatureEncoderLayer.from_config(
                node_input_size = node_input_size if use_node_encode else None,  
                edge_input_size = edge_input_size if use_edge_encode else None,
                node_output_size = broadcast_fn(node_latent_size),
                edge_output_size = edge_latent_size,
                config = replace(mpconfig,
                                    node_fn_config = replace(mpconfig.node_fn_config,
                                                                use_conditional_norm = False),
                                    edge_fn_config = replace(mpconfig.edge_fn_config,
                                                                use_conditional_norm = False)
                                    )
            )
            
        else:
            self.encoder  = None
        
        node_input_size = node_latent_size if use_node_encode else node_input_size
        edge_input_size = edge_latent_size if use_edge_encode else edge_input_size

        self.processor = nn.ModuleList([
            MessagePassingLayer.from_config(
                node_input_size = broadcast_fn(node_input_size),
                edge_input_size = edge_input_size,
                node_output_size = broadcast_fn(node_latent_size),
                edge_output_size = edge_latent_size,    
                config = mpconfig
            )  
        ])
        for _ in range(num_message_passing_steps-1):
            self.processor.append(
                MessagePassingLayer.from_config(
                    node_input_size = broadcast_fn(node_latent_size),
                    edge_input_size = edge_latent_size,
                    node_output_size = broadcast_fn(node_latent_size),
                    edge_output_size = edge_latent_size,
                    config = mpconfig
                )
            )

        if any(make_pair(use_node_decode)) or any(make_pair(use_edge_decode)):
            # any use_node_decode or use_edge_decode is True
            self.decoder = FeatureEncoderLayer.from_config(
                node_input_size = broadcast_fn(node_latent_size) if use_node_decode else None,
                edge_input_size = edge_latent_size if use_edge_decode else None,
                node_output_size = broadcast_fn(node_output_size),
                edge_output_size = edge_output_size,
                config = replace(mpconfig,
                                node_fn_config = replace(mpconfig.node_fn_config,
                                                        use_layer_norm = False,
                                                        use_conditional_norm = False),
                                edge_fn_config = replace(mpconfig.edge_fn_config,
                                                        use_layer_norm = False,
                                                        use_conditional_norm = False)
                )
            )
        else:
            self.decoder = None

        if is_bipartite:
            self.node_output_size = tuple(lambda o,l,d:o if d else l for o,l,d in zip(
                                            make_pair(node_output_size), 
                                            make_pair(node_latent_size), 
                                            make_pair(use_node_decode)))
        else:
            self.node_output_size = node_output_size if use_node_decode else node_latent_size
        self.edge_output_size = edge_output_size if use_edge_decode else edge_latent_size

    def forward(self, 
                graph:Graph,
                ndata:Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edata:torch.Tensor,
                condition:Optional[float] = None
                )->Union[
                    Tuple[torch.Tensor, torch.Tensor],
                    Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]  
                ]:
      
        if self.encoder is not None:
            ndata, edata = self.encoder(ndata, edata, condition=condition)
        
        for layer in self.processor:
            ndata, edata = layer(graph, ndata, edata, condition=condition)
        
        if self.decoder is not None:
            ndata, edata = self.decoder(ndata, edata, condition=None)

        return ndata, edata
    
    @classmethod 
    def from_config(cls,
                    node_input_size:Union[int,Tuple[int,int]], 
                    edge_input_size:int,  
                    node_output_size:Optional[Union[int,Tuple[int,int]]] = None,
                    edge_output_size:Optional[int] = None,
                    config:DeepGraphNNConfig = DeepGraphNNConfig()):

        return cls(node_input_size = node_input_size,
                    edge_input_size = edge_input_size,
                    node_output_size = node_output_size,
                    edge_output_size = edge_output_size,
                     **shallow_asdict(config))
    

class RIGNO(nn.Module):
    encoder:DeepGraphNN
    processor:DeepGraphNN
    decoder:DeepGraphNN

    def __init__(self,
                 node_input_size:Tuple[int,int], 
                 edge_encoder_input_size:int,
                 edge_processor_input_size:int,
                 edge_decoder_input_size:int,
                 output_size:int,
                 variable_mesh:bool = False,
                 drop_edge:float = 0.0,
                 config:DeepGraphNNConfig = DeepGraphNNConfig()):
        super().__init__()
        self.drop_edge = drop_edge

        self.encoder = DeepGraphNN.from_config(
            node_input_size = make_pair(node_input_size),
            edge_input_size = edge_encoder_input_size,
            config = replace(config,
                    use_node_encode = True, # it will be broadcasted to pair
                    use_edge_encode = True,
                    use_node_decode = False, 
                    use_edge_decode = False,
                    num_message_passing_steps = 1)
        )
        self.processor = DeepGraphNN.from_config(
            node_input_size = config.node_latent_size,
            edge_input_size = edge_processor_input_size,
            config = replace(config,
                             use_node_encode=False, 
                             use_edge_encode=True,
                             use_node_decode=False,
                             use_edge_decode=False    )
        )
        self.decoder = DeepGraphNN.from_config(
            node_input_size  = make_pair(config.node_latent_size),
            edge_input_size  = edge_decoder_input_size,
            node_output_size = output_size,
            config = replace(config,
                      use_node_encode = (True, False),
                      use_edge_encode = True,
                      use_node_decode = (True, False),
                      use_edge_decode = False,  
                      num_message_passing_steps = 1,
                      )
        )

    def forward(self, 
                graphs:RegionInteractionGraph,
                pndata:Optional[torch.Tensor] = None, 
                condition:Optional[float] = None
                )->torch.Tensor:
        """
        Parameters
        ----------
        graphs:RegionInteractionGraph
        pndata:Optional[torch.Tensor]
            ND Tensor of shape [..., n_physical_nodes, node_input_size]
        condition:Optional[float]
            The condition of the model
        
        Returns
        -------
        torch.Tensor
            The output tensor of shape [..., n_physical_nodes, output_size]
        """

        graph:Graph = graphs.physical_to_regional
        graph       = graph.drop_edge(self.drop_edge)

        pndata      = torch.cat([graph.get_ndata()[0], pndata], -1) if pndata is not None else graph.get_ndata()[0] 
        rndata      = graph.get_ndata()[1] 
        ndata       = (pndata, rndata)
        edata       = graph.get_edata()
        ndata, _ = self.encoder(graph,
                                    ndata,
                                    edata,
                                    condition = condition)
        pndata, rndata = ndata

        graph:Graph = graphs.regional_to_regional
        graph       = graph.drop_edge(self.drop_edge)
        edata       = graph.get_edata()

        rndata, _ = self.processor(graph, rndata, edata, condition = condition)

        graph:Graph = graphs.regional_to_physical
        graph       = graph.drop_edge(self.drop_edge)
        ndata       = (rndata, pndata)
        edata       = graph.get_edata()
      
        ndata, _ = self.decoder(graph, ndata, edata)

        rndata, pndata = ndata

        return pndata


    
