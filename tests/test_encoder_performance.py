import sys
sys.path.append("../")
import numpy as np 
import xarray as xr
import torch
import time
from dataclasses import dataclass, replace
from src.model.cmpt.gno import IntegralTransform
from src.model.cmpt.utils.gno_utils import native_neighbor_search
from src.model.cmpt.deepgnn import DeepGraphNN 
from src.model.cmpt.message_passing import MessagePassingLayerConfig
from src.model.cmpt.mlp import AugmentedMLPConfig
from src.graph.graph import radius_bipartite_graph, Graph
from src.graph.support import minimal_support
from src.utils.sample import subsample

"""
    def __init__(self, 
                 edges:torch.Tensor, 
                 ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 src_ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None, 
                 dst_ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 edata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 gdata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None):
"""

def rigno_encode(model, radii, physical_points, physical_u, regional_points):

    torch.cuda.synchronize()
    start = time.time()
    
    edges = radius_bipartite_graph(
        physical_points,
        regional_points,
        radii
    )
    graph = Graph(
        edges = edges, 
        src_ndata = {"x": physical_u},
        dst_ndata = {"x": regional_points}
    )
    ndata,edata = model(graph, (physical_u,regional_points))
    result = ndata[-1]
   
    torch.cuda.synchronize()
    end = time.time()
    
    return result, end-start

def gno_encode(mlp, radii, physical_points, physical_u, regional_points):
    model = IntegralTransform(
        channel_mlp=mlp
    )
    
    torch.cuda.synchronize()
    start = time.time()
    neighbor = native_neighbor_search(physical_points, regional_points, radii[:,None])

    result = model(y=physical_points, neighbors=neighbor, x=regional_points, f_y=physical_u)
   
    torch.cuda.synchronize()
    
    end = time.time()

    return result, end-start

def transformer_forward():
    pass


if __name__ == '__main__':
    nsample = 100000
    ngrid = 100
    physical_points = torch.rand(nsample, 2)
    physical_u      = torch.rand(nsample, 2)
    regional_points = torch.stack(torch.meshgrid(torch.linspace(0, 1, ngrid), torch.linspace(0, 1, ngrid)), 2).reshape(-1, 2)


    model = DeepGraphNN(
        node_input_size=(physical_u.shape[-1], regional_points.shape[-1]),
        edge_input_size=None,
        node_latent_size = 2,
        edge_latent_size = 2,
        node_output_size=physical_u.shape[-1],
        edge_output_size=None,
        num_message_passing_steps = 1,
        use_node_encode = True,
        use_edge_encode = False, 
        use_node_decode = False,
        use_edge_decode = False,
        mpconfig = MessagePassingLayerConfig(
            use_edge_fn = True,
            use_node_fn = False
        )
    )

    radii = minimal_support(regional_points)
  
    # cpu

    result_rigno, time_rigno = rigno_encode(model, radii, physical_points, physical_u, regional_points)
    result_gno , time_gno   = gno_encode(model.processor[0].edge_fn, radii, physical_points, physical_u, regional_points)


    print(f"[CPU]RIGNO: {time_rigno}s, GNO: {time_gno}s")

    # assert torch.assert_allclose(result_rigno, result_gino)

    # gpu 

    model = model.cuda()
    physical_points = physical_points.cuda()
    physical_u = physical_u.cuda()
    regional_points = regional_points.cuda()
    radii = radii.cuda()

    # warm up
    result_gno , time_gno   = gno_encode(model.processor[0].edge_fn, radii, physical_points, physical_u, regional_points)
    result_gno , time_gno   = gno_encode(model.processor[0].edge_fn, radii, physical_points, physical_u, regional_points)
    result_rigno, time_rigno = rigno_encode(model, radii, physical_points, physical_u, regional_points)
    result_rigno, time_rigno = rigno_encode(model, radii, physical_points, physical_u, regional_points)

    print(f"[GPU]RIGNO: {time_rigno}s, GNO: {time_gno}s")

