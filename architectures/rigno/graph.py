import torch 
import torch.nn as nn
import numpy as np 
import inspect
from scipy.spatial import Delaunay
from scipy.stats import qmc
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union, Tuple, Mapping, Callable, TypeVar, List, Sequence
from .pair import is_pair, make_pair, force_make_pair
from .buffer import BufferDict, BufferList

############
# Tri Utils
############

def tri_medians(tri_points:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    tri_points: torch.Tensor
        ND tensor of shape [..., 3, n_dimension]
    
    Returns
    -------
    tri_medians: torch.Tensor
        (N-1)D tensor of shape [..., 3]
    """
    tri_edges = tri_points - torch.roll(tri_points, 1, dims=-2) # [..., 3, dim]
    norm_ab = torch.linalg.norm(tri_edges, dim=-1) # [..., 3]
    norm_bc = torch.roll(norm_ab, 1, dims=-1)
    norm_ca = torch.roll(norm_bc, 1, dims=-1)
    tri_medians = .335 * torch.sqrt( 2 * ( norm_ca ** 2 + norm_ab ** 2 ) - norm_bc ** 2 )
    return tri_medians

def delaunay(x:torch.Tensor)->torch.Tensor:
    """
    Build the Delaunay triangulation of a set of points
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    
    Returns
    -------
    simplices: torch.Tensor
        2D tensor of shape [n_triangles, 3]
    """
    return torch.from_numpy(Delaunay(x.cpu().numpy()).simplices).to(x.device)

def delaunay_edges(x:torch.Tensor, bidirection:bool =  False)->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    bidirection: bool
        whether to return the edges in both directions
    
    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    indptr, cols = Delaunay(x.cpu().numpy()).vertex_neighbor_vertices
    indptr = torch.from_numpy(indptr).to(device=x.device).long()
    cols   = torch.from_numpy(cols).to(device=x.device).long()
    rows   = torch.arange(indptr.shape[0]-1).to(device=x.device).repeat_interleave(torch.diff(indptr))
    edges  = torch.stack([rows, cols], 0)
    if  bidirection:
        edges = torch.cat([edges, torch.stack([cols, rows], 0)], 1)
    return edges

##############
# Graph Utils
##############
def radius_bipartite_graph(
    points_a:torch.Tensor,
    points_b:torch.Tensor,
    radii_b:torch.Tensor,
    periodic:bool = False,
    p:float = 2,
    )->torch.Tensor:
    """
    Compute a biparite graph between points_a and points_b by radius
    Parameters
    ----------
    points_a: torch.Tensor
        2D tensor of shape [n_points_a, n_dimension]
    points_b: torch.Tensor
        2D tensor of shape [n_points_b, n_dimension]
    radii_b: torch.Tensor
        1D tensor of shape [n_points_b,]
    periodic: bool
        whether the graph is periodic
    p: float
        the p-norm to use for the distance

    Returns
    -------
    bipartite_edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    assert points_a.ndim == 2 and points_b.ndim == 2, f"The points_a and points_b are expected to be 2D tensors, but got shapes {points_a.shape} and {points_b.shape}"
    assert points_b.shape[0] == radii_b.shape[0], f"The points_b and radii_b should have the same number of points, but got shapes {points_b.shape} and {radii_b.shape}"
    if periodic:
        residual = points_a[:, None, :] - points_b[None, :, :]
        residual = torch.where(residual >= 1., residual - 2., residual)
        residual = torch.where(residual < -1., residual + 2., residual)
        distances = torch.linalg.norm(residual, axis=-1, ord=p) # [n_points_a, n_points_b]
    else:
        distances = torch.cdist(points_a, points_b, p=p) # [n_points_a, n_points_b]

    bipartite_edges = torch.stack(torch.where(distances < radii_b[None, :]), 0) # [2, n_edges]
    return bipartite_edges

def remove_duplicate_edges(edges:torch.Tensor,return_idx:bool = False
    )->Union[torch.Tensor,Tuple[torch.Tensor,torch.Tensor]]:
    """

    Parameters
    ----------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]

    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    # NOTE: edges should be sorted first
    assert edges.ndim == 2 and edges.shape[0] == 2, f"The edges are expected to be [2, n_edges], but got shape {edges.shape}"
    assert edges.min() >= 0, "The edges are expected to be non-negative"
    assert edges.max() < torch.iinfo(torch.int32).max, "The edges are expected to be less than the maximum integer value"
    dtype = edges.dtype
    edges_id = edges[0].type(torch.int64) << 32 | edges[1].type(torch.int64)

    if return_idx:
        unique_edges_id, counts = edges_id.unique(return_counts=True) # [n_unique_edges]
        unique_idx = torch.cumsum(counts, 0) - counts
    else:
        unique_edges_id = edges_id.unique() # [n_unique_edges]
    unique_edges = torch.stack([
        unique_edges_id >> 32,
        unique_edges_id & 0xFFFFFFFF
    ], 0)
    unique_edges = unique_edges.type(dtype)
    if return_idx:
        return unique_edges, unique_idx
    else:
        return unique_edges

def sort_edges(edges:torch.Tensor, return_idx:bool=False
               )->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Parameters
    ----------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    return_idx: bool
        whether to return the indices

    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    or 
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    idx: torch.Tensor
        1D tensor of shape [n_edges]
    """
    assert edges.ndim == 2 and edges.shape[0] == 2, f"The edges are expected to be [2, n_edges], but got shape {edges.shape}"
    assert edges.min() >= 0, "The edges are expected to be non-negative"
    assert edges.max() < torch.iinfo(torch.int32).max, "The edges are expected to be less than the maximum integer value"
    dtype = edges.dtype
    edges_id = edges[0].type(torch.int64) << 32 | edges[1].type(torch.int64)
    sorted_edges_id = edges_id.argsort()
    if return_idx:
        return edges[:, sorted_edges_id].type(dtype), sorted_edges_id   
    else:
        return edges[:, sorted_edges_id].type(dtype)

def hierarchical_graph(
    points:torch.Tensor, # [n_points, n_dim]
    level:int,
    sample_factor:float = 2.,
    domain_shifts:Optional[torch.Tensor] = None, # [n_shifts, n_dim]
    return_levels:bool = False
    )->Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    points: torch.Tensor
        2D tensor of shape [n_points, n_dim]
    level: int
        number of levels
    periodic: bool
        whether the graph is periodic
    sample_factor: float
        factor of subsampling
    domain_shifts: Optional[torch.Tensor]
        2D tensor of shape [n_shifts, n_dim]

    Returns
    -------
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    domains: torch.Tensor
        2D tensor of shape [2, n_edges]
    """
    assert level > 0, "The level should be positive"
    edges   = []
    domains = []
    if return_levels:
        levels = []
    for l in range(level):
        # Sub-sample the rmesh
        num_sampled  = int(points.shape[0] / (sample_factor ** l))
        if num_sampled < 4:
            continue
        level_points = points[:num_sampled]
        # Construct a triangulation
        if domain_shifts is not None:
            # Repeat the rmesh in periodic directions
            level_points = shift(level_points, domain_shifts)
        # Get the relevant edges
        extended_edges = delaunay_edges(level_points) # [2, n_edges]
        level_domains  = extended_edges // num_sampled # [2, n_edges]

        level_edges    = extended_edges % num_sampled # [2, n_edges]
      
        if domain_shifts is  not None: # periodic
            is_relevant = torch.any(level_domains == 0, dim=0)
        else: # not periodic
            is_relevant = torch.all(level_domains == 0, dim=0)
        
        level_edges   = level_edges  [:, is_relevant] # [2, n_edges]
        level_domains = level_domains[:, is_relevant] # [2, n_edges]

        if return_levels:
            levels.extend([l] * level_edges.shape[1])

        edges.append(level_edges)
        domains.append(level_domains)

    edges = torch.cat(edges, 1) # [2, n_edges]
    domains = torch.cat(domains, 1) # [2, n_edges]
    if return_levels:
        levels = torch.tensor(levels)

    edges, sort_idx = sort_edges(edges, return_idx=True)
    domains = domains[:, sort_idx]
    edges, unique_idx = remove_duplicate_edges(edges, return_idx=True)
    domains = domains[:, unique_idx]
    if return_levels:
        levels = levels[sort_idx]
        levels = levels[unique_idx]

    if return_levels:
        return edges, domains, levels
    else:
        return edges, domains

def minimal_support(points:torch.Tensor, 
                    domain_shifts:Optional[torch.Tensor] = None
                    )->torch.Tensor:
    """
    compute the maximal median support of a set of points. These medians are computed 
    by the Delaunay triangulation of the points.

    Parameters
    ----------
    points: torch.Tensor
        2D Tensor of shape [n_points, n_dim]
    domain_shifts: torch.Tensor
        2D Tensor of shape [n_shifts, n_dim]

    Returns
    -------
    radii: torch.Tensor
        1D tensor of shape [n_points,]
    """

    if domain_shifts is not None:
        shifted_points = shift(points, domain_shifts)
        triangles = delaunay(shifted_points)
        medians   = tri_medians(shifted_points[triangles])
    else:
        triangles = delaunay(points)
        medians   = tri_medians(points[triangles])
    
    mask = triangles < points.shape[0] # [N, 3] in case of shifted
    values = medians[mask]
    indices  = triangles[mask]
    radii = torch.full((points.shape[0],), -1, dtype=points.dtype, device=points.device)
    radii = torch.index_reduce(radii, 0, indices, values, reduce="amax")

    return radii

@lru_cache()
def domain_shifts(span:Sequence[int]=(2,2))->torch.Tensor:
    """
    Parameters
    ----------
    span: Sequence[int] 
        span of domains in each dimension

    Returns
    -------
    domain_shifts: torch.Tensor
        2D tensor of shape [3^n_dim-1, n_dim]
    """
    values = []
    for s in span:
        values.append(torch.tensor([0, -s, s]))
    
    grids  = torch.meshgrid(*values, indexing='ij')
    points = torch.stack(grids, dim=-1).reshape(-1, len(span))
    
    return points

def shift(points:torch.Tensor, domain_shifts:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    points:torch.Tensor
        2D Tensor of shape [n_points, n_dimension]
    domain_shifts:torch.Tensor 
        2D Tensor of shape [n_shifts, n_dimension]

    Returns
    -------
    points:torch.Tensor
        2D Tensor of shape [n_points*n_shifts, n_dimension]
    """
    assert points.ndim == 2, f"The points are expected to be 2D tensor, but got shape {points.shape}"
    assert domain_shifts.ndim == 2, f"The domain_shifts are expected to be 2D tensor, but got shape {domain_shifts.shape}"
    assert points.shape[-1] == domain_shifts.shape[-1], \
        f"The points and domain_shifts should have the same number of dimensions, but got {points.shape[-1]} and {domain_shifts.shape[-1]}"
    n_dimension = points.shape[-1]
    points = points[None, :, :] + domain_shifts[:, None, :]
    return points.reshape(-1, n_dimension)

##############
# Other Utils
##############

def subsample(values:torch.Tensor, 
              n:Optional[int] = None, 
              factor:Optional[float] = None)->torch.Tensor:
    """
    Parameters
    ----------
    values: torch.Tensor
        1D tensor
    n: Optional[int]
        number of samples  
    factor: Optional[float]
        factor of subsampling, should be in the range (0, 1]
    Returns
    -------
    subsampled_values: torch.Tensor
        1D tensor
    """
    assert n is None or (n > 0 and n <= values.shape[0]), "The number of samples should be in the range (0, n_values]"
    assert factor is None or factor <= 1 and factor > 0, "The factor should be in the range (0, 1]"
    if n is None and factor is None:
        raise ValueError("Either n or factor should be provided")
    if factor is not None:
        n = int(values.shape[0] * factor)
    idx = torch.randperm(values.shape[0])[:n]
    return values[idx]

def rescale(x:torch.Tensor, lims=(-1,1))->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        1D tensor
    
    Returns
    -------
    x_normalized: torch.Tensor
        1D tensor
    """
    return (x-x.min()) / (x.max()-x.min()) * (lims[1] - lims[0]) + lims[0]

def add_dummy(x:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        ND tensor of shape [..., n_points, n_features]

    Returns
    -------
    x: torch.Tensor
        ND tensor of shape [..., n_points+1, n_features]
    """
    assert x.ndim >= 2, f"The x is expected to be ND tensor, but got shape {x.shape}"
    device = x.device
    dtype  = x.dtype
    shape  = list(x.shape)
    shape[-2] = 1
    x = torch.cat([x, torch.zeros(shape, device=device, dtype=dtype)], -2)
    return x

def remove_dummy(x:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [..., n_points, n_features]

    Returns
    -------
    x: torch.Tensor
        2D tensor of shape [..., n_points-1, n_features]
    """
    assert x.ndim >= 2, f"The x is expected to be ND tensor, but got shape {x.shape}"
    return x[..., :-1, :]
    

#################
# Encoding Utils
#################
def node_pos_encode(x:torch.Tensor,
                    ndata:Optional[torch.Tensor]=None,
                    freq:int = 4,
                    periodic:bool = False,
                    add_dummy_node:bool = False
               )->torch.Tensor:
    """
    Parameters
    ----------
    x: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    ndata: Optional[torch.Tensor]
        2D tensor of shape [n_points, n_features] or None
    freq: int
        number of frequencies
    periodic: bool
        whether the encoding is periodic
    add_dummy_node: bool
        whether to add a dummy node

    Returns
    -------
    x_encoded: torch.Tensor
        2D tensor of shape [n_points or n_points+1, 2*freq or 2*freq+n_features or n_dimension or n_dimension+1]
    """
    assert x.ndim == 2, f"The x is expected to be 2D tensor, but got shape {x.shape}"
    assert ndata is None or (ndata.ndim == 2 and ndata.shape[0] == x.shape[0]), f"The ndata is expected to be 2D tensor, but got shape {ndata.shape}"
    if periodic:
        device = x.device
        freqs = torch.arange(1, freq+1).to(device=device) # [freq]
        phi   = np.pi * (x + 1)
        x = freqs[None, :] * phi[:, None] # [n_points, freq]
        x = torch.stack([x.sin(), x.cos()], axis=-1).reshape(x.shape[0],-1) # [n_points, 2*freq]
    
    if ndata is not None:
        dtype = x.dtype
        x = x.type(dtype)
        x = torch.cat([x, ndata], axis=-1)
 
    
    if add_dummy_node:
        x = add_dummy(x)
    
    return x

def edge_pos_encode(u:torch.Tensor,
                    v:torch.Tensor,
                    edges:torch.Tensor,
                    edata:Optional[torch.Tensor] = None,
                    periodic:bool = False,
                    max_edge_length:float = 2.0,
                    domain_shifts:Optional[torch.Tensor] = None,
                    domain_edges:Optional[torch.Tensor] = None
                    )->torch.Tensor:
    """
    Parameters
    ----------
    u: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    v: torch.Tensor
        2D tensor of shape [n_points, n_dimension]
    edges: torch.Tensor
        2D tensor of shape [2, n_edges]
    edata: Optional[torch.Tensor]
        2D tensor of shape [n_edges, n_features]
    periodic: bool
        whether the encoding is periodic
    max_edge_length: float
        maximum edge length
    domain_shifts: Optional[torch.Tensor]
        2D tensor of shape [n_shifts, n_dimension]
    domain_edges: Optional[torch.Tensor]
        2D tensor of shape [2, n_domain]

    Returns
    -------
    edata: torch.Tensor
        2D tensor of shape [n_edges, 2*n_dimension+1]
    """

    u_e, v_e = u[edges[0]], v[edges[1]] # [n_edges, n_dim]
    z_uv = u_e - v_e # [n_edges, n_dim]
    assert (z_uv.abs() <= 2.0).all(), "The edge length should be less than 2.0"
    
    if periodic:
        if domain_shifts is None:
            z_uv = torch.where(z_uv >= 1., z_uv - 2., z_uv)
            z_uv = torch.where(z_uv < -1., z_uv + 2., z_uv)
        else:
            z_uv = ( u[edges[0]] + domain_shifts[domain_edges[0]] ) \
                 - ( v[edges[1]] + domain_shifts[domain_edges[1]] ) # [n_edges, n_dim]
        
    d_uv = torch.linalg.norm(z_uv, axis=-1, keepdims=True) # [n_edges, 1]

    assert (z_uv.abs() <= max_edge_length).all(), z_uv.abs().max()
    assert (d_uv.abs() <= max_edge_length).all(), d_uv.abs().max()

    z_uv = z_uv / max_edge_length
    d_uv = d_uv / max_edge_length
    edata = torch.cat([z_uv, d_uv], axis=-1) # [n_edges, 2*n_dim+1]

    if edata is not None:
        edata = torch.cat([edata, edata], axis=-1)
    return edata


###############
# Mesh
###############

class Mesh(nn.Module):

    points:torch.Tensor # [n_points, dim]
    is_boundary:torch.Tensor # [n_points,]

    def __init__(self, points:torch.Tensor, is_boundary:torch.Tensor):
        super().__init__()
        self.register_buffer('points', points)
        self.register_buffer('is_boundary', is_boundary)

    @classmethod
    def grid(cls, x_lims=(-1,1), y_lims=(-1,1), n_points=10, periodic:bool=False):
        x = torch.linspace(*x_lims, n_points)
        y = torch.linspace(*y_lims, n_points)
        points = torch.stack(torch.meshgrid(x, y), -1).reshape(-1, 2) # [n_points, 2]
        is_boundary = (points[:, 0] == x_lims[0])\
                    | (points[:, 0] == x_lims[1])\
                    | (points[:, 1] == y_lims[0])\
                    | (points[:, 1] == y_lims[1])
        
        if periodic:
            is_bottom_or_right = (points[:, 0] == x_lims[1]) | (points[:, 1] == y_lims[0])
            points      = points[~is_bottom_or_right]
            is_boundary = is_boundary[~is_bottom_or_right]
        return cls(points, is_boundary)
    
    @classmethod 
    def circle(cls, center=(0,0), radius=1, n_points_per_radii=10, n_radii=10):
        theta = torch.linspace(0, 2*np.pi, n_radii)
        radius= torch.linspace(0, radius, n_points_per_radii)
        theta, radii = torch.meshgrid(theta, radius)
        theta, radii = theta.flatten(), radii.flatten()
        points = torch.stack([torch.cos(theta), torch.sin(theta)], -1) * radii[:, None] # [n_points, 2]
        
        points = points + torch.tensor(center).to(device=points.device, dtype=points.dtype)[None, :]
        
        is_boundary = radii == radius
        return cls(points, is_boundary)


T = TypeVar("T")
def make_dict(data:Union[Mapping[str,T],T], default_key:str = 'x')->Mapping[str,T]:
    if not isinstance(data, Mapping):
        return {default_key:data}
    else:
        return data

class Graph(nn.Module):
    edges:torch.Tensor # [2, edges]
    ndata:Optional[BufferDict] # [..., n_nodes, n_features]
    src_ndata:Optional[BufferDict] # [..., n_src_nodes, n_features]
    dst_ndata:Optional[BufferDict] # [..., n_dst_nodes, n_features]
    edata:Optional[BufferDict] # [..., n_edges, n_features]
    gdata:Optional[BufferDict] # [..., n_features]


    def __init__(self, 
                 edges:torch.Tensor, 
                 ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 src_ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None, 
                 dst_ndata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 edata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None,
                 gdata:Optional[Union[Mapping[str,torch.Tensor],torch.Tensor]] = None):
        super().__init__()

        self.register_buffer("edges", edges)

        if ndata is not None:
            self.ndata = BufferDict(make_dict(ndata))
            self.src_ndata = None
            self.dst_ndata = None
        else:
            self.ndata = None 
            self.src_ndata = BufferDict(make_dict(src_ndata)) if src_ndata is not None else None 
            self.dst_ndata = BufferDict(make_dict(dst_ndata)) if dst_ndata is not None else None

        if edata is not None:
            assert edata.shape[0] == edges.shape[1]
            self.edata = BufferDict(make_dict(edata))
        else:
            self.edata = None   

        if gdata is not None:
            self.gdata = BufferDict(make_dict(gdata))
        else:
            self.gdata = None
      
    @property 
    def ndim(self)->Union[int, Tuple[int, int]]:
        if self.ndata is not None:
            assert 'x' in self.ndata, f"The node data should have a key 'x', got keys {self.ndata.keys()}"
            return self.ndata['x'].shape[-1]
        else:
            assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
            assert 'x' in self.dst_ndata, f"The node data should have a key 'x', got keys {self.dst_ndata.keys()}"
            return self.src_ndata['x'].shape[-1], self.dst_ndata['x'].shape[-1]
        
    @property 
    def edim(self)->int:
        assert 'x' in  self.edata, f"The edge data should have a key 'x', got keys {self.edata.keys()}"
        return self.edata['x'].shape[-1]
    
    @property 
    def src_ndim(self)->int:
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
        return self.src_ndata['x'].shape[-1]
    
    @property
    def dst_ndim(self)->int:
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.dst_ndata, f"The node data should have a key 'x', got keys {self.dst_ndata.keys()}"
        return self.dst_ndata['x'].shape[-1]

    @property 
    def num_edges(self)->int:
        return self.edges.shape[1]
    
    @property 
    def is_bipartite(self)->bool:
        return self.src_ndata is not None and self.dst_ndata is not None

    @property 
    def num_nodes(self)->Union[int, Tuple[int, int]]:
        if self.ndata is not None:
            assert 'x' in self.ndata, f"The node data should have a key 'x', got keys {self.ndata.keys()}"
            return self.ndata['x'].shape[-2]
        else:
            assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
            return self.src_ndata['x'].shape[-2], self.dst_ndata['x'].shape[-2]

    @property 
    def num_src_nodes(self):
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.src_ndata, f"The node data should have a key 'x', got keys {self.src_ndata.keys()}"
        return self.src_ndata['x'].shape[-2]
    
    @property 
    def num_dst_nodes(self):
        assert self.is_bipartite, "The graph is not bipartite"
        assert 'x' in self.dst_ndata, f"The node data should have a key 'x', got keys {self.dst_ndata.keys()}"
        return self.dst_ndata['x'].shape[-2]

    def message(self, fn:Callable, 
                    ndata:Optional[
                        Union[
                            torch.Tensor,
                            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
                        ] = None,
                    edata:Optional[torch.Tensor] = None,
                    gdata:Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        fn: Callable
            function to apply to the edge data
            the function should recieve parameters

            src_ndata:torch.Tensor
                ND tensor of shape [..., n_edges, n_src_features]
            dst_ndata:torch.Tensor
                ND tensor of shape [..., n_edges, n_dst_features]
            edata:torch.Tensor
                ND tensor of shape [..., n_edges, n_features]
            
            additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
                ND tensor of shape [..., n_src/dst_nodes,n_features]

            and return a tensor of shape [..., n_edges, n_features]

        ndata: Optional[torch.Tensor or Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
            ND tensor of shape [..., n_nodes, n_features]
            if graph is bipartie, ndata could be None or Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]
            else:            

        edata: Optional[torch.Tensor]
            ND tensor of shape [..., n_edges, n_features]

        gdata: Optional[torch.Tensor]
            ND tensor of shape [..., n_features]


        Returns
        -------
        message: torch.Tensor
            ND tensor of shape [..., n_edges, n_features]
        """
        if self.is_bipartite:
            assert ndata is None or is_pair(ndata), f"The ndata is expected to be pair, but got {type(ndata)}"
    
        sig = inspect.signature(fn)
        n_parameters = len(sig.parameters)

        ndata = make_pair(ndata)
        src_ndata, dst_ndata = ndata # [n_src_nodes, n_src_features], [n_dst_nodes, n_dst_features]
        if src_ndata is not None:
            assert src_ndata.shape[-2] == make_pair(self.num_nodes)[0]
            src_ndata = src_ndata[..., self.edges[0], :]
        if dst_ndata is not None:
            assert dst_ndata.shape[-2] == make_pair(self.num_nodes)[1]
            dst_ndata = dst_ndata[..., self.edges[1], :]
        if edata is not None:
            assert edata.shape[-2] == self.num_edges

        
        if n_parameters == 2:
            return fn(src_ndata, dst_ndata)
        elif n_parameters == 3:
            return fn(src_ndata, dst_ndata, edata)
        elif n_parameters == 4:
            if gdata is not None:
                expand_dim = [-1] * (gdata.ndim + 1)
                expand_dim[-2] = self.num_edges  
                gdata = gdata[..., None, :].expand(*expand_dim)
            return fn(src_ndata, dst_ndata, edata, gdata)
        else:
            raise ValueError(f"The function {fn} should have 2, 3 or 4 parameters, but got {n_parameters}")
              
    def aggregate(self, fn:Union[Callable, Tuple[Callable, Callable]], 
                  edata:torch.Tensor,
                  ndata:Optional[torch.Tensor]=None,
                  gdata:Optional[torch.Tensor]=None,
                  dtype:Optional[torch.dtype] =None,
                  reduce:str="mean"):
        """
        Parameters
        ----------
        fn: Callable
            function to apply to the edge data
            the function should recieve parameters

            for bipartite graph:

                edata:torch.Tensor
                    ND Tensor of shape [..., n_src/dst_nodes, n_edge_features]
                ndata:Optional[torch.Tensor]
                    ND Tensor of shape [..., n_src/dst_nodes, n_node_features]

                additionally, you could add gdata:Optional[torch.Tensor] as the last argument 
                    ND tensor of shape [..., n_src/dst_nodes,n_features]

            for homogeneous graph:

                src_edata:torch.Tensor
                    ND Tensor of shape [..., n_nodes, n_edge_features]
                dst_edata:torch.Tensor
                    ND Tensor of shape [..., n_nodes, n_edge_features]
                
                ndata: Optional[torch.Tensor]
                    ND tensor of shape [..., n_nodes, n_features]

                additionally, you could add gndata:Optional[torch.Tensor] as the last argument 
                    ND tensor of shape [..., n_nodes,n_features]

        edata: torch.Tensor
            ND tensor of shape [...,  n_edges, n_features]

        ndata: Optional[torch.Tensor]
            ND tensor of shape [..., n_nodes, n_features]

        gdata: Optional[torch.Tensor]
            ND tensor of shape [..., n_features]
            
        reduce: str
            reduce operation to apply to the aggregated data
            choose from ["sum", "mean", "prod", "amax", "amin"]
            
        """
        assert reduce in ["sum", "mean", "prod", "amax", "amin"], \
            f'The reduce is expected one of ["sum", "mean", "prod", "amax", "amin"], but got {reduce}'
        dtype = edata.dtype if dtype is None else dtype
        src_shape, dst_shape = force_make_pair(list(edata.shape))
        src_shape[-2] = make_pair(self.num_nodes)[0]
        dst_shape[-2] = make_pair(self.num_nodes)[1]

        src_edata = torch.zeros(*src_shape, dtype=dtype, device=edata.device)
        dst_edata = torch.zeros(*dst_shape, dtype=dtype, device=edata.device)      
    
        src_edata = src_edata.index_reduce_(dim=-2, index=self.edges[0], source=edata, reduce=reduce)  
        dst_edata = dst_edata.index_reduce_(dim=-2, index=self.edges[1], source=edata, reduce=reduce)
       
        if is_pair(fn):
            n_parameters = [len(inspect.signature(f).parameters) for f in fn]
            assert n_parameters[0] == n_parameters[1], f"The functions {fn} should have the same number of parameters, but got {n_parameters}"
            n_parameters = n_parameters[0]
        else:
            assert isinstance(fn, Callable), f"The fn is expected to be Callable, but got {type(fn)}"
            n_parameters = len(inspect.signature(fn).parameters)

        if self.is_bipartite:
           
            src_ndata, dst_ndata = make_pair(ndata)
            if n_parameters == 2:
                fn = make_pair(fn)
                return fn[0](src_edata, src_ndata), fn[1](dst_edata, dst_ndata)
            elif n_parameters == 3:
                if gdata is not None:
                    expand_dim = [-1] * (gdata.ndim + 1)
                    expand_dim[-2] = self.num_nodes[0]  
                    src_gdata = gdata[..., None, :].expand(*expand_dim)
                    expand_dim[-2] = self.num_nodes[1]  
                    dst_gdata = gdata[..., None, :].expand(*expand_dim)
                    fn = make_pair(fn)
                return fn[0](src_edata, src_ndata, src_gdata), fn[1](dst_edata, dst_ndata, dst_gdata)
            else:
                raise ValueError(f"The function {fn} should have 2 or 3 parameters, but got {n_parameters}")
        else:
            # NOTE: we only consider the directed graph here
            if n_parameters == 3:
                return fn(src_edata, dst_edata, ndata)
            elif n_parameters == 4:
                if gdata is not None:
                    expand_dim = [-1] * (gdata.ndim + 1)
                    expand_dim[-2] = self.num_nodes  
                    gdata = gdata[..., None, :].expand(*expand_dim)
                return fn(src_edata,dst_edata, ndata, gdata)
            else:
                raise ValueError(f"The function {fn} should have 2 or 3 parameters, but got {n_parameters}")

    def get_ndata(self, key:str = 'x')->Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.ndata is not None:
            return self.ndata[key]
        else:
            assert self.src_ndata is not None, "The source node data is not defined"
            assert self.dst_ndata is not None, "The destination node data is not defined"
            return self.src_ndata[key], self.dst_ndata[key]

    def get_edata(self, key:str = 'x')->torch.Tensor:
        assert self.edata is not None, "The edge data is not defined"
        return self.edata[key]

    def get_gdata(self, key:str = 'x')->torch.Tensor:
        assert self.gdata is not None, "The graph data is not defined"
        return self.gdata[key]

    def drop_edge(self, p:float, seed:Optional[int]=None)->"Graph":
        if p <= 0.0:
            return self
        
        if seed is not None:
            torch.manual_seed(seed)
        
        mask = torch.rand(self.num_edges) >= p
        return Graph(
            edges = self.edges[:, mask],
            ndata = self.ndata,
            src_ndata = self.src_ndata.asdict(),
            dst_ndata = self.dst_ndata.asdict(),
            edata = {k:v[...,mask,:] for k,v in self.edata.asdict().items()}
        )

    @classmethod
    def with_pos_encode(cls,
        pos:torch.Tensor, # [n_points, n_dim]
        edges:torch.Tensor, #[2, n_edges],
        ndata:Optional[torch.Tensor] = None,
        edata:Optional[torch.Tensor] = None,
        domain_shifts:Optional[torch.Tensor] = None, # [n_shifts, n_dim]
        domain_edges:Optional[torch.Tensor] = None, # [2, n_domain]
        max_edge_length:float = 2.0,
        node_freq:int = 4,
        periodic:bool = False,
        add_dummy_node:bool = False,    
        with_additional_info:bool = True,
        ):

        ndata = node_pos_encode(pos, ndata=ndata, freq=node_freq, periodic=periodic, add_dummy_node=add_dummy_node)
        edata = edge_pos_encode(u = pos, v = pos, edges=edges, edata=edata, periodic=periodic, max_edge_length=max_edge_length, domain_shifts=domain_shifts, domain_edges=domain_edges)
        
        return cls(edges=edges, 
                   ndata={
                       "x":ndata,
                       "pos":pos
                       } if with_additional_info else ndata, 
                   edata=edata)
    
    @classmethod
    def bipartite_with_pos_encode(
        cls,
        edges:torch.Tensor, # [2, n_edges]
        src_pos:torch.Tensor, # [n_src_nodes, n_dim]
        dst_pos:torch.Tensor, # [n_dst_nodes, n_dim]
        src_ndata:Optional[torch.Tensor] = None, # [n_src_nodes, n_src_features]
        dst_ndata:Optional[torch.Tensor] = None, # [n_dst_nodes, n_dst_features]
        edata:Optional[torch.Tensor] = None, # [n_edges, n_features]
        domain_shifts:Optional[torch.Tensor] = None, # [n_shifts, n_dim]
        domain_edges:Optional[torch.Tensor] = None, # [2, n_domain]
        max_edge_length:float = 2.0,
        node_freq:int = 4, # number of frequencies for node features
        periodic:bool = False, # whether the graph is periodic
        add_dummy_node:bool = False, # whether to add a dummy node
        with_additional_info:bool = True,
        ):
        """
        Build the bipartite graph with positional encoding.  

        Parameters:
        -----------
        edges: torch.Tensor
            2D tensor of shape [2, n_edges]
        """
        src_ndata = node_pos_encode(src_pos, src_ndata, freq=node_freq, periodic=periodic, add_dummy_node=add_dummy_node)
        dst_ndata = node_pos_encode(dst_pos, dst_ndata, freq=node_freq, periodic=periodic, add_dummy_node=add_dummy_node)
        edata = edge_pos_encode(src_pos, dst_pos, edges, periodic=periodic, max_edge_length=max_edge_length, domain_shifts=domain_shifts, domain_edges=domain_edges)
        return cls( edges, 
                    src_ndata = {
                       "x":src_ndata,
                       "pos":src_pos
                    } if with_additional_info else src_ndata, 
                    dst_ndata = {
                        "x":dst_ndata,
                        "pos":dst_pos
                    } if with_additional_info else dst_ndata, 
                    edata = edata)


class RegionInteractionGraph(nn.Module):
    physical_to_regional:Graph # bipartite
    regional_to_regional:Graph # homogeneous
    regional_to_physical:Graph # bipartite

    def __init__(self, 
                 physical_to_regional:Graph,
                 regional_to_regional:Graph,
                 regional_to_physical:Graph):
        super().__init__()
        self.physical_to_regional = physical_to_regional
        self.regional_to_regional = regional_to_regional
        self.regional_to_physical = regional_to_physical

    @classmethod 
    def from_mesh(cls, mesh:Mesh, 
                  output_mesh:Optional[Mesh] = None,
                  periodic:bool = False,
                  sample_factor:float = 0.5,
                  overlap_factor_p2r:float = 1.0,
                  overlap_factor_r2p:float = 1.0,
                  regional_level:int = 1,
                  add_dummy_node:bool = True,
                  with_additional_info:bool = True):

        physical_points = rescale(mesh.points, (-1, 1))  
        if output_mesh is not None:
            output_physical_points = rescale(output_mesh.points, (-1, 1))
        else:
            output_physical_points = physical_points

        if periodic:
            _domain_shifts = domain_shifts((2.0,2.0))
            regional_points = subsample(physical_points, factor=sample_factor)
        else:
            _domain_shifts = None
            regional_points = torch.cat([
                physical_points[mesh.is_boundary],
                subsample(physical_points[~mesh.is_boundary], factor=sample_factor)
            ])

        radii = minimal_support(regional_points, _domain_shifts)
        # compute physical to regional edges
        p2r_edges = radius_bipartite_graph(
            physical_points, 
            regional_points, 
            radii * overlap_factor_p2r, 
            periodic
        )
        physical_to_regional = Graph.bipartite_with_pos_encode(
            p2r_edges, 
            src_pos   = physical_points, 
            dst_pos   = regional_points,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            dst_ndata       = (radii * overlap_factor_p2r)[:,None],
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        r2r_edges, domains = hierarchical_graph(
            regional_points, 
            level=regional_level, 
            domain_shifts=_domain_shifts)

        regional_to_regional = Graph.with_pos_encode(
            edges = r2r_edges, 
            pos = regional_points,
            ndata = (overlap_factor_r2p * radii)[:,None],
            domain_shifts = domain_shifts,
            domain_edges = domains,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        # compute regional to physical edges
        if output_mesh is None:
            r2p_edges = torch.flip(p2r_edges, dims=(0,))
        else:
            r2p_edges = torch.flip(radius_bipartite_graph(
                output_physical_points, 
                regional_points, 
                radii * overlap_factor_r2p, 
                periodic
            ), dims=(0,))

        regional_to_physical = Graph.bipartite_with_pos_encode(
            r2p_edges, 
            src_pos   = regional_points, 
            dst_pos   = physical_points,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            src_ndata       = (overlap_factor_r2p * radii)[:, None],
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )
        

        return cls(physical_to_regional, regional_to_regional, regional_to_physical)

    @classmethod
    def from_point_cloud(cls, points: torch.Tensor,
                         output_points: Optional[torch.Tensor] = None,
                         periodic: bool = False,
                         sample_factor: float = 0.5,
                         overlap_factor_p2r: float = 1.0,
                         overlap_factor_r2p: float = 1.0,
                         regional_level: int = 1,
                         add_dummy_node: bool = True,
                         with_additional_info:bool = True):
        
        physical_points = rescale(points, (-1, 1))  
        if output_points is not None:
            output_physical_points = rescale(output_points, (-1, 1))
        else:
            output_physical_points = physical_points

        if periodic:
            _domain_shifts = domain_shifts((2.0,2.0))
            regional_points = subsample(physical_points, factor=sample_factor)
        else:
            _domain_shifts = None
            regional_points = subsample(physical_points, factor=sample_factor)

        radii = minimal_support(regional_points, _domain_shifts)
        # compute physical to regional edges
        p2r_edges = radius_bipartite_graph(
            physical_points, 
            regional_points, 
            radii * overlap_factor_p2r, 
            periodic
        )
        physical_to_regional = Graph.bipartite_with_pos_encode(
            p2r_edges, 
            src_pos   = physical_points, 
            dst_pos   = regional_points,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            dst_ndata       = (radii * overlap_factor_p2r)[:,None],
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        r2r_edges, domains = hierarchical_graph(
            regional_points, 
            level=regional_level, 
            domain_shifts=_domain_shifts)

        regional_to_regional = Graph.with_pos_encode(
            edges = r2r_edges, 
            pos = regional_points,
            ndata = (overlap_factor_r2p * radii)[:,None],
            domain_shifts = domain_shifts,
            domain_edges = domains,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )

        # compute regional to physical edges
        if output_points is None:
            r2p_edges = torch.flip(p2r_edges, dims=(0,))
        else:
            r2p_edges = torch.flip(radius_bipartite_graph(
                output_physical_points, 
                regional_points, 
                radii * overlap_factor_r2p, 
                periodic
            ), dims=(0,))

        regional_to_physical = Graph.bipartite_with_pos_encode(
            r2p_edges, 
            src_pos   = regional_points, 
            dst_pos   = physical_points,
            max_edge_length = 2 * np.sqrt(regional_points.shape[1]),
            src_ndata       = (overlap_factor_r2p * radii)[:, None],
            periodic        = periodic,
            add_dummy_node  = add_dummy_node,
            with_additional_info = with_additional_info
        )
        

        return cls(physical_to_regional, regional_to_regional, regional_to_physical)
    