from typing import Union, List, Optional
from numbers import Number
from copy import deepcopy
import itertools

from torch import nn
import torch
import torch.nn.functional as F

import opt_einsum
import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')
use_opt_einsum('optimal')


###########
# Base_Spectral_Conv
###########

class BaseSpectralConv(nn.Module):
    def __init__(self, device=None, dtype=None):
        """Base Class for Spectral Convolutions
        
        Use it when you want to build your own FNO-type Neural Operators
        """
        super().__init__()

        self.dtype = dtype
        self.device = device

    def transform(self, x):
        """Transforms an input x for a skip connection, by default just an identity map 

        If your function transforms the input then you should also implement this transform method 
        so the skip connection can also work. 

        Typical usecases are:

        * Your upsample or downsample the input in the Spectral conv: the skip connection has to be similarly scaled. 
          This allows you to deal with it however you want (e.g. avoid aliasing)
        * You perform a change of basis in your Spectral Conv, again, this needs to be applied to the skip connection too.
        """
        return x


###########
# Complex
###########
"""
Functionality for handling complex-valued spatial data
"""

def CGELU(x: torch.Tensor):
    """Complex GELU activation function
    Follows the formulation of CReLU from Deep Complex Networks (https://openreview.net/pdf?id=H1T2hmZAb)
    apply GELU is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """

    return F.gelu(x.real).type(torch.cfloat) + 1j * F.gelu(x.imag).type(
        torch.cfloat
    )


def ctanh(x: torch.Tensor):
    """Complex-valued tanh stabilizer
    Apply ctanh is real and imag part of the input separately, then combine as complex number
    Args:
        x: complex tensor
    """
    return torch.tanh(x.real).type(torch.cfloat) + 1j * torch.tanh(x.imag).type(
        torch.cfloat
    )


def apply_complex(real_func, imag_func, x, dtype=torch.cfloat):
    """
    fr: a function (e.g., conv) to be applied on real part of x
    fi: a function (e.g., conv) to be applied on imag part of x
    x: complex input.
    """
    return (real_func(x.real) - imag_func(x.imag)).type(dtype) + 1j *\
          (real_func(x.imag) + imag_func(x.real)).type(
        dtype
    )

class ComplexValued(nn.Module):
    """
    Wrapper class that converts a standard nn.Module that operates on real data
    into a module that operates on complex-valued spatial data.
    """

    def __init__(self, module):
        super(ComplexValued, self).__init__()
        self.fr = deepcopy(module)
        self.fi = deepcopy(module)

    def forward(self, x):
        return apply_complex(self.fr, self.fi, x) 

###########
# einsum_utils
###########

def einsum_complexhalf_two_input(eq, a, b):
    """
    Compute (two-input) einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    assert len(eq.split(',')) == 2, "Equation must have two inputs."

    # cast both tensors to "view as real" form, and half precision
    a = torch.view_as_real(a)
    b = torch.view_as_real(b)
    a = a.half()
    b = b.half()

    # create a new einsum equation that takes into account "view as real" form
    input_output = eq.split('->')
    new_output = 'xy' + input_output[1]
    input_terms = input_output[0].split(',')
    new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
    new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

    # convert back to complex form
    tmp = tl.einsum(new_eqn, a, b)
    res = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
    return torch.view_as_complex(res)

def einsum_complexhalf(eq, *args):
    """
    Compute einsum for complexhalf tensors.
    Because torch.einsum currently does not support complex32 (complexhalf) types.
    The inputs and outputs are the same as in torch.einsum
    """
    if len(args) == 2:
        # if there are two inputs, it is faster to call this method
        return einsum_complexhalf_two_input(eq, *args)

    # find the optimal path
    _, path_info = opt_einsum.contract_path(eq, *args)
    partial_eqns = [contraction_info[2] for contraction_info in path_info.contraction_list]

    # create a dict of the input tensors by their label in the einsum equation
    tensors = {}
    input_labels = eq.split('->')[0].split(',')
    output_label = eq.split('->')[1]
    tensors = dict(zip(input_labels,args))

    # convert all tensors to half precision and "view as real" form
    for key, tensor in tensors.items():
        tensor = torch.view_as_real(tensor)
        tensor = tensor.half()
        tensors[key] = tensor

    for partial_eq in partial_eqns:
        # get the input tensors to partial_eq
        in_labels, out_label = partial_eq.split('->')
        in_labels = in_labels.split(',')
        in_tensors = [tensors[label] for label in in_labels]

        # create new einsum equation that takes into account "view as real" form
        input_output = partial_eq.split('->')
        new_output = 'xy' + input_output[1]
        input_terms = input_output[0].split(',')
        new_inputs = [input_terms[0] + 'x', input_terms[1] + 'y']
        new_eqn = new_inputs[0] + ',' + new_inputs[1] + '->' + new_output

        # perform the einsum, and convert to "view as real" form
        tmp = tl.einsum(new_eqn, *in_tensors)
        result = torch.stack([tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1)
        tensors[out_label] = result

    return torch.view_as_complex(tensors[output_label])

###########
# embedding
###########
class GridEmbedding2D(nn.Module):
    """A simple positional embedding as a regular 2D grid
    """
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        """GridEmbedding2D applies a simple positional 
        embedding as a regular 2D grid

        Parameters
        ----------
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grid_x, grid_y = regular_grid_2d(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)
        out =  torch.cat((data, x.expand(batch_size, -1, -1, -1),
                          y.expand(batch_size, -1, -1, -1)),
                         dim=1)
        # in the unbatched case, the dataloader will stack N 
        # examples with no batch dim to create one
        if not batched and batch_size == 1: 
            return out.squeeze(0)
        else:
            return out

class GridEmbeddingND(nn.Module):
    """A positional embedding as a regular ND grid
    """
    def __init__(self, dim: int=2, grid_boundaries=[[0, 1], [0, 1]]):
        """GridEmbeddingND applies a simple positional 
        embedding as a regular ND grid

        Parameters
        ----------
        dim: int
            dimensions of positional encoding to apply
        grid_boundaries : list, optional
            coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """
        super().__init__()
        self.dim = dim
        assert self.dim == len(grid_boundaries), f"Error: expected grid_boundaries to be\
            an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims: torch.Size, device: str, dtype: torch.dtype):
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.Size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grids_by_dim = regular_grid_nd(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            # add batch, channel dims
            grids_by_dim = [x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids_by_dim]
            self._grid = grids_by_dim
            self._res = spatial_dims

        return self._grid

    def forward(self, data, batched=True):
        """
        Params
        --------
        data: torch.Tensor
            assumes shape batch (optional), channels, x_1, x_2, ...x_n
        batched: bool
            whether data has a batch dim
        """
        # add batch dim if it doesn't exist
        if not batched:
            if data.ndim == self.dim + 1:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        grids = self.grid(spatial_dims=data.shape[2:],
                          device=data.device,
                          dtype=data.dtype)
        grids = [x.repeat(batch_size, *[1] * (self.dim+1)) for x in grids]
        out =  torch.cat((data, *grids),
                         dim=1)
        return out
    
class SinusoidalEmbedding2D(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        """SinusoidalEmbedding2D applies a 2d sinusoidal positional encoding 

        Parameters
        ----------
        num_channels : int
            number of input channels
        max_positions : int, optional
            maximum positions to encode, by default 10000
        endpoint : bool, optional
            whether to set endpoint, by default False
        """
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        """
        Applying rotary positional embedding (https://arxiv.org/abs/2104.09864) to the input feature tensor.
        The crux is the dot product of two rotation matrices R(theta1) and R(theta2) is equal to R(theta2 - theta1).
        """
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, coordinates):
        """coordinates is tensor of [batch_size, num_points]"""
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', coordinates, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]

    @staticmethod
    def apply_1d_rotary_pos_emb(t, freqs):
        return apply_rotary_pos_emb(t, freqs)

    @staticmethod
    def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
        """Split the last dimension of features into two equal halves
           and apply 1d rotary positional embedding to each half."""
        d = t.shape[-1]
        t_x, t_y = t[..., :d//2], t[..., d//2:]

        return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                          apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)
    

# Utility functions for GridEmbedding
def regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Creates a 2 x height x width stack of positional encodings A, where
    A[:,i,j] = [[x,y]] at coordinate (i,j) on a (height, width) grid. 
    """
    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1],
                        height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1],
                        width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    grid_x = grid_x.repeat(1, 1)
    grid_y = grid_y.repeat(1, 1)

    return grid_x, grid_y

def regular_grid_nd(resolutions: List[int], grid_boundaries: List[List[int]]=[[0,1]] * 2):
    """regular_grid_nd generates a tensor of coordinate points that 
    describe a bounded regular grid.
    
    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid. 

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding 
    """
    assert len(resolutions) == len(grid_boundaries), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start,stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(torch.linspace(start, stop, res + 1)[:-1])
    grid = torch.meshgrid(*meshgrid_inputs, indexing='ij')
    grid = tuple([x.repeat([1]*dim) for x in grid])
    return grid

  
# Utility fucntions for Rotary embedding
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
def rotate_half(x):
    """
    Split x's channels into two equal halves.
    """
    # split the last dimension of x into two equal halves
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotation matrix computed based on freqs to rotate t.
    t: tensor of shape [batch_size, num_points, dim]
    freqs: tensor of shape [batch_size, num_points, 1]

    Formula: see equation (34) in https://arxiv.org/pdf/2104.09864.pdf
    """
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


###########
# Normalization_layers
###########
class AdaIN(nn.Module):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Linear(512, 2*in_channels)
            )
        self.mlp = mlp

        self.embedding = None
    
    def set_embedding(self, x):
        self.embedding = x.reshape(self.embed_dim,)

    def forward(self, x):
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(self.mlp(self.embedding), self.in_channels, dim=0)

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)

class InstanceNorm(nn.Module):
    def __init__(self, **kwargs):
        """InstanceNorm applies dim-agnostic instance normalization
        to data as an nn.Module. 

        kwargs: additional parameters to pass to instance_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.kwargs = kwargs
    
    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x

###########
# resample
###########
def resample(x, res_scale, axis, output_shape=None):
    """
    A module for generic n-dimentional interpolation (Fourier resampling).

    Parameters
    ----------
    x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic 
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed.
    output_shape : None or tuple[int]
    """

    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale]*len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
              res_scale = [res_scale]*len(axis)
    else:
        assert len(res_scale) == len(axis), "leght of res_scale and axis are not same"

    old_size = x.shape[-len(axis):]
    if output_shape is None:
        new_size = tuple([int(round(s*r)) for (s, r) in zip(old_size, res_scale)])
    else:
        new_size = output_shape

    if len(axis) == 1:
        return F.interpolate(x, size=new_size[0], mode='linear', align_corners=True)
    if len(axis) == 2:
        return F.interpolate(x, size=new_size, mode='bicubic', align_corners=True)

    X = torch.fft.rfftn(x.float(), norm='forward', dim=axis)
    
    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1]//2 + 1 # Redundant last coefficient
    new_fft_size_c = [min(i,j) for (i,j) in zip(new_fft_size, X.shape[-len(axis):])]
    out_fft = torch.zeros([x.shape[0], x.shape[1], *new_fft_size], device=x.device, dtype=torch.cfloat)

    mode_indexing = [((None, m//2), (-m//2, None)) for m in new_fft_size_c[:-1]] + [((None, new_fft_size_c[-1]), )]
    for i, boundaries in enumerate(itertools.product(*mode_indexing)):

        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

        out_fft[idx_tuple] = X[idx_tuple]
    y = torch.fft.irfftn(out_fft, s= new_size ,norm='forward', dim=axis)

    return y


def iterative_resample(x, res_scale, axis):
    if isinstance(axis, list) and isinstance(res_scale, (float, int)):
        res_scale = [res_scale]*len(axis)
    if not isinstance(axis, list) and isinstance(res_scale,list):
      raise Exception("Axis is not a list but Scale factors are")
    if isinstance(axis, list) and isinstance(res_scale,list) and len(res_scale)!=len(axis):
      raise Exception("Axis and Scal factor are in different sizes")

    if isinstance(axis, list):
        for i in range(len(res_scale)):
            rs = res_scale[i]
            a = axis[i]
            x = resample(x, rs, a)
        return x

    old_res = x.shape[axis]
    X = torch.fft.rfft(x, dim=axis, norm='forward')    
    newshape = list(x.shape)
    new_res = int(round(res_scale*newshape[axis]))
    newshape[axis] = new_res // 2 + 1

    Y = torch.zeros(newshape, dtype=X.dtype, device=x.device)

    modes = min(new_res, old_res)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, modes // 2 + 1)
    Y[tuple(sl)] = X[tuple(sl)]
    y = torch.fft.irfft(Y, n=new_res, dim=axis,norm='forward')
    return y


###########
# skip_connection
###########
def skip_connection(
    in_features, out_features, n_dim=2, bias=False, skip_type="soft-gating"
):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, optional
        whether to use a bias, by default False
    skip_type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return Flattened1dConv(in_channels=in_features,
                               out_channels=out_features,
                               kernel_size=1,
                               bias=bias,)
    elif skip_type.lower() == "identity":
        return nn.Identity()
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'id'}."
        )


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D.
    bias : bool, default is False
    """

    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but these two must be the same for soft-gating"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * x + self.bias
        else:
            return self.weight * x

class Flattened1dConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, bias=False):
        """Flattened3dConv is a Conv-based skip layer for
        input tensors of ndim > 3 (batch, channels, d1, ...) that flattens all dimensions 
        past the batch and channel dims into one dimension, applies the Conv,
        and un-flattens.

        Parameters
        ----------
        in_channels : int
            in_channels of Conv1d
        out_channels : int
            out_channels of Conv1d
        kernel_size : int
            kernel_size of Conv1d
        bias : bool, optional
            bias of Conv3d, by default False
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              bias=bias)
    def forward(self, x):
        # x.shape: b, c, x1, ..., xn x_ndim > 1
        size = list(x.shape)
        # flatten everything past 1st data dim
        x = x.view(*size[:2], -1)
        x = self.conv(x)
        # reshape x into an Nd tensor b, c, x1, x2, ...
        x = x.view(size[0], self.conv.out_channels, *size[2:])
        return x
        
###########
# Validate_scaling_factor
###########

def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        return [[float(s)] * n_dim for s in scaling_factor]
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None
