"""Utility functions for reading the datasets."""
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Sequence, NamedTuple, Literal
from copy import deepcopy

@dataclass
class Metadata:
  periodic: bool
  group_u: str
  group_c: str
  group_x: str
  type: Literal['poseidon', 'rigno']
  fix_x: bool
  domain_x: tuple[Sequence[int], Sequence[int]]
  domain_t: tuple[int, int]
  active_variables: Sequence[int]  # Index of variables in input/output
  chunked_variables: Sequence[int]  # Index of variable groups
  num_variable_chunks: int  # Number of variable chunks
  signed: dict[str, Union[bool, Sequence[bool]]]
  names: dict[str, Sequence[str]]
  global_mean: Sequence[float]
  global_std: Sequence[float]

ACTIVE_VARS_NS = [0, 1]
ACTIVE_VARS_CE = [0, 1, 2, 3]
ACTIVE_VARS_GCE = [0, 1, 2, 3, 5]
ACTIVE_VARS_RD = [0]
ACTIVE_VARS_WE = [0]
ACTIVE_VARS_PE = [0]

CHUNKED_VARS_NS = [0, 0]
CHUNKED_VARS_CE = [0, 1, 1, 2, 3]
CHUNKED_VARS_GCE = [0, 1, 1, 2, 3, 4]
CHUNKED_VARS_RD = [0]
CHUNKED_VARS_WE = [0]
CHUNKED_VARS_PE = [0]

SIGNED_NS = {'u': [True, True], 'c': None}
SIGNED_CE = {'u': [False, True, True, False, False], 'c': None}
SIGNED_GCE = {'u': [False, True, True, False, False, False], 'c': None}
SIGNED_RD = {'u': [True], 'c': None}
SIGNED_WE = {'u': [True], 'c': [False]}
SIGNED_PE = {'u': [True], 'c': [True]}

NAMES_NS = {'u': ['$v_x$', '$v_y$'], 'c': None}
NAMES_CE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$'], 'c': None}
NAMES_GCE = {'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', 'E', '$\\phi$'], 'c': None}
NAMES_RD = {'u': ['$u$'], 'c': None}
NAMES_WE = {'u': ['$u$'], 'c': ['$c$']}
NAMES_PE = {'u': ['$u$'], 'c': ['$f$']}

DATASET_METADATA = {
  # incompressible_fluids: [velocity, velocity]
  'incompressible_fluids/brownian_bridge': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/gaussians': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/pwc': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/shear_layer': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/sines': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'incompressible_fluids/vortex_sheet': Metadata(
    periodic=True,
    group_u='velocity',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  # compressible_flow: [density, velocity, velocity, pressure, energy]
  'compressible_flow/gauss': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 2.513],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/kh': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.0],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/richtmyer_meshkov': Metadata(
    periodic=True,
    group_u='solution',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 2),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[1.1964245, -7.164812e-06, 2.8968952e-06, 1.5648036],
    global_std=[0.5543239, 0.24304213, 0.2430597, 0.89639103],
  ),
  'compressible_flow/riemann': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.215],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/riemann_curved': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.553],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/riemann_kh': Metadata(
    periodic=True,
    group_u='data',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 1.33],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'compressible_flow/gravity/rayleigh_taylor': Metadata(
    periodic=True,
    group_u='solution',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 5),
    fix_x=True,
    active_variables=ACTIVE_VARS_GCE,
    chunked_variables=CHUNKED_VARS_GCE,
    num_variable_chunks=len(set(CHUNKED_VARS_GCE)),
    signed=SIGNED_GCE,
    names=NAMES_GCE,
    global_mean=[0.8970493, 4.0316996e-13, -1.3858967e-13, 0.7133829, -1.7055787],
    global_std=[0.12857835, 0.014896976, 0.014896975, 0.21293919, 0.40131348],
  ),
  # reaction_diffusion
  'reaction_diffusion/allen_cahn': Metadata(
    periodic=False,
    group_u='solution',
    group_c=None,
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 0.0002),
    fix_x=True,
    active_variables=ACTIVE_VARS_RD,
    chunked_variables=CHUNKED_VARS_RD,
    num_variable_chunks=len(set(CHUNKED_VARS_RD)),
    signed=SIGNED_RD,
    names=NAMES_RD,
    global_mean=[0.002484262],
    global_std=[0.65351176],
  ),
  # wave_equation
  'wave_equation/seismic_20step': Metadata(
    periodic=False,
    group_u='solution',
    group_c='c',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.03467443221585092],
    global_std=[0.10442421752963911],
  ),
  'wave_equation/gaussians_15step': Metadata(
    periodic=False,
    group_u='solution',
    group_c='c',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.0334376316],
    global_std=[0.1171879068],
  ),
  # poisson_equation
  'poisson_equation/sines': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  'poisson_equation/chebyshev': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  'poisson_equation/pwc': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  'poisson_equation/gaussians': Metadata(
    periodic=False,
    group_u='solution',
    group_c='source',
    group_x=None,
    type='poseidon',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
  # steady Euler
  'rigno-unstructured/airfoil_grid': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x=None,
    type='poseidon',
    domain_x=([-.75, -.75], [1.75, 1.75]),
    domain_t=None,
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False]},
    names={'u': ['$\\rho$'], 'c': ['$d$']},
    global_mean=[0.92984116],
    global_std=[0.10864315],
  ),
  # rigno-unstructured
  'rigno-unstructured/airfoil_li': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-1, -1], [2, 1]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],  # Only the density
    chunked_variables=[0, 1, 1, 2, 3],
    num_variable_chunks=4,
    signed={'u': [False, True, True, False, False], 'c': [False]},
    names={'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', '$Ma$'], 'c': ['$d$']},
    global_mean=[0.9637927979586245],
    global_std=[0.11830822800242624],
  ),
  'rigno-unstructured/airfoil_li_large': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-3, -3], [+5, +3]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],  # Only the density
    chunked_variables=[0, 1, 1, 2, 3],
    num_variable_chunks=4,
    signed={'u': [False, True, True, False, False], 'c': [False]},
    names={'u': ['$\\rho$', '$v_x$', '$v_y$', '$p$', '$Ma$'], 'c': ['$d$']},
    global_mean=[0.9637927979586245],
    global_std=[0.11830822800242624],
  ),
  'rigno-unstructured/elasticity': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=False,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [False], 'c': [False]},
    names={'u': ['$\\sigma$'], 'c': ['$d$']},
    global_mean=[187.477],
    global_std=[127.046],
  ),
  'rigno-unstructured/poisson_c_sines': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=None,
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': [True]},
    names={'u': ['$u$'], 'c': ['$f$']},
    global_mean=[0.],
    global_std=[0.00064911455],
  ),
  'rigno-unstructured/wave_c_sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0.],
    global_std=[0.011314605],
  ),
  'rigno-unstructured/wave_c_sines_uv': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([-.5, -.5], [1.5, 1.5]),
    domain_t=(0, 0.1),
    fix_x=True,
    active_variables=[0, 1],
    chunked_variables=[0, 1],
    num_variable_chunks=2,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[0., 0.],
    global_std=[0.004625, 0.3249],
  ),
  'rigno-unstructured/heat_l_sines': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0., 0.], [1., 1.]),
    domain_t=(0, 0.002),
    fix_x=True,
    active_variables=[0],
    chunked_variables=[0],
    num_variable_chunks=1,
    signed={'u': [True], 'c': None},
    names={'u': ['$u$'], 'c': None},
    global_mean=[-0.009399102],
    global_std=[0.020079814],
  ),
  'rigno-unstructured/NS-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/NS-PwC': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/NS-SL': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/NS-SVS': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_NS,
    chunked_variables=CHUNKED_VARS_NS,
    num_variable_chunks=len(set(CHUNKED_VARS_NS)),
    signed=SIGNED_NS,
    names=NAMES_NS,
    global_mean=[0.0, 0.0],
    global_std=[0.391, 0.356],
  ),
  'rigno-unstructured/CE-Gauss': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 2.513],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'rigno-unstructured/CE-RP': Metadata(
    periodic=True,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_CE,
    chunked_variables=CHUNKED_VARS_CE,
    num_variable_chunks=len(set(CHUNKED_VARS_CE)),
    signed=SIGNED_CE,
    names=NAMES_CE,
    global_mean=[0.80, 0., 0., 0.215],
    global_std=[0.31, 0.391, 0.356, 0.185],
  ),
  'rigno-unstructured/ACE': Metadata(
    periodic=False,
    group_u='u',
    group_c=None,
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 0.0002),
    fix_x=True,
    active_variables=ACTIVE_VARS_RD,
    chunked_variables=CHUNKED_VARS_RD,
    num_variable_chunks=len(set(CHUNKED_VARS_RD)),
    signed=SIGNED_RD,
    names=NAMES_RD,
    global_mean=[0.002484262],
    global_std=[0.65351176],
  ),
  'rigno-unstructured/Wave-Layer': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=(0, 1),
    fix_x=True,
    active_variables=ACTIVE_VARS_WE,
    chunked_variables=CHUNKED_VARS_WE,
    num_variable_chunks=len(set(CHUNKED_VARS_WE)),
    signed=SIGNED_WE,
    names=NAMES_WE,
    global_mean=[0.03467443221585092],
    global_std=[0.10442421752963911],
  ),
  'rigno-unstructured/Poisson-Gauss': Metadata(
    periodic=False,
    group_u='u',
    group_c='c',
    group_x='x',
    type='rigno',
    domain_x=([0, 0], [1, 1]),
    domain_t=None,
    fix_x=True,
    active_variables=ACTIVE_VARS_PE,
    chunked_variables=CHUNKED_VARS_PE,
    num_variable_chunks=len(set(CHUNKED_VARS_PE)),
    signed=SIGNED_PE,
    names=NAMES_PE,
    global_mean=[0.0005603458434937093],
    global_std=[0.02401226126952699],
  ),
}
