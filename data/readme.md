# Datasets with unstructured mesh

## General descriptions

### Dataset groups
There are two or three groups in each file:
  - `u`: the vectorial solution function
  - `c`: the parameters (available but not part of the model predictions)
  - `x`: spatial coordinates in all directions (x and y)
All these arrays are 4-dimensional with the following axes: `[batch, time, space, variables]`.

All time-dependent datasets (except `Wave-Layer.nc`) have 21 uniformly spaced time snapshots.
The first snapshot is the initial condition.
For time-independent datasets, the size of the time axis is always 1.

Some of the time-dependent datasets do not have known parameters (`c`), for these datasets, only the initial condition in `u`
must be used as the input. When `c` is available, it should be concatenated to the initial condition.
For time-independent datasets, the input is only `c` (e.g., source function, wave propagation speed, etc.) and the output is `u`.

The space coordinates are ordered randomly. The size of the time axis of `x` is always one, even for time-dependent problems.
These coordinates are constant along time. In other words, the same coordinates
should be used for all time snapshots of a single trajectory.
Except for `airfoil_li.nc`, `airfoil_li_large.nc`, and `elasticity.nc`, the coorindates are also constant for all
the samples or trajectories along the batch axis.

### Training-validation-test split
- **time-dependent datasets**: Let's train with 1024 training trajectories, validate with the following 128 samples, and test with the **last** 256 samples.
- **time-independent datasets**: Let's train with 2048 training trajectories, validate with the following 128 samples, and test with the **last** 256 samples.

## File-specific descriptions

### `airfoil_li.nc` and `airfoil_li_large.nc`: Steady-state transonic flow over an airfoil [1]
- The original dataset from [1] is augmented with a geometric parameter (group `c`) which is the distance of each coordinate from the boundary of the airfoil.
- Although other variables (density, velocity components, and pressure) are available, we only predict the density (first variable).
- In `airfoil_li_large.nc`, a larger domain around the airfoil is considered.

### `elasticity.nc`: Tensile Stress in Hyper-elastic material [1]
- The original dataset from [1] is augmented with a geometric parameter (group `c`) which is the distance of each coordinate from the boundary of the central hole in the specimen.
- Since this dataset has only 2000 samples, we exceptionally train with 1024, validate with 128, and test with 256 samples.

### `heat_l_sines.nc`: Heat equation in an L-shaped domain
- There is no known parameter as the heat equation is considered without any source.

- Initial condition are sampled as below with $d=16$ and $\mu \sim \operatorname{Unif}([-1,1]^d)$.
$$
\begin{gathered}
u\left(0, x_1, x_2, \mu\right) = u_0\left(x_1, x_2, \mu\right) = \frac{1}{d} \sum_{m=1}^d u_0^m\left(x_1, x_2, \mu_m\right) = -\frac{1}{d} \sum_{m=1}^d \mu_m \sin \left(\pi m x_1\right) \sin \left(\pi m x_2\right) / m^{0.5} \\
u_0^m\left(x_1, x_2, \mu_m\right) = -\mu_m \sin \left(\pi m x_1\right) \sin \left(\pi m x_2\right) / m^{0.5}
\end{gathered}
$$

- Zero Dirichlet boundary conditions are considered.

### `wave_c_sines.nc`: Heat equation in a circular domain
- There is no known parameter. The propagation speed is constant everywhere.

- Initial condition are sampled as below with $K=10$ and $a \in \mathbb{R}^{K \times K}$, with values uniformly sampled from the range $[-1, 1]$.
$$
u_0(x, y) = \frac{\pi}{K^2} \sum_{i, j} a_{i j}\left(i^2+j^2\right)^{-r} \sin (\pi i x) \sin (\pi j y).
$$

- Zero Dirichlet boundary conditions are considered.

### `poisson_c_sines.nc`: Poisson's equation in a circular domain

- The source function is stored in `c`.

- The source term is sampled as below with $K=16$, and $a_{ij}$ uniformly sampled from $[-1, +1]$.
$$
f(x, y) = \frac{\pi}{K^2} \sum_{i, j=1}^K a_{ij} \cdot (i^2 + j^2)^{-r} \sin(\pi i x) \sin(\pi j y), \quad \forall (x, y) \in D,
$$

- Zero Dirichlet boundary conditions are considered.

### Poseidon datasets [2]

The Poseidon datasets (`ACE.nc`, `Wave-Layer.nc`, `Poisson-Gauss.nc`, `NS-Gauss.nc`, `NS-SL.nc`, `NS-SVS.nc`, `NS-PwC.nc`, `CE-RP.nc`, `CE-Gauss.nc`) are originally generated on a uniform 128x128 grid in $\Omega_x=[0, 1]^2$.
The coordinates are computed based on the grid, and then the structure is distroyed by randomly shuffling them. Note that the random shuffle is different for each sample, but constant in time.

There are 16384 (128x128) points in the random point cloud. For benchmarking architectures that work on arbitrary point clouds, we only take the first 9216 (96x96) points in this dataset. Note that we only do this for Poseidon datasets to avoid using full-grid coordinates. For the original unstructured datasets, we use all coordinates.

In `CE-Gauss.nc`, the last variable is "energy" and must be excluded from all trainings and tests.

## Metrics

Are evaluations should be done with the metrics that are defined in [2]. All arrays must be normalized before evaluation using "equation-wide" global means and standard deviations which can be found [here](https://github.com/sprmsv/rigno/blob/298858abc5b3c664763989bb72bad8288163132e/rigno/dataset.py#L68) for each dataset.

Here are the steps for calculating the metric:
1. Normalize all data using global statistics.
2. Compute relative l1-norm for each variable chunk. (reduce on the space axis)
3. Take the median over the batch axis. (reduce on the batch axis)
4. Compute the average of variable chunks. (reduce on the variable axis)

## References
[1] Li, Zongyi, et al. "Fourier neural operator with learned deformations for pdes on general geometries." Journal of Machine Learning Research 24.388 (2023): 1-26.

[2] Herde, Maximilian, et al. "Poseidon: Efficient Foundation Models for PDEs." arXiv preprint arXiv:2405.19101 (2024).
