# Model
Currently, this model zoo supports the following four architectures, but I try to develop this repo with modularity, you can do the combination of these encoder, processor and decoder and generate the new architectures. Next, I will illustrate the difference between these models.

|Model | Encoder | Processor | Decoder|
-------| --------|-----------|--------|
|GINO| GNO | FNO | GNO|
|RIGNO| MP | MP | MP|
|MeshGraphNet| MLP | MP | MLP|
|AIFS | MP | Transformer| MP|


## GINO
Geometry informed Neural Operator
### Encoder
Support four different kernels:
|Kernels| Formula|
|-------|--------|
|Linear Kernel|$\int_{A(x)} k(x, y) d y$|
|Linear Kernel with input|$\int_{A(x)} k(x, y) \cdot f(y) d y$|
|Nonlinear kernel| $\int_{A(x)} k(x, y, f(y)) d y$|
|Nonlinear kernel with input|$\int_{A(x)} k(x, y, f(y)) \cdot f(y) d y$|

where $x$ is the specified point of the output function. $y$ is the specified point of the input function. The$y$ and $x$ can be different. $A(x)$ is that for every point $x$, we need a group of $y$ point to do the integration. So this is basically a local operator, not a non-local operator like Fourier Domain.

How to define the $A(x)$?
The answer is that we can Find the neighbors, in data, of each point in queries within a ball of radius, and store them into a dictionary with keys: "neighbors_index" and "neighbors_row_splits". In 3D, this process can be accelerated with open3d library.

After Build the neighbors connection, we should do:
1. concatenate the features [x, y, f(y)], because one x correponds to multiple y, therefore, x should be repeat the k number of times, the k means the number of neighbors.
2. Use an MLP to apply these concatenated features. which is kernel function. the input dimension is the [x,y,f(y)], but we have another dimension which can be the number of edges. You can view that these information is stored on the edge. 
3. Apply the numerical quadrature, because we want to calculate this integral, now we have already calculated the results for every $A(x)$. 
3. Finally, this information is like stored on the edge, therefore, we need to aggregate the information from these edges to the node. We need a Edge2Node process. His original code use [torch_scatter](https://github.com/rusty1s/pytorch_scatter/tree/master) to quickly implement this process. If you don't successfully install the torch_scatter, it can use the naive PyTorch implementation to achieve it.![torch scatter](assets/image.png)


### Processor
1. lifting
2. Just pass several ordinary Fourier Neural Operator block.

### Decoder
1. add positional embedding (optional).
2. pass the GNO
3. projection.

## RIGNO
### Encoder
1. Construct the Bipartite graph. 
    - In RIGNO, the node for regional node is randomly sampled from the physical nodes. 
    - apply medium to generate the radius of the regional node for physical node.
    - Construct the graph 
2. Aggregate the information using scatter.
    - original code use jraph.segment_mean. [Jraph](https://github.com/google-deepmind/jraph) is a library for building graph neural networks in JAX. 
    - Illustration: the original code use the TypedGraph structures implemented with JAX and Flax, allowing for multiple types of nodes and edges within the same graph. This flexibility is mainly essential for accurately modelling heterogeneous systems where different entities and interactions have distinct properties.

### Processor
1. Use Delaunay algorithm on these regional nodes. Then Randomly sample the points, and use scipy.spatial Delaunay to build new edges. This process will be continued with several times.
2. Do the message passing on this regional nodes.

### Decoder
1. Construct the Biparatite graph.
2. Do the MLP and aggregate the information.

## MeshGraphNet
### Encoder
Without changing the graph structures, just use MLP to do the channel encoding (lifting).
### Processor
Multilayer of Message Passing. 
### Decoder
MLP to do the projection.
## AIFS
### Encoder


### Processor
| Transformer | Characteristics | Complexity| Drawbacks|
|-------------|-----------------|-----------|----------|
|Transformer with flash attention|kernel fusion, directly process in the register and shared memory without saving intermediate results, avoid the access of global memory.|space: $\mathcal{O}(n)$ time:$\mathcal{O}(n^2)$| implementation|
|Swin Transformer| Shifted window attention | $\mathcal{O}(n^2)$ $\rightarrow$ $\mathcal{O}(n\cdot w^2)$| more layer and carefully parameter tunning|
|Local Attention| local regional attention calculation | $\mathcal{O}(n\cdot k)$ | hard to capture long-dependent data.|



