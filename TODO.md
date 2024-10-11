**已测试结论：**
NS-Gauss:
1. 不同的latent grid/patch size + batch size  -> 模型的效果表现的并不好，仍旧是无法学习，模型很容易掉入局部最优

NS-Gauss/NS-PwC:
1. 模型的效果表现的更好一些了，但仍旧是学习到一定阶段后无法收敛

目前发起的实验：
1. rfno的两个实验
2. gino的两个实验
3. 

LANO的实验中，感觉增大latent sizes效果并没有明显的改进，怀疑还是优化的问题，测试改变一下optimizer试一下

**待做:**- 
- [x] 实现RFNO，这个模型至少可以用来快速测试我的训练是否make sense，使用相同的encoder和decoder配置，如果RFNO能够work的很好的话，那么RANO也应该work的很好。所以这也间接说明了model的参数初始化可能很重要。**running**
- [x] 实现LANO，这个模型可以用来测试一下我的训练是否make sense。
    - [x] 首先，对于processor进去的部分，应该采用一个lifting process，这个lifting process加载encoder模型的末尾
    - [ ] 在研究一下数据集的readme，检查一下airfoil和poisson那几个数据集是否正确。结果给我的感觉非常不make sense。感觉需要对wave_c_circle_sines的输入和输出进行一下缩放。不然总感觉loss似乎已经到头了，但是还是学不会。
    - [ ] 可以新设计一个网络，将rigno那些structural information考虑进去。具体如下：在GNO中的kernel function，输入使用latent structures，以及edge information。然后对于输入，先过一个lifting
    - [x] 没必要每一次都重新计算一次连接关系矩阵，直接存好，这个可以用我之前写的那段代码 **这个在之后如果一个batch中网格是不规则的需要修改一下**

 - [x] 测试一下LANO在所有数据集上的性能
 - [ ] 接着，选定一个数据集
    - [ ] 对ce-gauss和Poisson_c_sines做一下消融实验

- [ ] 

- [ ] 步进学习的策略就是max_time steps一开始不用取那么大，先用一个level，再逐步增大
- [ ] 全resolution学习
- [ ] 实现一下RscoT，也就是下载一下poseidon的模型参数，直接加载processor里面，然后在小数据上做微调
- [x] 选择一个模型来做微调，建议用NS-gauss和Ns-pwc中的一个：
    - [x] latent grid，同时改变一下window size
    - [x] batch_size大小，这也是训练中很重要的一步，防止模型参数过拟合
    - [x] p2r和r2p的overlap factor
- [ ] 读一下Transformer的几篇论文，总结整理一下
- [ ] 在整理一下代码的架构，好好思考一下相应的策略（从头到尾好好梳理一遍）
- [ ] 先检查一下相应的结果，检查一下代码的正确性，尤其是time-conditioned layer的部分。那部分的代码可以被精简再优化一下。
- [ ] visualize一下相关的graph，弄清楚edges，minimal support和graph那个地方的代码
- [x] output_points那个地方是None，也就是模型没有建立新的graph，而是对原有的graph做flip，这点也要fix。

- [x] 把adamw那部分的代码完成，并且配上相应的scheduler。
- [x] 修正time-conditioned layer，没有啥大问题，主要是def forward(self, c, x):这个地方的名称和原文件中不匹配。此外，需要在代码中指示这一部分。
- [x] 每一次进来都会执行batch_broadcast操作，这个应该可以被进一步优化吧。-> 这个地方应该是inplace的，所以没有问题。
- [x] domain_shift这个地方没有搞懂它是怎么work的。主要是rigraph那一部分，建图的各个步骤到底是怎么work的，这点很重要。
- [x] 修正drop edge
    - [x] 关于drop edge这一部分，确定inference那个地方没有drop edge，这点很重要

**观点：**
 1. 感觉基于FNO的模型收敛速度很快，但模型很容易在训练集上过拟合。这是加了物理先验的原因，但是模型有很高的上限。基于transformer的模型，模型的能力上限似乎很大，并且不容易在训练数据集上过拟合。