**已测试结论：**
NS-Gauss:
1. 不同的latent grid/patch size + batch size  -> 模型的效果表现的并不好，仍旧是无法学习，模型很容易掉入局部最优

NS-Gauss/NS-PwC:
1. 模型的效果表现的更好一些了，但仍旧是学习到一定阶段后无法收敛



**待做:**- 
- [x] 实现RFNO，这个模型至少可以用来快速测试我的训练是否make sense，使用相同的encoder和decoder配置，如果RFNO能够work的很好的话，那么RANO也应该work的很好。所以这也间接说明了model的参数初始化可能很重要。**running**
- [ ] 实现LANO，这个模型可以用来测试一下我的训练是否make sense。
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
