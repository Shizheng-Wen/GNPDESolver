**Ongoing experiments:**
- [x] `config_lano_ce_gauss_learnrate.json`的实验继续，因为之前的模型并没有收敛，所以继续进行学习，看一下继续学习能否进一步优化模型。
    - 进一步优化感觉也没什么用，学习率基本上收敛不下降了。
- [x] `config_lano_time_conditionedlayer.json`的实验，仅在attention层添加了：
    - 实验结果感觉没啥用，模型表现更差了，误差到了50%，模型基本上很快就局部收敛了。


**待做:**- 
- [ ] 将不想管的config和结果全部都删掉
- [ ] 对于processor部分，施加新的components，具体来说，将[U-ViT](https://arxiv.org/pdf/2209.12152) 以及ScoT施加进来，这周找[bogdan](https://arxiv.org/pdf/2409.18359)探讨一下这个方面。
- [ ] 对于poseidon，这周末微调一下这个模型在不规则数据集上的表现，微调一下encoder和decoder
- [ ] 这周检查一下代码，重新整理一下，确保每一个components都work。
    - [ ] 检查一下metric那个地方用的对不对。
    - [ ] 为什么提高batch_size没有提升模型训练性能，但是显存开销却提升了，这个很诡异，因为模型的训练速度基本都没变。
    - [ ] 模型重视容易在训练数据集上过拟合
- [ ] 这周figure一下GNO和Transformer那几篇文章，Sid认为encoder那个地方得换，他不是很信赖GNO这种方法。
    - [ ] 可以在GNO那个地方加入GAT的机制，用attention来确定应该regional points should focus哪些physical points。
- [ ] incremental training的策略可以使用，把目标和方法再整理一下
- [ ] 就是我发现一个现象，就是对于那几个数据集，NS_gauss，CE_RP以及CE_Gauss，模型很容易在训练数据集上过拟合，并且训练数据集上loss下降很快，但是在测试数据集上loss完全不动的，感觉似乎训练数据集和测试数据集不属于相同的数据分布，很奇怪。一个比较直接的方法就是：
    - [ ] 重新测试一下数据分布。
    - [ ] 仔细研究一下表现比较差的那几个数据集
    - [ ] 对于这几组数据，换一下数据集重新测试一下,或者用更多的数据集测试一下，其实也可以用不同的batch_size测试一下
    - [ ] 对于CE那两组数据，尝试不用那么多channels，测试一下效果。




- [x] 修正一下time-conditioned，看一下max怎么做的，然后跑一个时空的数据集(就用ce_gauss) 看一下。接着再再这个ce_gauss的数据集上测试一下消融实验
- [x] 重新写一个trainer，专门给不规则的网格来使用，然后将静态的几个数据集全部跑完，包括gino的和LANO的
    - [x] fix 一下airfoil grid的训练，目前只是用把坐标加在input的方式进行训练
- [ ] 将GINO的几个Time-dependent dataset全部重新训练一下，看看效果

- [ ] 几个新的trick:
    - [x] 施加一下rope策略:换了一下rope，并且尝试了用patch size = 2的情况，模型的性能仍旧无法得到提升
    - [ ] 将position encoding之后的数据送进GNO中
    - [ ] 采用incremental training的策略，one step pair训练完后，再上multi-step继续精调模型
    - [ ] Poseidon pretrain一下




- [x] 先检查一下新训练的结果，看看哪些数据集的表现还是很差
- [x] 再检查一下ce—gauss ablation studies那几个结果
- [x] 将testing的度量重新写一下，检查一下模型.
    - [x] direct step有点问题，fix一下
    - [x] 给一个选项，直接一下子求三个度量，并存在database里面，然后再把所有模型的最优统计一下
    - [x] 给一个选项，不用自己求的u_mean,直接用他们提供的。
    - [x] 并把config文件中所有的参数全部写进一个文本中，用来merge，这样可以指代默认值。
- [x] 实现RFNO，这个模型至少可以用来快速测试我的训练是否make sense，使用相同的encoder和decoder配置，如果RFNO能够work的很好的话，那么RANO也应该work的很好。所以这也间接说明了model的参数初始化可能很重要。**running**
- [x] 实现LANO，这个模型可以用来测试一下我的训练是否make sense。
    - [x] 首先，对于processor进去的部分，应该采用一个lifting process，这个lifting process加载encoder模型的末尾
    - [x] 在研究一下数据集的readme，检查一下airfoil和poisson那几个数据集是否正确。结果给我的感觉非常不make sense。感觉需要对wave_c_circle_sines的输入和输出进行一下缩放。不然总感觉loss似乎已经到头了，但是还是学不会。
    - [ ] 可以新设计一个网络，将rigno那些structural information考虑进去。具体如下：在GNO中的kernel function，输入使用latent structures，以及edge information。然后对于输入，先过一个lifting
    - [x] 没必要每一次都重新计算一次连接关系矩阵，直接存好，这个可以用我之前写的那段代码 **这个在之后如果一个batch中网格是不规则的需要修改一下**

 - [x] 测试一下LANO在所有数据集上的性能
 - [x] 接着，选定一个数据集
    - [x] 对ce-gauss和Poisson_c_sines做一下消融实验

 

- [ ] 步进学习的策略就是max_time steps一开始不用取那么大，先用一个level，再逐步增大
- [ ] 全resolution学习
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
 NS-Gauss:
1. 不同的latent grid/patch size + batch size  -> 模型的效果表现的并不好，仍旧是无法学习，模型很容易掉入局部最优
2. 不知道为什么带Gauss的部分模型效果都表现的那么差，感觉是Gauss这部分数据集的问题，没有产生足够的数据集分布。

NS-Gauss/NS-PwC:
1. 模型的效果表现的更好一些了，但仍旧是学习到一定阶段后无法收敛