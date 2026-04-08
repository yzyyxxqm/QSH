# **面向不规则多元时间序列的四元数脉冲超图网络 (QSH-Net) 架构演进与 SOTA 性能优化指南**

## **核心理论审视与顶会评审标准对齐**

在当前的人工智能与机器学习领域，针对不规则多元时间序列（Irregular Multivariate Time Series, IMTS）的建模已成为极其关键且极具挑战性的研究前沿。四元数脉冲超图网络（Quaternion Spiking Hypergraph Network, 简称 QSH-Net）的理论构想展现了极高的学术野心。该模型试图通过融合三个高度复杂的子领域——用于捕获跨通道多维特征的四元数神经网络（QNN）、用于处理连续时间事件驱动动力学的脉冲神经网络（SNN），以及用于表征变量间非欧几里得高阶关系的超图神经网络（HGNN）——来彻底解决 IMTS 数据中固有的异步性、缺失性和多尺度颗粒度问题 1。

然而，要在国际顶级学术会议（如 NeurIPS、ICML、ICLR）上使这样一种高度复合的模型经受住严苛的同行评审并达到 State-of-the-Art (SOTA) 的表现，必须跨越巨大的经验与理论鸿沟。当前的顶会评审环境正面临着前所未有的压力与危机：随着提交量的指数级增长，评审质量的方差显著扩大，部分评审甚至依赖于大语言模型（LLM）生成模板化意见 3。在这种环境下，如果一个新模型仅仅是复杂模块的堆砌，而未能从底层数学逻辑上解决优化不稳定性，或者未能提供无可辩驳的消融实验与基准对比，它将极易被以“过度工程化（Over-engineered）”或“缺乏简单有效基线（e.g., 线性模型）对比”为由直接拒稿 4。经验证据一再表明，即便是在架构上极其强大的 Transformer 模型，在某些时间序列预测任务中也往往无法超越结构极其简单的线性回归模型 5。

因此，本研究报告旨在对 QSH-Net 进行详尽的、剥丝抽茧式的架构批判与重构。报告将深入剖析该模型在四元数空间反向传播、脉冲梯度代理、超图拓扑演化以及目标损失函数设计上的潜在数学缺陷，并结合 2025 至 2026 年间最新的 SOTA 研究成果（如 TFMixer、Hi-Patch、Directed HONN、PrimeNet 和 DBLoss 等），提供一套全面完善 QSH-Net 的工程与理论指南。

## **数据域解析：不规则多元时间序列（IMTS）的固有挑战**

要优化 QSH-Net，首先必须深刻理解其目标数据域的极端特殊性。现实世界中的 IMTS 数据（如医疗重症监护室的电子健康记录、天体物理观测数据或气候传感器网络数据）与标准规则时间序列存在本质区别。IMTS 的核心“怪异性（Peculiarity）”体现在两个维度：变量内采样间隔的不均匀性，以及跨变量采样率的异步性 1。

在这种数据环境下，传统的循环神经网络（RNN）和标准的 Transformer 架构面临失效的风险。RNN 依赖于固定的时间步长推进状态，面对异步数据时会产生严重的误差累积；而 Transformer 的全局注意力机制虽然强大，但其计算复杂度和内存消耗随序列长度呈超线性增长，且在处理极度稀疏的混合颗粒度（Mixed Granularity）数据时，往往无法提取出有效的时间动态特征 1。此外，许多现有图模型在处理 IMTS 时，依赖于将时间节点和变量节点进行强行对齐或插值填充（Padding）。这种方法不仅会显著增加数据维度并降低计算效率，更致命的是，插值会破坏原始数据的真实分布，导致模型学习到虚假的动态模式 1。因此，QSH-Net 的基础设计必须坚决摒弃任何形式的破坏性插值，转向完全的数据保真（Data Fidelity）建模范式。

## **四元数特征空间的数学重构：从理论缺陷到高效实现**

四元数神经网络（QNN）在 QSH-Net 中扮演着处理变量间复杂耦合关系的核心角色。四元数存在于四维超复数空间 ![][image1] 中，定义为 ![][image2]。相较于实数域的线性层，QNN 的核心优势在于它能够通过汉密尔顿乘积（Hamilton Product）在保持输入特征（如彩色图像的 RGB 通道，或临床数据中的心率、血压、血氧通道）内在相位和幅度关系的同时，直接学习跨通道的相关性 2。

### **汉密尔顿乘积的计算瓶颈与优化崩塌**

然而，在深层动态网络中部署 QNN 存在两个致命缺陷。首先是计算复杂度的爆炸。汉密尔顿乘积是非交换的（即 ![][image3]），其底层实现需要一系列固定的实数乘法和加法序列。在处理长周期、高维度的多元时间序列时，这会造成严重的计算显存瓶颈 9。其次，更为严重的是超复数空间中的梯度消失与优化不稳定问题。在反向传播过程中，如果激活函数的应用方式不当，四元数的梯度流极易相互干扰，导致模型无法收敛。

### **四元数模块的 SOTA 改进策略**

为了使 QSH-Net 的四元数模块达到顶会认可的 SOTA 标准，必须从代数底层进行重构。

为了突破计算瓶颈，QSH-Net 必须放弃原生的汉密尔顿乘积迭代计算，转而采用基于克罗内克积（Kronecker Product）的矩阵重构技术 11。对于四元数权重矩阵 ![][image4] 和输入 ![][image5]，其等价的实数值矩阵乘法可以通过 ![][image6] 来实现。这种重构利用了现成的、高度优化的实数值科学计算库（如 PyTorch 或 CUDA 底层接口），生成分块矩阵，从而在保留超复数几何特性的同时，将运算速度提升至接近标准实数网络的水平 11。

在非线性激活机制上，QSH-Net 必须严格采用分离激活函数（Split Activation Functions）。将激活函数（如 Sigmoid 或 Tanh）作为一个整体应用于四元数的幅值或相位，虽然在数学上看似优雅，但在工程实践中往往会阻断有效的梯度流动 9。相反，分离激活方法独立地将非线性变换应用于四元数的四个分量上：![][image7]。这种机制确保了每个独立分量的梯度能够单独回传，极大地提升了深层 QNN 的优化稳定性和收敛速度 9。

## **脉冲时间动力学（SNN）的深度优化与序列爆炸抑制**

脉冲神经网络（SNN）被引入 QSH-Net 是为了利用其事件驱动的天然属性。在 SNN 中，信息通过离散的脉冲（Spikes）进行传递，这与 IMTS 数据中非均匀分布、稀疏触发的观测事件在底层逻辑上完美契合 12。SNN 不仅能够提供极高的能源效率（在神经形态硬件上），而且能够原生处理连续时间流 13。

### **替代梯度（Surrogate Gradient）困境与长序列建模失效**

在实际应用中，SNN 的训练面临着巨大的理论障碍。由于神经元的脉冲发放通常由不可导的阶跃函数（如 Heaviside 函数）控制，传统的反向传播算法在此完全失效 15。研究界普遍采用替代梯度（Surrogate Gradient）方法来近似求导，但这一方法存在固有的梯度不匹配问题。在深度网络中，固定斜率的替代梯度往往会导致深层梯度的剧烈衰减或爆炸（即所谓的“死神经元”现象） 15。

此外，基础的泄漏积分发放（Leaky Integrate-and-Fire, LIF）神经元模型由于其固有的马尔可夫性质，难以捕获时间序列中跨越数百个时间步的长程依赖关系 13。当试图利用基于 Transformer 的位置编码来解决长序列问题时，传统的连续位置编码会破坏 SNN 的纯脉冲性质，导致模型在序列长度超过一定阈值（如 100 步）时性能骤降 16。

### **脉冲动力学模块的 SOTA 改进策略**

为了使 QSH-Net 的时间动力学模块具备真正的竞争力，必须实施以下三项关键架构升级。

第一，引入自适应替代梯度斜率调度（Adaptive Surrogate Gradient Slope Scheduling）。最近的研究表明，在序列建模和强化学习任务中，维持固定的替代梯度斜率是极其低效的。较平缓的斜率可以在深层网络中增加梯度幅度，但代价是降低与真实梯度的对齐度 14。QSH-Net 必须集成动态斜率调度机制，在训练初期使用较陡峭的斜率以确保方向准确性，而在训练后期动态平缓斜率以促进深层参数的精细更新。这种方法已在复杂控制任务中被证明能够带来超过 200% 的性能提升 14。

第二，将基础 LIF 神经元升级为脉冲状态空间模型（Spiking State-Space Model, S-SSM）。状态空间模型（如 Mamba 和 S4）在过去两年内彻底颠覆了长序列建模领域。SSM 通过对关键状态转移矩阵施加结构化参数化，实现了兼具并行化训练与极速自回归推理的混合架构 7。QSH-Net 应当将连续时间的常微分方程（ODE）通过线性递推操作进行离散化，并利用 S-SSM 来替代传统的循环脉冲层。这不仅赋予了模型处理不规则时间间隔 ![][image8] 的原生能力，还极大地增强了网络捕获多尺度时间动态的表现力 7。

第三，采用格雷码相对位置编码（Gray Code Relative Positional Encoding）。为了在极长的 IMTS 预测中保留神经元的纯脉冲（二值）性质，必须放弃连续位置嵌入。格雷码 RPE 通过将相对时间距离映射为二值格雷码，直接向脉冲神经元注入位置信息。对于序列长度为 ![][image9] 的数据集，必须动态调整编码位数 ![][image10]，严格确保 ![][image11]，从而根据鸽巢原理彻底避免长序列中的位置信息重叠与冲突 16。

## **超图拓扑（HGNN）的因果性控制与稀疏性表征**

超图神经网络（HGNN）通过允许一条超边连接任意数量的节点，从根本上超越了传统图神经网络（GNN）只能捕捉成对交互的限制 18。在 QSH-Net 中，超图被寄予厚望，用于在不需要任何时间对齐或插值的情况下，直接建模跨时间、跨变量的高阶动态关系 1。

### **KNN 噪声污染与无向传播的物理学违背**

现有超图模型在时间序列领域的应用普遍存在两个致命缺陷。首先，许多框架依赖 K 近邻（KNN）算法基于特征相似度来静态构建超图。在高度稀疏且充满噪声的临床数据（如 MIMIC-III）中，这种方法极易将由于缺失模式偶然重合的毫不相关的变量错误地连接在同一超边内，导致严重的噪声传播和语义污染 18。

其次，绝大多数 HGNN 采用固定的拉普拉斯矩阵进行特征扩散，这意味着超图中的信息传递是无向的 20。在时间序列预测中，因果关系具有严格的单向性（时间之箭：过去决定未来）。使用无向超图进行信息聚合，等同于让未来的未观测状态反向影响历史表征，这在逻辑上是站不住脚的，也是顶会审稿人极易攻击的理论漏洞 20。

### **超图模块的 SOTA 改进策略**

为了确保超图拓扑在 IMTS 任务中的理论严密性与实证有效性，QSH-Net 需要采取以下架构革新。

集成基于自注意力的超图边剪枝机制（HGNN-AS）。为了对抗 KNN 构建带来的噪声超边，模型必须在超图拓扑之上叠加多头自注意力机制（Multi-head Self-Attention）。该机制通过评估超边内各个节点的语义一致性，动态分配注意力权重。对于被错误连接的无关节点，模型能够自动赋予极低的权重，从而实现拓扑结构的软剪枝（Soft Pruning），极大地稳定了训练效果并隔离了稀疏片段带来的噪声干扰 18。

部署有向高阶神经网络（Directed HONN）框架。QSH-Net 的超图拉普拉斯矩阵必须进行有向化重构。引入一种灵活的谱拉普拉斯公式，通过一个可调谐的 ![][image12] 参数来控制有向扩散 20。这一机制不仅能够平衡局部节点身份的保留与全局特征的扩散，更重要的是，它从数学底层强制保障了特征只能沿着时间轴向未来演进，严格维护了时间序列的因果推断完整性 20。

实施无填充的时间自适应节点表征。遵循当前最新的 HyperIMTS 和 Hi-Patch 设计哲学，QSH-Net 必须将每一个孤立的、未对齐的有效观测值直接初始化为一个独立的超图节点 1。每个节点内嵌其特有的连续时间编码 ![][image13] 和变量标识编码 ![][image14]。超边不再固定，而是随着数据信息密度的波动，在特定时间尺度上动态收缩与扩张，彻底消除传统图模型对数据对齐和零填充（Zero-padding）的依赖 1。

## **IMTS 领域特定创新：联合时频建模与自适应分块机制**

在完善了底层数学架构后，QSH-Net 还需要针对时间序列的领域特性进行专项优化。仅在时间域内操作，模型极易陷入局部最优，难以捕捉环境或生理数据中存在的全局周期性与季节性趋势 1。

### **克服信息密度不平衡：自适应查询分块（Query-Based Adaptive Patching）**

最新的时间序列模型（如 PatchTST）证明了将序列分割为 Patch 能够有效提取局部语义并降低计算复杂度。然而，由于 IMTS 的极度不规则性，固定长度的分块会导致某些 Patch 包含密集的高频数据，而另一些则完全为空，引发严重的信息密度不平衡 1。

QSH-Net 应当借鉴 TFMixer 和 TAPA 模型的策略，引入可变形分块与查询混合机制。模型不应切分固定的时间窗口，而应通过时间感知加权平均策略，学习动态可调的 Patch 边界，将不规则序列无损地转化为高质量的正则化表征 6。随后，引入固定的可学习查询令牌（Query Tokens），让这些令牌主动且选择性地关注那些信息密集的 Patch。这一查询机制能够有效过滤掉由于数据稀疏带来的空白噪声，最终输出高度紧凑且语义丰富的局部时域表征（![][image15]） 1。

### **全局频域建模：可学习非均匀离散傅里叶变换（Learnable NUDFT）**

为了在不进行数据插值的前提下捕获全局频率特征，QSH-Net 不能使用经典的离散傅里叶变换（DFT）或快速傅里叶变换（FFT），因为这些算子严格依赖于等距采样假设 1。模型必须整合一个全局频率模块，部署可学习的非均匀离散傅里叶变换（Learnable NUDFT）。

Learnable NUDFT 直接作用于不规则的时间戳上，提取真实的频谱系数。同时，为了抵消稀疏数据可能造成的频谱偏差，必须在计算中引入一个基于有效观测数量的归一化因子（![][image16]） 1。提取出的原始复数频谱随后通过多层感知机（MLP）进行相位与幅度的非线性精炼，投影为全局频域表示（![][image17]） 1。在预测阶段，模型将局部时域特征与全局频域特征进行深度融合，并通过逆向 NUDFT 提供显式的季节性外推偏差，从而大幅提升长期预测的准确率 1。

## **训练范式的代际升级：自监督预训练与分解损失函数**

复杂架构的另一个痛点是极易在有限的监督数据上产生过拟合。在医疗和工业物联网等真实场景中，高质量标注的时间序列数据极其昂贵且稀缺。

### **PrimeNet 框架：时间敏感的对比学习预训练**

为了提升 QSH-Net 的泛化能力并降低对庞大标记数据集的依赖，模型应引入类似于 PrimeNet 的自监督表征学习框架进行预训练 1。传统的对比学习方法通过切分具有相同数量观测值的“时间片”来生成正负样本，这会破坏 IMTS 固有的不规则密度特征 1。

QSH-Net 在预训练阶段应采用时间敏感的对比学习（Time-Sensitive Contrastive Learning）。样本的三元组生成策略必须严格遵循原始数据在时间轴上的真实采样密度。同时，在进行掩码重建任务时，必须采用恒定持续时间掩码（Constant Time Duration Masking）技术，即不论局部区域的采样频率多高或多低，始终屏蔽固定时间跨度的数据。这使得模型在无监督阶段能够深刻理解数据中“观测缺失”本身所隐含的领域信号（例如，患者病情恶化导致监测频率升高的隐藏信息） 1。

### **超越 MSE：分解驱动的损失函数（DBLoss）**

在模型的最终微调与预测阶段，绝大多数研究仍盲目依赖于均方误差（MSE）或平均绝对误差（MAE）等基于距离的损失函数。理论分析与实证研究（如针对 Transformer 局限性的 ICL 理论分析）表明，即便模型在前向传播中有效提取了趋势与季节性，基于距离的损失函数仍无法在预测范围内有效约束这些结构特性，往往导致预测结果向均值坍缩 5。

QSH-Net 必须采用 2025 年最新提出的基于分解的损失函数（Decomposition-Based Loss, DBLoss） 24。DBLoss 机制通过指数移动平均（Exponential Moving Averages）在预测范围（Forecasting Horizon）内动态地将目标时间序列分解为季节性分量和趋势分量 24。随后，损失函数分别计算各个独立分量的重构误差，并进行动态加权。通过将这种结构化的归纳偏置（Inductive Bias）强行注入优化景观中，QSH-Net 能够持续且稳定地超越仅仅优化全局距离误差的传统基线模型 24。

## **顶会评审防御策略：严谨的实证评估框架设计**

正如前文所述，无论理论模型多么精妙，如果未能提供无可挑剔的实证评估，顶会审稿人都会迅速将其拒之门外。审稿指南甚至明确指出，模型应当透明地承认自身的局限性，诚实的缺陷分析不仅不会导致拒稿，反而会建立学术信任 27。因此，QSH-Net 在提交论文时，其实验设计必须具备强烈的防御性和压倒性的说服力。

### **1\. 无可辩驳的基线分类对比（Baseline Taxonomy）**

QSH-Net 必须与至少五个不同家族的 SOTA 模型进行全面对抗。试图仅与表现较差的旧模型对比以获取虚高指标的做法，是顶会审稿人最反感的“基准追逐（Benchmark-chasing）”行为 3。

*表 1：QSH-Net 必须对比的 SOTA 基线模型分类学*

| 模型类别 | 代表性 SOTA 基线 | 纳入实证评估的核心理论依据 |
| :---- | :---- | :---- |
| **纯线性 / MLP 层** | RLinear, TiDE, DLinear | 应对审稿人对模型复杂度的核心质疑。如果拥有四元数与超图的 QSH-Net 无法击败极简的 RLinear，则说明其理论复杂性是不合理的增益 28。 |
| **常微分方程 (ODEs)** | Neural ODE, Latent ODE | ODE 模型是处理连续时间和不规则间隔的数学黄金标准。对比旨在证明脉冲 S-SSM 动力学在参数演化上的效率优势 1。 |
| **状态空间模型 (SSMs)** | S-Mamba, Samba | 提供长序列建模能力和自回归推理速度的直接参考系，验证脉冲状态空间融合的先进性 28。 |
| **时间序列 Transformer** | PatchTST, iTransformer | 测试模型在多尺度表征提取和跨变量交互方面是否优于当前最强的大型基础序列架构 28。 |
| **IMTS 专属与图网络** | TFMixer, Hi-Patch, Raindrop | 直接对抗为不规则多元数据设计的专有 SOTA 模型，证明自适应超图和 Learnable NUDFT 带来的决定性增量 1。 |

### **2\. 粒度极细的消融实验（Granular Ablation Studies）**

由于 QSH-Net 结合了四元数、脉冲网络和超图，审稿人会不可避免地质疑：“这三个组件真的是缺一不可的吗？”为防止这一质疑，必须设计严密的剥离测试：

1. **移除四元数空间 (w/o Quaternion)：** 将四元数层替换为维度匹配的实数值多层感知机（MLP），量化超复数空间在处理跨通道耦合时的实际贡献。  
2. **移除脉冲动力学 (w/o Spiking)：** 移除 S-SSM 和格雷码编码，回退至标准的连续激活 LSTM 或 GRU，验证事件驱动范式在处理稀疏 IMTS 数据时的优越性。  
3. **移除有向超图 (w/o Directed Hypergraph)：** 使用标准的无向图注意力网络（GAT）替换 Directed HONN，展示高阶拓扑和时间有向拉普拉斯算子在隔离噪声超边上的不可替代性。  
4. **替换损失函数 (w/o DBLoss)：** 退回使用标准 MSE 损失，清晰剥离出 DBLoss 对长期趋势预测准确率的具体提升幅度。

### **3\. 核心数据集目标与 SOTA 阈值**

要证明模型达到顶会级别的 SOTA，必须在公认的高难度、高代表性数据集上取得可量化的突破。以下三大数据集是评估 IMTS 模型的试金石：

* **MIMIC-III (Medical Information Mart for Intensive Care)：** 作为临床重症监护的旗舰数据集，其包含了极度不规则的生命体征观测、不频繁的实验室化验以及异步的医疗干预记录（采样频率低至每小时一次甚至更长） 31。其混合颗粒度和多模态缺失率极高。  
* **PhysioNet-2012：** 同样专注于重症监护场景，被广泛用于预测患者死亡率和临床恶化指标。模型在此数据集上的表现将直接反映其在极端缺失环境下的表征稳健性 31。  
* **USHCN (U.S. Historical Climatology Network)：** 气候变化监测网络数据，具有强烈的季节性和长期趋势演变特征，是测试时频联合模块（TFMixer 架构）与 DBLoss 预测有效性的最佳靶场 33。

*表 2：目标数据集的历史 SOTA 基准与 QSH-Net 期望阈值 (MSE 指标参考)*

| 核心评估数据集 | SOTA 参考一 (f-CRU) | SOTA 参考二 (mTAND) | SOTA 参考三 (GRU-D) | QSH-Net 需达到的突破阈值 |
| :---- | :---- | :---- | :---- | :---- |
| **USHCN** (![][image18]) | ![][image19] | ![][image20] | ![][image21] | 稳定低于 ![][image22] 34 |
| **MIMIC-III (Small)** (![][image18]) | ![][image23] | ![][image24] | ![][image25] | 稳定低于 ![][image26] 34 |
| **MIMIC-III (Large)** (![][image18]) | ![][image27] | ![][image28] | ![][image29] | 稳定低于 ![][image30] 34 |

### **4\. 学术诚信与局限性声明**

迎合审稿人的最佳策略是主动展示严谨的学术防御姿态。报告结尾或论文讨论部分必须包含详尽的局限性分析（Limitations）。QSH-Net 应当透明地指出：尽管采用克罗内克积优化了计算，超复数网络的显存消耗依然显著高于普通线性模型；虽然 SNN 理论上具备极低能耗，但由于底层硬件的适配鸿沟（Hardware Mismatch），在目前基于标准 GPU 架构的训练中，其计算时钟往往不及高度并行的 Transformer 矩阵乘法快 11。同时，针对超图节点与超边阈值构建的高度超参数敏感性（Hyperparameter Sensitivity），应当提供充分的敏感度分析图表。

综合而言，QSH-Net 在宏观构想上极具颠覆潜力，它从数学维度直击了时间序列建模的三大痛点：特征耦合、事件动力学与高阶关系。但要将其转化为一篇能够傲视机器学习顶级会议的杰出工作，关键在于能否通过上述的克罗内克分解、自适应替代梯度、S-SSM 动力学、有向超图控制、时频联合与 DBLoss 等一系列微观、深度的架构工程，将其打磨为兼具理论深度与实证无敌的 SOTA 范式。遵循本指南重构的 QSH-Net，将完全具备跨越评审严冬、引领下一代时间序列基础模型演进的卓越实力。

#### **Works cited**

1. PrimeNet：不规则多变量时间序列预训练模型  
2. WCCI 2026 \- Special Session: Complex- and Hypercomplex-valued Neural Networks, accessed April 7, 2026, [https://www.ime.unicamp.br/\~valle/CFPs/wcci2026/](https://www.ime.unicamp.br/~valle/CFPs/wcci2026/)  
3. \[D\] ICML 2026 Review Discussion : r/MachineLearning \- Reddit, accessed April 7, 2026, [https://www.reddit.com/r/MachineLearning/comments/1s1yz2t/d\_icml\_2026\_review\_discussion/](https://www.reddit.com/r/MachineLearning/comments/1s1yz2t/d_icml_2026_review_discussion/)  
4. \[D\] Transformers for time series forecasting : r/MachineLearning \- Reddit, accessed April 7, 2026, [https://www.reddit.com/r/MachineLearning/comments/18ax51t/d\_transformers\_for\_time\_series\_forecasting/](https://www.reddit.com/r/MachineLearning/comments/18ax51t/d_transformers_for_time_series_forecasting/)  
5. Why Do Transformers Fail to Forecast Time Series In-Context? \- OpenReview, accessed April 7, 2026, [https://openreview.net/forum?id=eBCk0nXz17](https://openreview.net/forum?id=eBCk0nXz17)  
6. Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline \- arXiv, accessed April 7, 2026, [https://arxiv.org/html/2505.11250v1](https://arxiv.org/html/2505.11250v1)  
7. ASGMamba: Adaptive Spectral Gating Mamba for Multivariate Time Series Forecasting \- arXiv, accessed April 7, 2026, [https://arxiv.org/html/2602.01668v1](https://arxiv.org/html/2602.01668v1)  
8. Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline \- arXiv, accessed April 7, 2026, [https://arxiv.org/pdf/2505.11250](https://arxiv.org/pdf/2505.11250)  
9. Hypercomplex neural networks: Exploring quaternion, octonion, and beyond in deep learning \- PMC, accessed April 7, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12513225/)  
10. Advances in Quaternion-Valued Neural Networks \- AFIT Scholar, accessed April 7, 2026, [https://scholar.afit.edu/cgi/viewcontent.cgi?article=8666\&context=etd](https://scholar.afit.edu/cgi/viewcontent.cgi?article=8666&context=etd)  
11. Understanding Vector-Valued Neural Networks and Their Relationship with Real and Hypercomplex-Valued Neural Networks \- arXiv, accessed April 7, 2026, [https://arxiv.org/pdf/2309.07716](https://arxiv.org/pdf/2309.07716)  
12. Decide When Ready: Stepwise Incremental Inference with Early-Exit in Spiking Neural Networks | OpenReview, accessed April 7, 2026, [https://openreview.net/forum?id=LUnYc9Grm8](https://openreview.net/forum?id=LUnYc9Grm8)  
13. Learning Neuron Dynamics within Deep Spiking Neural Networks \- OpenReview, accessed April 7, 2026, [https://openreview.net/forum?id=9W38INOZ00](https://openreview.net/forum?id=9W38INOZ00)  
14. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks \- arXiv, accessed April 7, 2026, [https://arxiv.org/pdf/2510.24461](https://arxiv.org/pdf/2510.24461)  
15. Adaptive Surrogate Gradients for Sequential Reinforcement Learning in Spiking Neural Networks | OpenReview, accessed April 7, 2026, [https://openreview.net/forum?id=oGmROC4e4W](https://openreview.net/forum?id=oGmROC4e4W)  
16. Toward Relative Positional Encoding in Spiking Transformers \- arXiv, accessed April 7, 2026, [https://arxiv.org/pdf/2501.16745](https://arxiv.org/pdf/2501.16745)  
17. CMDMamba: dual-layer Mamba architecture with dual convolutional feed-forward networks for efficient financial time series forecasting \- Frontiers, accessed April 7, 2026, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1599799/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1599799/full)  
18. HGNN-AS: Enhancing Hypergraph Neural Network for Node Classification Accuracy with Attention and Self-Attention \- MDPI, accessed April 7, 2026, [https://www.mdpi.com/2079-9292/14/21/4282](https://www.mdpi.com/2079-9292/14/21/4282)  
19. Hypergraph-based Techniques To Map Spiking Neural Networks on Neuromorphic HW (Politecnico di Milano) \- Semiconductor Engineering, accessed April 7, 2026, [https://semiengineering.com/hypergraph-based-techniques-to-map-spiking-neural-networks-on-neuromorphic-hw-politecnico-di-milano/](https://semiengineering.com/hypergraph-based-techniques-to-map-spiking-neural-networks-on-neuromorphic-hw-politecnico-di-milano/)  
20. Appl. Sci., Volume 15, Issue 20 (October-2 2025\) – 439 articles \- MDPI, accessed April 7, 2026, [https://www.mdpi.com/2076-3417/15/20](https://www.mdpi.com/2076-3417/15/20)  
21. ICML Poster Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach, accessed April 7, 2026, [https://icml.cc/virtual/2024/poster/33940](https://icml.cc/virtual/2024/poster/33940)  
22. Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline \- arXiv, accessed April 7, 2026, [https://arxiv.org/html/2505.11250v4](https://arxiv.org/html/2505.11250v4)  
23. A Joint Modeling Framework for Irregular Multivariate Time Series Forecasting \- arXiv, accessed April 7, 2026, [https://arxiv.org/html/2602.00582v1](https://arxiv.org/html/2602.00582v1)  
24. DBLoss: Decomposition-based Loss Function for Time Series Forecasting \- arXiv, accessed April 7, 2026, [https://arxiv.org/pdf/2510.23672](https://arxiv.org/pdf/2510.23672)  
25. DBLoss: Decomposition-based Loss Function for Time Series Forecasting \- OpenReview, accessed April 7, 2026, [https://openreview.net/pdf?id=SbhBIkiRLT](https://openreview.net/pdf?id=SbhBIkiRLT)  
26. NeurIPS Poster DBLoss: Decomposition-based Loss Function for Time Series Forecasting, accessed April 7, 2026, [https://neurips.cc/virtual/2025/poster/117918](https://neurips.cc/virtual/2025/poster/117918)  
27. Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators \- NIPS papers, accessed April 7, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/2f55a8b7b1c2c6312eb86557bb9a2bd5-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/2f55a8b7b1c2c6312eb86557bb9a2bd5-Paper-Conference.pdf)  
28. MODE: Efficient Time Series Prediction with Mamba Enhanced by Low-Rank Neural ODEs, accessed April 7, 2026, [https://arxiv.org/html/2601.00920v1](https://arxiv.org/html/2601.00920v1)  
29. SEDformer: Event-Synchronous Spiking Transformers for Irregular Telemetry Time Series Forecasting \- arXiv, accessed April 7, 2026, [https://arxiv.org/html/2602.02230v1](https://arxiv.org/html/2602.02230v1)  
30. ICLR 2025 Papers, accessed April 7, 2026, [https://iclr.cc/virtual/2025/papers.html](https://iclr.cc/virtual/2025/papers.html)  
31. MIMIC-III Clinical Database v1.4 \- PhysioNet, accessed April 7, 2026, [https://physionet.org/content/mimiciii/](https://physionet.org/content/mimiciii/)  
32. Mortality prediction performance on MIMIC-III for different percentages of labeled data averaged over 10 runs. \- ResearchGate, accessed April 7, 2026, [https://www.researchgate.net/figure/Mortality-prediction-performance-on-MIMIC-III-for-different-percentages-of-labeled-data\_fig4\_353634807](https://www.researchgate.net/figure/Mortality-prediction-performance-on-MIMIC-III-for-different-percentages-of-labeled-data_fig4_353634807)  
33. Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting \- arXiv, accessed April 7, 2026, [https://arxiv.org/html/2602.21498v1](https://arxiv.org/html/2602.21498v1)  
34. Still Competitive: Revisiting Recurrent Models for Irregular Time Series Prediction \- arXiv.org, accessed April 7, 2026, [https://arxiv.org/html/2510.16161v2](https://arxiv.org/html/2510.16161v2)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAAA70lEQVR4Xu2TvwtBURzFv8kq8QdYlPwZ/gAmilmZlJQkgx+bzY4MDAxYLSYLNsUfwH+g7OK83rm9L716xSROfTrf3ul0b+/eK/KbKoA9uIMVOIELZz/YMdvy2wGsQcwqGy3oUxDVAdQBKc41HRiN6DMQ1gFUARnOTR0Y6XJIB1AZZDm7lgd0t3JRPMo9+lvlPn0u9t99xZTb9CfplZMgoeiCHHPXlT/a9pBulYM6ELuc5tzQgZE+qoAOoJJ4nPOY7raydc6m3NKB0YRuXc+IDqCqONezroO8PD+MM7iCJfCBDbiJ8zCOYj+MuPz1LXoABtQ5+ABFwV0AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAK0AAAAZCAYAAABD9K4gAAAEk0lEQVR4Xu2aWahcRRCGK8YVN8QNdxSXiIqoeYhRJLiLS8A8hBDMg0oMRn0QlSwGMS8qLqCiuOCLG7iDCypRk6gRd1AhKgkJGImgIagQRUW0Pqr7Tk3fuXPP9Ln3cJL0Bz/3dHXPnDM11dVdPVekUCgUCoVCoVAoFIy9VCemxpayi+rk1FhoB5eqdk+N48Rm1X+qaYl9NA5QnZ0ax5nPxJ51UJrw52OqH8Seb5+kb7tgher41NgyzlM9nRpbygppxp/XSN6k2iZYKc04uQ7ny9YTtE358wXZjoP2fdWk1NgyLlA9lRpbShP+nCi21dqSdvRiJ9W1qo9UH6qOUV0tzTv0MtU7YvunPVUPie3BLvSDKvKB5GeGB1Rfit37PrH7r1J9orrLjTtN9bLqQdWuzl4V3jc30/Ld8Dzc/yhnn6Za59qRr1XfqB5V7Zb0VaGOP+F21SbVX2L+onD9TronwlSxLPtSaF+sek61bGhEYGfVv9L9wQleXsyLmuIG1ZJwzb2vDNf3qF4P14OQ6+SFSZtJfKrqULHniksXy9gZ4RrbueF6EHKDdk74e5XYve90fRtUv7o2E4vJBzPFxj/R6a5Mrj8jFJ3++hXVDs4G+JrnI6AXBxtxgI04HWKB6nlvkE41zFFOU9ymOlosC3D/HYP9WNXecVAP9pNOMFVVP2a7axzll6rrVaeE65vFljN8tEY1IQ4agfQZRlM/bg1/l4uNjQnnhNB+LbThHNXkcE11Tn9MCL3An3/L8OfpJ1blqhwplvHTbL+v2Hvx3T/s7Pg3BvAQP6lmJTZe/F5iawqyD0teXepmBqC6xxf9mCc9nFqR3EwLbOF4tnedjUmF7RZn86wX6z8k7ajAWPjzLNXP0ln+PXEVQBtV07u7O5DZGORTN2BLl8mmuFc1PzVmMBZOZj87WtCyrzw4NVakTtAuEnu2G53txWCb4mwRfEHfV2lHRer68wqx/ewjMnxbAHEVYOtzZri+rmtEIC4nKdjYC/WDgiTOjCoio4+2hMLnMjZVaq6Tj5BOUUXx4v2Dfalrs+zGfrYNfDGDUCdoKU64d9yuEAi/SPfWyhOzMEkhh1x/Rrj3Ha5NouT0JLJWbMz+oc01sQDD4oZKLmaKy1W/SfOnBhHu74OkDjlOPkzs/nuI+eBtsUAATjbeFPsZNMIyx3j2dBSLvTJIP+oELRmLe8df1L4P7ZH2qwQzmY7PlkOOP4HVnMKQZ/tTLNN/K3ZKRE0Qod9vs2g/I7b35Yy4C467iPLVqjdUr8rIH3y8oQjCsWNBrpPnis3wi8RmOIUqwXq3WLHgIdP+KHZE1lXdVqRO0PIz55Oqf8QKspipDvKDHPS9lRoHINefTGYKq8NVz4o9BxMoLfLZx/oTrJvEYoEVZYaz9+RjsYp9ayfXyU1SJ2jTz/aH2OQaCYLF738HpbX+ZPZy3LEtwIaeHyjaDNnnktRYgQPFgvCk0GaZ/V113NAIO4vlRwa2PPST3XILRmidPzmcxglep3eNKLSNL8QO48mwjyd98KnYf0yxn7w/6SsUCoVCoVAoFAqFQqEa/wOQ9R+WCG7RcAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFsAAAAZCAYAAABeplL+AAADPUlEQVR4Xu2YS8hNURiGP3fllnvIpTAgJgYyIGUgE6VcolzCCAMpA6SMiDKQO+EvwhS5lSQD5H4nISNTihiY8D2ttf6z9mfvc/b+nVNkPfX2n/Xu9e6z9trfXmufXySRSCT+e0arNlkz0RqOqpZbM9EaXqs6WzPRfKZaI9E6Llkj0Rqmq+5Z81+hm2qd6pbqtrhdfr24Dagscf6mVM9Xge+Zbc2SDFLtVt1XnVL1V+0TN/4yxPlHUjHfXfVDNSrynqt+qmZEXhHkb0g2D2XzVWAzfKaaYg+U5KPqiaqLbzNxjPOruIJrxFjJ5qFKXraozhjvm+qL1E4wVLVdNbO9Rw3yfGFMT8nmZ6neqC6qpoVOHWCp6rQ1K8A410Rtrgfvim9vUJ1XvVW1hU4RVyWbhzgPk1XHVP0irx3u9kLjcQK+FIaIC+PNb+9Rg7ydbNbUkJ8kbjnpKu6x/awa6Y9VgWp6qRpvD5Rkorhxcj2Bzd7b6NvL/N/BqseSreC8PMR53pDOeY+nJsM4f2Cg8fHWRm0qG29B5EHI28nmIkJ+r+qy/zxGXN8dvl2F1arj1qzAKtVD41GRjIeCgJ3RsRWqOVGbvL1OiPOwyHvcsAyUfNEJmMhA0WSH/HvjX5Nafpe4agY2E/qf8O16hJtYVntcrBA2sDbjsdR9iNrXo8/zVCujNnm+J4blMs5D4WTDAdUwVSfVYtV3+f0tgiAnyFtGyH+SbN4OCtjcqCwezz7mWCO4KG74n8D4mEw29B7i9iDGGZaOmBHi9q0Y8k+lcZ6CxP9tGQE2sXeqV6oL4n4sLMn0qFW2XduB/H7J5vMme6u4V7bcO94ANq5mwAbNT3yq8bC4cQ7P9HBFwfJyxPhAUcZ5NlKbD5Vt1/ZcqD7bsd5kW8jnTfZdVW9rlqCXuItqNgNUL6wp7h36oDVzIJ93naUnO0yqJficqB6hHy/7MROkVtF9VYeiY41gs91mzSbA427Xef6DeFZq/9yyT3hMWC4sYbILlz3uJh1i8aoD3ME73mO5YfPLoyh/MufYXH+sEUwyk91MeMLisfDqynoMdpwUhqVent8TLDP4D6R4rv5KeEI6svQkEolEItFhfgH5LNhdXdpzAgAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAZCAYAAADe1WXtAAABFklEQVR4XmNgGAVDCqwE4v9oGATkgPgrmvg7IJaEyh9Dk8uBisPBOahEDboEEPgwQOTY0SWAoAWIS9EFYWAnA0RjP7oEA8InUmji6kB8GIiZ0cThYBkDROMiNHFfIE6Fyumgye0BYjM0MRQwhQGicSuSGAcQTwZibqicE5IcCIDk8IJcBojGm1A+KJLKENIMD4A4E4m/GYmNE0QxQAx9A+WDIgA5Ys4AcTWUHQTE25DkcAJ3BkTSYAFiR1RpcET2MUAi5TwQK6JKYwcmDAhDxdDkQGA5EM8D4m4grkSTwwlANsMMrUWTA4FpQHyLARIMbGhyeMFPIL7PAIl1dFDOAHEpyeASEIeiC0JBHBDzoQuOglEwVAEAckQ9/dfHWuAAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAAt0lEQVR4XmNgGAUDBtyB+D8W7ACVRxcHYYKgC4gfATErktgRIK4GYiYkMYJAgQFiYyiUzwjEzXBZEgHIoN1QdhMDxDCyACwcJgPxEjQ5ksA8BohBv4BYDE2OJHCaAWIIyDBQIJMFEoFYAIgXMkAMAsUgySANiG9C2aYMJKQZZBAGxI+BWBZJ7CADkQZxAHEtEH9mgEQzMgBFuRUDxCBQ4ONMjF4MmMm+DEl+GRb5o0jyo2AUUA0AAMK5Mfi/uC9bAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD4AAAAZCAYAAABpaJ3KAAACl0lEQVR4Xu2WS6hNURjH/8orSnnmTRIGJkxkdssr5VEIRV0DkcfgDihCKHmkEJIyoEyYYOSZgbzyCHkkiYTCRKQYSPz/51u7s/Z39z5nbwPqtn/1r7O+/1nnft9d61trARUVFRUdmNPUbycxkPrm4l+pUcG/4LzNIf4vOYf2uX+KfO8ti7wa92DGdm+QKTCvtzfIOmqHDzZgLnUDNqeFWki1wf6J4+pfK0136i21z8XfUHNcLMV5WHGHvUGOw7zRLj6Uuk11c/E8DsFWqIc3Ai+pxT5YAu24L1SvMB5CTavb2ZyEFXfKxTVxRfAmOe8MbNWKcpN6R43xRuA19Ywa4Y2CqFDluZrqSV1L29kcgE26FMW6UkeozsGbFXlamRPRuBkqRq2yAFbgyJRr8WHUPOqi88qgPB9RV6lW52WyBjbpVRgPojbV7Zq3PhprF5Thshur0P7UZOo50m20NOhv+A7L9Yo38lgEm6BTXGyDHRgJ8naGz9OpTpFXhA8+QO5QT9B+68+k9rtYEbpQt2C5ShPSdjZTUZ+ggmek7Vr8KKxgJVyWjz4Au3beI7twf9Y0Qz2tNtXB+RmW77HUN3KYiHrhg50nFNd9v5Xa7bwiXHfj+bC/uQp2DY2NvLXIvlbzULHq6YNhvAvp3duQ4agXnvUQUVz9/xj511Ej9kSfZ8NWJWEDrPgEXXkt0bgR2p3qZ91KSfvpmv0Jy7kQP2AJxL2doB8564MluQvb2lrpLHTVPYVt9WZoV6o9f8EKjVH+W2A5tzovk4fUEh8M3Kf6+WBJHsAeSnm/o7Mj6wHlUdskuzNR3+DpMaUF9P7e4P8X9PbXAamV3Qi71/tQ46mVIdahGQA7JPW8fAHrz+Wpb1RUVFRUVBTmD2QynbFLsPRmAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAaCAYAAABctMd+AAABQUlEQVR4Xu2TvS5FQRSFl59I/EWBQuUBNAoFnXgE4Q383EQhohASL6BREFGotEQlUVKoPYBKFBINCSFRsXb2zDVn3TmuyO2cL/mSO2vPmZkzdx+g4g/U6KCGreKTHmjYCvrgi3/QUamVMUAXNcyxTl/hG+xLrYxD+Pwf6aIPdAc++Z0OF2bkuadvGqbYwnd0Mox34Ru80KGQ5ViCz9vQQsoKPU/GI/CT24P2Jorl6mVhRkBPHdnD9+lzdNCnoP3OYn19oSG8W6xryv6sCXjtTAuRTvippySPHMEXyN37Gry2qoXILZ3RUDiGv3p/krXR52B7khe40iDDGPyEW0k2HrKTJLPOqWO7b9PpX2gLPdIeOAshi1fSDbnaOTS2UzPtCzZmw3iZ9tLTkNe5RuPDzbQv2O7Y3nqT3sC7ZR4VFf+IL4t3W+vhLtOAAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA0AAAAaCAYAAABsONZfAAAAoUlEQVR4XmNgGAVYQSEQ/8eBxZDUYQV5DBCFlkAsjCaHEyxmgGgiGmgyIJxENMhmgGg4iy6BD2xkgGjqQpfABZiB+BMDRJMbmhxOYM8A0fATiNnR5HCCQwwQTR3oElDQgy4gBMS/GSCafNDkQABksxa6YDADRMM3IOZBk2MB4jloYmAwlQGiaTeUzwbEVgwQwy5C5eBAGSpACINcMAqGEAAANNYsAcrGdYYAAAAASUVORK5CYII=>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAaCAYAAACKER0bAAAAn0lEQVR4XmNgGEJACYg/A/F/IF6FJgcGjEC8hgGioABNDg6OMkAU6KBLwABIciq6IDIAKagH4hNAfBKIw1GlIQr2AbEIEj8MJqkAxFeAmB8mwABRAHIXGKQCcSNCDgxACp6CGExA/AGImVGkIQq6QQwOKAcZgMRAAacGEziHkAODNCDOQxaoAmJ1KDsYiN8hycEByB0HgHgFENuhSo0CAKJgHz2x7UuOAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFkAAAAZCAYAAABaU4LDAAACWklEQVR4Xu2XTYhNYRzG/yhELCTJ5/hWyrdSk5RCEhtRdsiCjTQ1CysbpYiFsrAQWRCS8h1FYuNzyVYp+VhYKELTeJ7+55p3nnPmzjlnrnPcen/1a859/ud27zz33Pc91ywSaTO+w7caRlrLV7hHw3ZhGBylYQ2cgR/ge3hPZqQXztWwHVgAL8NpOqiYTngUboXH4G+4pd8ZZl/gIfP5DZn9hZc6P40sq4av2QOfJX//RckX4S0NM+A36ZVku+BnOCnInlvfN+5SkGdyEt7UsEa4oUzXsAQHzDcm/m8bZNaMLvMPfaTkzJ4kx2vg5GD2LjjO5CXs1rBGhlryTPML5wJcJLM8HDEvdLnkzJ4mx9uCfGkyG5AJ5ies0EGNlC15Jbxqvn6y4LKwtG9whOTs6VxyPD/Ib8PzweMUO8yfzHXof6FMyQvhT3gazpNZK1hn3tPqIDtlvjkehsODPAVvUZpe6jXAkmdo2IQxcJ/5LVd4hbWKjZa9RudisdV3N9EMlsx1tSgs4775krFEZmWZAj/CxzrIy0Hzgt/oIIGL/FQNE9bDBwUsAkvu0LAAa+FDuEwHBRkLX5j3MF5mueGCzZLP6sB8Zy1aTqtgybM0LAFv3a5Z+bKvm38zWDaZCPf2jQdnNPxhXnLW7++7cKeGFcGSZ2tYEhbM5YOFFeWO9f95vxleCR4PynHzgl8nj7nRcOfcBB+ZbyJ1MAf+gvvhOJkNFX5w2zXMYJX5bWBjvwrdHZw3IB2WfmKWfKEq+WTp99CQ9/JVcsLS76FhZ3BeJBKJRCKRSCQSyckf7k2Ob7tiHVUAAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAaCAYAAABl03YlAAAAq0lEQVR4XmNgGAU0ARxAXAbER4H4FBBvBeJWIK5GVnQMiPcDsRiUD9LwH4hDYAo4gfgzEIvCBICAlQGiSBgm0ADEM2AcKFAA4nMwjjoDRAc7TAAKQJpcYRx9BogidHCbAUkjNxD/QsiBQSQDFo2zGCAOZALiFCD+y4BFEcjY+0B8HYhXAvFBBiyKkIEuA0TBWnQJGJAC4gkMEEWlaHJgAPIySBIZ86CoGGoAAFenIZiS4PU5AAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACkAAAAZCAYAAACsGgdbAAACiUlEQVR4Xu2WS+hNURTGl/cjISYMlIS8y0ASoUiRlDBhoCSkRPJMhCh5lJFHSR5FSQp5FAMlEyRGXgOSVzEQpWTA9/3XPt31/+4553/uTRj8f/V19/nW3ufcs/faax+zdv4P+kDPoYkaaIIdUA81q7AMWqdmogN0HdqugSbpDF1QswqXoElqJnZCb81v/qf4BS1Ssww+/EP6zeMntEnNQHeor5ptcA16CnXUQBFboV1qBi6qITyAhqhZgffQbTWLuApNVzMxDNqiZqC/+dI1A/Pyh/lKFDIeOgZ9hW5CJ6GBrXqYLYZmihdZaM3/yQ3mY6dpgPCPnIUeQiuhy8mfDD3LOiUOWf1SjjO/uepT7FSBeebjVmigF/QI6ha8+CZzoLXh+oZ5CcpjlPlD7mmgIsPNxx/QAPNriXg9Q3s+dDhc3wltZZX5Q/ZqIMDZeqVmop/5+BMauG+e7BkjQpucMi/sGczVIs6YP6QsZ1l7d6uZGGA+/rgGvsn18tBmrfsOjQneOWs905E35rszpk4jjLSC5X4t16dDm5uIgyJHrH7Hk6HmfW+la/bZUwtXgrPMe2zTAPPgMTTLfMbumpchdt4Hdap1bWECtEY8Mtt8DA+CrlarEBlZnp6HjsZA4KD5PfjCdWyGXpp34HJthAbHDgEeW6yfeaw3f2Ge+10klhVoniqrYyDAVeCXVSlMWBbrtnhn9X+iCmPNJ2K0BhI8RMq+CVp4YsUzGOGDWJoahTPIzZXHXPNVZPoVwqLNU6cKzJkv0BQNlMAvKo5ZqoHER2iQmgrzsO44KmEB9MKqf1FnOzevMtBr5IUbghtuv5rCDOiz+SpdkVgGN9o/Zar5juYh0Fti7fw1fgOGP3aM5wQ1DQAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABsAAAAaCAYAAABGiCfwAAABcUlEQVR4Xu2UPyjFURTHvyQLSlEog00Uk0XKICVlk7Ipkz+7DDJYJAaDGJTCpgySzG9hlVKK8hZJBlkUC9/TOa/fdXh473ff9j716Z17zu/3O/f3fvdeoEyZAlikH/903e5JRS3dgD6w0nJVtJn20xWrzVktNafQB+bjnI76ZDEMQBs9u3wmiHdpfTAumiVos70gt0CfgnE05A38YhAPwotiIIvjHfrwRtoG/Vtf6WxyWRyGoY0uXT5L2y3ugk4gNRfQZpO+YHTQa1rhC79w4hNCK5Lv0+lqOY5R+P5a9glhAtrowRcC5Hs2WCwbfhPJSdJL7y3+kx1osyNfgJ4e+9Brcsimlgne2riF3iTln+nG92Wezz67Rxih1XQ1yG0F8Qz02JNJRmEcyfnZQ+ctHrJfefusxamRrSIrUxpu07qvZawh4ptJo0M6Dd1/HjmNpnyyFNTQN+iaKDmD9NEnY/NCm+gVHXO16NzRMxR+0sTlE7plVZmR+GWmAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAZCAYAAABD2GxlAAABmElEQVR4Xu2VPShGYRTHD4VIyGLwWT4iE4My+khJSQqLMiFlsShlkVImFikmsRgkE4uPwWcxWGSgZJBJyWQQ/9M513s6qLdeL3e4v/p1z/N/3rfOfe59nksUERERNy/wXT13c6FhjaTBFj8RBlLgI3yG6W4uFDSQrN6mnwgLEyQNjsIpeAJ34YD90X/yCt9gv8mWKUTvJDey5LJTzStd/h33MN+Hvwk30mPGhZpdmSxg2Ackr0hS4cebZ8ZjJA1Omizg0gfJpggeueyMpMFiHWfqtVTzP2UFNruMm5iGufDY5Ktw3Yz5vxlwXMcjsJ3kPOWji2mEc1oz+6Yuo9gTKSFZqNTPWWWDvh7O3MgTvICDJufNwE0w2bAV1sAKzbphGpzVcZDxjQbcmXoHHsAFkk0az4b8kSqSla11+Ywb8262R5M9Heop9orkaN0Xm06MIfigdade+RN5C6thuWZdMEtr5sbU8/AQdpCsKjfYZuYToglek+zwgAKSx15nsm1TcyN7ZrwIt0hujOEb4w8Cnxa9Jo+IiIiHD/ISSvG0VwzgAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABLElEQVR4XmNgGAXDCgQB8X8CeAlcNQlgMQNE8wUg5kOTA4lfAmIhNHGCgBWIPwHxLCBmRpNrBOJbQCyNJk4UmAnESWhiZQwQl75EEycJ6AIxJxI/lQFiKMgXZkjiFIPfQPwNiB3QxCkCRgwQg33RJSgB6kD8HIij0MRF0PhEA1Egvs4ACddsNDk2Bog8yUAAiM8xQAxtR5MDgSx0AWLBEQaIofPQJYBAB4jfIfENgHgREKsBcSUQlzNAcqMhkhowYGKAGLoeygZlFHkgTgDiqUD8FSoPA3OAWBCITwIxB1QMpA5kGQpoZcAsD9DxdrhqBoZwIJYCYmsksa1A3IfEJxvMQGKDMtILBsxigCxwA4m9ggESN5JIYmQBdiB+jMQHpZiDQByLJDYKhioAAFwIQD0/ChIrAAAAAElFTkSuQmCC>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACcAAAAZCAYAAACy0zfoAAABvklEQVR4Xu2WOyjFcRTHjzwHJo9IipCFTZEkyaA8Bhm8IotFbAbEqAzKoOQ1yKKUQR4ljAYGJhsWUiiPojDwPc65+t3jGq77173D/dSn////PcM9/9/r/omiRIkSkBf4oW6YWkSwRdJciS2EmyT4DM9tIRKoJRm1WVuIBMZJmuuCU/CYZJq56bDDjb3CBifb1zzDycICNzFhslt4D+NM7qONZBkcwXRT8xRuriZAtmIyl3c4BO9glql5Cu/URJNxc80mczkhGdVMW/CSArhtsnh4rVc+Zgb8y18s2OA/WIalJuuGPTAXnsJCvypRKv1cBq1wDp7BFHioOb887/58WAR3NI+Bq3qfBt/gmD5/sw5jTZYMr+ABrDM1ppqkQZcmkqke1ucWve7BXTgKZ2CO5r3wRu+zSZZRhT6HRJ8NFP6BMueZR4QzbtzCI7uk9+3wkX4/GYJi3gYK71yeLh98xHBzVU7m4wH26/00yaEfMryreX0Eot4GoJhkLQ7CRifnf59LuEnyApVOLWh4yHlxl8MLUwuFDvhEcjL8mUmS83CE5PD1ikXy4Buyk+RIWIMJpvZX8kjWHs8Ef3xECZlPCdNSONBfassAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAUCAYAAAAN+ioeAAAC9klEQVR4Xu2YS6hOURTHl+RREmLkdT8G3gbKK3lc8kgpEgaKIrcYESWGRJF3kVdKmCkxMiAl5JE8Uwy4N48wMVCSDPj/rX2cddbd3+Oa3ON0fvWvvdfe53zn/M9+rP2JlJSUlPwTi6FfRu+hrpkeWbZKtv9z09YL2gndg75CO6DB0DxoX+hzQbLX11KhGAA1Q1+C+IILbQfHC+gtdFb0uskhTpMfQNeh6VBf6Fro+w46GvqNEb1uu+hvbQp1aja0BjoOff/Tu4C8gU6Jvvx515YwAboMPYI2u7ZDoh+qn4l1gy6K3vOYiRN+DMYnunjCaqinDxYBGj0UuinxadtFdJngy8eM5jVXXSzhinTcaDLLB4pAYvQGiRvNaX0mlKsZ/QMa6eKkRRozehu0xdRXmnKMddBeHzTMFJ2luYJGD4H6ixrGsuUkNDeUY0Y/FTXuI7QWGpFtbkfM6MOSNXqQKcfgvnAX2u8bAq3QWB/sbGh0UyhfEh1dlpeSZiM02hpC5kM/JZs13ICW2k6GmNGvpf1968HNnBs0N1XLFGiqi+UCazTXYZowLm3+m12QmNEJ3Ay5/HBDtaavsJ0kNZrLSnPQN6l+33pURDOW8aKzakGmNUfQ6Iqp04RdocxRY4ktHTEWiY5S3uuha4uNaKaA1miWa+X0ng/QZ9GzQW6JGc2H7iE6Qi0xo5e5esJA0ZyY97PEjPabYZspN0Kb6DNXW65yAY0e5uo0glPwtomT2NLBdbIaTPsaMdrDw0+jVOQ/WjqGmzpPcMn6yizCEhvR7HfQxQjzb26kj128ntFNosf9ejBLYsaz0cW5GU5zsU6HD8uXPiKaSyfcF532PE4TnvQmiR6rT0i6eZLko/BozX4Je0Kc67WFOTLjq6DuJt4HWg69gpaYeAymd3egA74h0Co5Su/8n0rU7tDGA8HpUE5GoNez0M4ZMUr0APEEOgfdgj5JdnPqyJ9Ko8M11VgvtQ8sMyQ9ZBWGOabcO9S5iXLUlZSUFIrfd6HM5/qHZMIAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAADHklEQVR4Xu2YS8hNURTH/+QZIgPkkS8yMJIi8uomhJRHIQOSMuArlEdJyiMGHinEQEkpESIDEvlIDJTylsdIiSREKBLr/629P+use8899xrdk/Orf9393+uc29nr7LX3PkBBQUFBQzNOdEr0VvRK1CfZXRXG3hN9El0RjU52t3EWGvNMdMT1RZaILkHjdog6JrtrZoI38sJn0R5RP9EI0WtRkw1IYRB0YHkNE7JP9CMRoawRrRItgibkt2hgIgLYKPoiWibqKXopahF1skE1MtsbeaCX6IbzOCAcsCyuQd9ey21RV9Nmsr6aNjkOnT0WJme7ac8IHpNYL7lMxEnRfOcdgg7CAudbmqEx+53Pt/6AaZ8TfTdt0gF67Rbjse1L4qMUP4tcJuKxaKrzdkIH4KDzLYdR/haT6aIHps3ZxjgLZ4y9dkBot2+LUG4Gf6Hzs5jjjUanHbSm+8VtM3QAuHCmcREaw9puKYl+mfZylJemsdBrF4f2eFReW1i+Kv1HFrlLRDfog45xfkxEi/Mtt6Ax651fCn61Hc9e0TtR99CeCV2XPDER23xHBrlLRBfog/INtWwK/nXnW2LJ2eD8ScFP2+3sgvZbpqF81pDL0Fhf/rLIXSIIdz0+EXFGVNs5cRGulIiS6IPzInzzeQ23uZZRqJyIOCPq3TnlMhHcr/MttrAUcADSDl7kKDSGs8cyBXq28AyHJt0ngQwR/fQmtDTyP3jQq4dcJuK8aJbzOFgcgNXOt6yFxrDUWOaJzjiPPBdtNW2uS5PD77id7fG3u5W7wR/p/CxymYih0MXT8hA6UyI8BT8RPQ2/Iy9Ed0yb8AzR5Lxj0JJldR86eyIccLv/7xu8arMyjVwmgryBlocIB8A+DEsDPWqp8SciWVKGid6bNuGnjXit12ATdwF6Uo87KR4UWcr8p5BayG0iTkAXWD48F+iVye7Wk+3VoN6ub4XoNPTw9xH6+cLCGeMTQH1D8gDH70ucdSxh/O7Fe/nzjWc3yu9bTTypNzQ82PHDHQeVO6Z64cmX182F3utf6QzdOKwT9Xd9BQUF/wV/AOCCxJ1IRIxoAAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAACpElEQVR4Xu2YT4hNYRjGX4XIRhIKdSMLK6X8SehmKAvlTyEpscTCQpRkQ1n4kzLTWCgbJUKsWJAZwsJK/iZslFjITOxIPI/3fOY973zfuGdW9+ueXz11vue852nmfc+953xXpKampqatWQZdgT5DH6Bp5dMjwtqn0CB0B1pUPv0Pm3/RnSPvoOneHCXLvZEL36BT0AxoAfQRatiCBLOhN6LXcCBnoB+lCmWVlPM5sIYtAL//oyqs90YOTIbuO+87dN15Me6JfhIsj6GJZs38L2ZNpsrwfN94qwemrhWyHMRlaLPzekUbsMX5lr2iNWedvw3qNmvmx+5onz/fHAduQTu82QJZDuIltMZ5x0Ub1eN8yznRmmPOXws9M2vmpwZh88eZY7JLdBCjYYM32p0xot/p/uF2RLRRt51vYZNYc8j5TehXcRzyU4MYKf89NNObLZLdICaJNmSJ88Mg+pxveSRac8D5zcLnHR7yU4NI5Y+FDnqzAtkNYoJoQ5Y6/3Dh9zvfwgc8a3zDVhb+eBnKTw2i35sFD71RkewGQfjW4wcRPhH+zcZyQ+KDaEJfzZr5qUHE8vm3xOqrkOUguJHiXWw5KtqM8863XBCt4afHslp0bxFgfqyxqfwTEq+vQpaDuAmtcx43ZmzGPudb9ovWsHGWTdA1s2Z+rLGp/D6J11chy0HMhU4777nonRyYBb2CXhfHgbfQE7Mm3EM0zJr54S3KYvMDC0WH0JGDIJ+gOWbNRth/hpuq0KCdxl8B/TTreTJ8F034HLH53EDGmsX9TEcP4pLoA5a7ZD5A95RP//0d6W6hKe7cbuiq6OZsQOI/6BGbn2r0Yqk2iJMyVN+KXuhl7Qs3Xvwxjk3lG1NVtopet1E0K4bN51dQii5ouzdramo6hT8VI7fJiv9vAAAAAABJRU5ErkJggg==>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAAC00lEQVR4Xu2YS6hNURjHP+WZPBIy8LiRDAyYiIQOEUl55JGJboyQ7kCUMKEMPFKIgTIgrxAjFHmFYiTPxIg8kjyLIvH/32+v7rc/5+x99hnorOxf/Ttn/dd3Tvd+/7PXXmuLlJSUlDQ1E6GT0FvoBTQwPZ0Ja+9Bn6BL0Lj0dIpO0Bqot58Afd24C7QJWuj8epjkjVj4Au2EBkFjoFdQiy2owRDoqehnGMhu6EeqooNRomH/hga7OfIdOgO1QbOhR6LfNdUW1clcb8QAf4nXnfdVtCl5XBG9Eiy3oR5m3Eu0+XeS11pBhLmg16JXaiNEGcQJaJHz9os2Y7HzLatFa/Y4fym013mBX5IdRGdvNkiUQXAJmOG8baKN2ed8ywHRmq3OnwXdd17gXwUxzxvNDm+eXIf9zW2zaGMuON9yXrRmg/Mrog2vRl4Qo6GL0GfRoPulKuonuiB6ijZgvPNDEFedb7klWrPO+ZXE567HkxfES6gV6i+6Ybgp+jcWJboguos2YILzNyb+NedbeINnzXrnT0n8rs4nWUH4G/NM0VretIsuWdEFQbjr8UGEKyJr53RWqgdRgT44LxCC4LY3D55HWEuNdHN5RBnEc9FfsWWLaAMOOt9ySLSGV49luujZohpFgmDzQxDT3FweUQZxDprjPB7M2AAermqxVrRmu/MXQKedFwhBDPUT4AE01oz5PgTBA2MRogxiBLTLeWwKr5QA1/TH0JPkfeAZdNeMCc8QLc4LhCCG+QlwQ9IN526Mtdw9FSXKIMgbaLgZswH2n1mWeFSr8SdDP82Yy8l7M7bYpWaFmyPHoW5m/E006D7Gq5dogzgmeoPlKZk36FXp6fbnSJcT+b39SuiU6OHvI3QkPd3OO+kIwYqPPwKHRR868vWo6DllgJmvxg75+zuz9FA/1rzwYMdlgU3ljqkoS0Q/N1/0uxqFV+Vy0ccnJSUl/yV/ABSrtt0X7MOXAAAAAElFTkSuQmCC>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAUCAYAAAD2rd/BAAACAUlEQVR4Xu2VvUscURTFn5+glRYWWq1fhdFCsbEQooGAIAoBGwshgqSwiKggCAprI0ggRBCEGBDUPyNNKi2sghaJSppEhFQhpJAges/sue6ZwVlcknJ+cGDvb97e92b2vZ0QMjIyMh7LH8ut5aNlwnLBek8HleBZiPeYCaV7wCPvLeOWv6xf6SDSEgrXvriooTi2VNEN0SGTdKX4Fkr3UHro3olbpbu25MQ3WC4tR5Z6ly9CYfCGC6M6FO/6s/g0fGFpPZS3dKPihuiQLfHrdGPiogGQyyqN7/TJCR/Cx6X1aBaHBwA3KK6LDjmhw3d+0zXRRXylnFNpnNGXs+C0Hu5zrJE+OtAq3udbkhrbIk8fflC+dkH8RspZcFoPf/JPWCO9PijEb8Tn862DrNFFYPUPTXZO/y8L9h6LrLtZJxfcJt7neyM1vnePP4V5laF48stZcFqPadY51kg/HegU7/PhJr2upYvYpsyrNG7oy1lwPuG9R4e4Uzr8dzsDdAi2KMiJi4G/DMhNcXV0CF4ADg4HTvmCOOBj03oo/lPjBeWM0CE74vESgmsUFyopDy0VdLh7b7BCB/bF4wA5/tTSeijtdFi4k6dDnoqfonspLuJnKFzYtQxbrlh/CvH9c0CP6KHBodAesyHeIwk8tgv26XN+htMXj4OXxy/Lh+SFjP/NHaA91Xz1EyogAAAAAElFTkSuQmCC>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAAC80lEQVR4Xu2YWahOURTH10XGzEMUEuWRMiV5MBUhQ4Y3U1eGF8MLKTxIMpPkRZ6U3DI/8GC6yZsp402RUkpdCS9C4v9vn33POuv7ztc+34u7dX71q7PW3vd2715nT0ekpKSkpN3SCU62yQIMgW9gK7wOx2eb22iCH+F7eNa0eZbA2/Ab3Am7ZpuDmWoTMTAK/qnhurRrBSPhOzgQdoBz4ZdMD8cMeBgOhmPhTThCdwD74CNxg8iX4xy8B7vrToEstIkYmCWVg+/lWz407VrBRXjI5FbAniruAz+pmAwQ97MazgLOLk9n+AzuVrlQoizEAZsA2+FpmzRwgFksO6Bj4DQVnxfXz8LcchX/hBNVTLiccfYUJcpCVHvjXsP+NlkFrvkc0F4qd1A9k5eSX4iTJuasWJrEw+BnuKitRzj1/Ey7gxvkHJvMgcsSB/AJnA7niyuOp0Hcm55XiBsqZhGY+w33w2bJFqoI/0UhjtpEDbiOX5Z0T+EgTlDtPVSbhbm7KuaG7otBL4grZD1EX4i98JdN1oCnIG6oC+BTSQeRpx7C2VWrEM0q3iRuKZoJTyTtdJ7qE0rUhegmbiBe2IYcusDn8JqKj4sbvB2+k7jjbF4h9EbPeIqK3yY5ntz0KSyEqAvB9d0vCSGsgj/gcJPfAO+omJe9vEL4kxnvIKdUG+kr6SzjLClC1IXYI+6fPmIbcrgKH9pkwlf1fEXyC7E5eZ4EF6s2Dw8Nul8oUReCy5JdVjy81L2CLckz4V2D/XlvsBxTz7y5cxO3cKZ4uCzyvmHhLfu7ZC96IURbCJ5+OKh0q2kjKyVtX53kesMH8H4Se5ZJ9l5Bton7HOJhHztY/N1c7jxcmniZ4yZeFPu7o4FrtN9UG00bGQRvJfZTeX5jegzPiDvz8y2/pNo1nHE8DXGDrrZUrReX50c/3vY/SFr0PPw9JtTQg8g/ZTTcIm52FKEjXAPXiluG8uDHvo1wFxxn2jy8zc8W93fU++W1pKQkav4CHUS473w8NeIAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAADD0lEQVR4Xu2YWahNYRTHF5kSGTKUISXJlCkPpnhQihdDoSgZHqVMJVMnJIUiD15M4QUpQ2ZylTxIyoMpQ+6DmbiShAf+/7P256y97t7OPm+2u3/1q/ut7zvdc9ba37A/kYKCgoJcMQtu9MEUusCX8Bc8BvvEu//QC+6Hz+FbSR43H16ADXALbBnvzsx4H8gL1+FaOB0uEU3qGjsghcGwHg6FLeA2+M4OMHyAF+Ek0eJdi3eX/98XuAh2gE9hHWxlB2Vkmg/kBSbeymJUgwm6Lzo+wGLUm3ZgJDwMm0ft/qKfY8IDbG827SlRbJmJZSW3heBTWiuHRBN1zsW3wtWmzdliixUY4doc083F7qXEq9GkCnFeNEmnXbwEj5r2OkkuhKWn6JgwYwI3ovgcF68Gl9hcMhMegC9Ef/igeHciZ0XHnnLxDaJre+Ck6LiJohsxN+rjpp+Mgz9cjFwR/WyW/cqS20J8hptgP9Gl5A0cEBvRGBYuaWnaBR+Z9m2pzJxhoqelfXCMGTNVdKP2hELwu9VCbgvR1rX54ymLkkYb0aOoTWBn+E10Ew+wP2lpYowzkUyGX01f4JLoOLuJZyG3hfBwhjABi32HY4LosZRrO09Ru+EJeNmMeSbpheCsIqMkuRBhRtR6cvpvCsEnnQko+Y4EuJ+8Ek0+C7MTHjT9tyS9EGGj7wt/mr5Aneg4vujVQi4LMRvOdbH3oglY6uJZOAIXmnbYSzyMhU2b7x9st690l7kTxf1Rtxq5LASvHea5GH/8XdEEEV5PPIAPo79JR9Gn2CaZG7BPeldJPhFx3BDXtuf/7lFsr4llJZeFGC6V5BI+ld/haBPj0sCk0AVRrLVognkVQbjh34z0rBA9ogZmSOO7rDOi1x7tojb3mwaJf7es5LIQhEsRE7EHvha9krDwzfZqJE9GgeXwiehy9FH0JY8zJQleDJZEx7KAzeLd5esOzrrHcAf8JNUv77ZL5QHJIt/U/2k6iT6lK+FY11eN3qJ7yd+OumQgXC/6wsdNPQnOMvatgj1cX0FBQZPgN4p6ujHsuxhbAAAAAElFTkSuQmCC>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAACnUlEQVR4Xu2YS6hNYRTH/15FKXmO6A6MlMgIketVygCZUV5JMkGKwhViIM+BmMgjjJQk5U0RyWNAiYiMFJJECAP+/9b+rrXXOWfnxOCe9v7Vr+63vrXP6a61z/d9ewMVFRUVXZ6h9DB9RU/Stvx0IZvoY/qRXg5zidOw+Wf0UJiLDIyBJpkYA63CFPqeXqTTYA25nstozDBYcefRAXQ5HZTLAFbTlbAcNeQXrPH1mEHvx2CTzI6BVuE7PU67Z+MvsGL168yoT2/6kvbPxt1gv4xZnRnWKH2e5wS9EmL6vq/0Yfb3v9CSjRhFr4WY7soxIRbpQd/AiucZEcZn6LcQ6wkr9pYQF5NQ0kZspDti8C9IBbsdJwI3UFvYPllsW4iL/9GIOTHQCuiOVTPa6QX6lk7IZdRnDaxgl2DL2jvYRr3IJ5FlqF2axsGuXRDiorSN0MaoYp6lo2GnJS03431SHfbCCvYAtsFro15Ff/qkBuyBNa5vnECJG/EZ9kvwpOVkboh71DjlxD1iO90XYp6dKC50aRuhU8/BENMxVsU4EuKeo7AcnZI8G+iLEEvMhF1T1KjSNuIubI/wnIMVQ3d9I3bjzx7hWUc/hZjQaUoPdEVNEKVthO76rSF2HlaMUyHuWQLLiQ9+62Hrf+Q58t8zlk5140RpGzGY3gwxPSmrGCOzsZ6Cn9Cn2d8J/Zr0MOjR64sVIXaMTg4+otOzeU9pGyH0j/sjq8b+7l2YxeRiF9dd/cONh9MPsAe2hF5tpGujbS4vsRQ214785zRDyzZiP31NN8NePxyAva5IDKFXM3VM9WiJ0lFWa78+41Z+GvdQ2wCp01Z6pSLifPKOy4nsQm1+kfFg0SXRZtoBe5PaLGth182nvcJcRUVFRRP8Brr3sL28ToSNAAAAAElFTkSuQmCC>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAUCAYAAAD2rd/BAAACH0lEQVR4Xu2VS0hVURSGFxQNohoUTXRyMxr0IBJBHASFIohJOhCCaBAETmqkiE5CoVHlC1GwbBBY0cS5ITqIBHFmD4QKFEQQRyVYqIit371W9z/7PhSi2fngB/e319l3efbZ54ikpKSkHJTdAnnGRUWo1PzSLGvaNY2adxLWWMqW/cXXf665qdm2cQsXKSWa75pvmns8ETe6o+nggn2YknDd+ch/MM9cMjdA7qG5TU2G/BvzpeT24GZxl8qS0/vi1+LOMo/MM33mbpC7bg4ZMndas2UuB2znv+BbinV6yU+YZz6au0oOO+MNfzZ3i1wOvzV3JDx3PzQzmgeaw1xUhCbJLs5Z1ZRTXYbm2J8h7w2+iMYJIOc19zX1kt2KOc1RqitGv+Q2jOCxcC6Qv0I+Q94bfB2NE7zSnKAx/3cj5AtRpVmQUI8zEDftXCTHDePMxPV+4PI2HOOHBfkZzeVjQ0Jtp+aQ5quNPcetLkOuwhw4R94bLPhIPDE5Sa7LXN4L8oAaPBIxeK9i7i65L+aqyWGH/LdWzN0ml2DQ5Di5HnPIInkcDpzyVnIAdS8jB45ImLtM7qm5ZnJ15pBRc6ckvJdzGq7RrEloxvkk2QX4CzNGHgfIwXhdkncNNNgcc9YcGne6zSHXyONLC3eS3B6QODS42/7VQYa5SMLh9Dk+NNPk30tY462N/b3KwONr2qaptb/hHnORckwza0Fdyn/lD/JD0YCLwBkMAAAAAElFTkSuQmCC>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAADD0lEQVR4Xu2YWahOURTHF5kyJzxICUkyKw+mbihTicSDSF/qPkk8KJmSvJhSkqEMmYWElKEQ8kDejEm6DyIikRAe+P9b57D2cr57zhcPd+f86tf99tr7O313r3P22meLlJSUlDR5esJ9sAEegb3C7ky6+AAY5gMJV+FXeBvucX2ks2u3hKvhbBcvwlgfiIXx8C28BCeKJuRaMCKbTfAxnC86YQfh92CEshROhx3gOPgRTgpGiHyBp+ESOA0+hN9Ef1utzPCBWOCdegg2T9qf4A/Y6deIbLaJjrPuDUaIdBe9XisT2w3vmDbx13kJRwcjihNlIoaILhuWyXC4i2WxFa7zQUed6MT2N7GVok9OCxPjGNv+G6JMxCq4wQcLskXyE0GmuDaXsOsu9i8TMdMHYuCMaDJ4516Er+GYYER1WCPWi9YJTuR9ODUY8ScsyqxHc1yc3x8oWqc+wF2SvRkoQpSJuAsvw3NwqOhu6TMcZQdVYaPopHEt7yY6eZzQanQVTXzWGMaew4rouBfwFmxnxhQlykRwB8MnwXJDdGJmubinN+zhYjfhYhezsFa8g/1c3Bdm1qm0aNe6ZEWZiGdwp4txeeAk7HfxIpwVrQGNUQ8fwda+wzBSfu+gfNLyiDIR3EayRljOi04Al6taYRKu+KBjhOj1+c5QDU5+mogJri+PKBPBu97vfC6ITsBJF7cMFi3Om138lITf41vuAdMmHUWvf8LEeC37Vs7PaSJYu2ohykSwyHJdtzwRnYBBSZvHH1xKuDviZ1IRXb9ZsFPawzewr4mx6PrizF0ZY/NMjL/BTvgK0THcANRKlIkg/IftlpVt+5QsSGK0ksTaiCanLmkTnlH5SV8muguzHBNdvtI3eXJcwprB7zyV/Lf7LKJNxHbRO3ctPAx3wGamn8cUnDhq9/ZcuxtE37DvwfdwruknnGy+SfPsiOO4Q2Oy/CEfj1heJX+Pii6PfFobg8tieoMU8YF+rWkzQPS0c43vyIGnpLzreYjX1vVZeOi3XPT6/JxFH7gQLvIdJSUl/ws/Aahsr+bIR0i7AAAAAElFTkSuQmCC>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAADPUlEQVR4Xu2YWaiNURTHl5lkSIgSRR5kDslQpgclIZEnXDdFUZIHCcmTqVCmFzwYkjlDhgxFXkRkTqYHkSkZQpT4/61v37P2cj7nnHi4X75f/bvfXnud79671tlrr71FcnJycmo9HaBt0BNoF9Qpnk6lPfQQeg2dgPrF0zXsh95Aj6Htbo60dOMG0BJokrOXw1BvyAojRIN0CholmpDzkUdxOosmrg1UFxoDvYs8lHHQFKgp1ARaAHWLPES+QAeheaLvuQN9E/3bKmW8N2SFr9AO0WCST9APqEWNR3EYuDXONhVqZsaNoWdmHDjjxvx9Vs+hwZFH+WQyEb2gc842GurrbB6WEgaMybDwfcPNmEGhn+ej6OoI0Ke+Gf8NmUzEYmilN5bJC9EANje21eaZ9BT1WQHVSWwM+NoaD+VfJmKCN2SBw6LJGAadhF5CQyKPdFiWGMDrorV8rGhyPJelsHq6QKskLl+E891F96n30BaoVeRRPplMxBXoNHQE6i3aLX2GBlmnFBqKJjLU9e9Q/8hDYUdm6//9ePoXtD+FqqDWovvKJdENvlIymQjWaq4EywXRwEx0dk876KZoV3RDCoH2JYbt5CxoPvRK1IerwuI3Zu5T9OOm7d9Xikwm4hG02dlYHhiEYv1+oBF0CzpqxutFP7coOImWGyY7wHPHWVE/H3zLACkktqubK0UmE8H6zT3Cckw0ACxXaUwXbXs7Ovtsic8gW0VXi4VtMt+/3NktDH5IxEg3V4pMJoLfeh8QnpAZgH3ObmGSrnpjAjfbwAf5vUMiPKscMGOurj5mzOeQCO5dlZDJRPBUfNHZuJkyAD2SMTfbu9C95JksTHx4bvCsM8/HRU/fnrcS70H8G2zAWd74fnZPlZLJRBD+w7Zl9WVjWmKjqhIbT93suNjZWCZLfK7gN5v3ULw7CrDbsquB7BHdZwLs3B5I6dN9MTKbiA2i7eIyaCe0SQqHL9JWdIOlbG/P1XRN9G5qo+jl3yEzHxgoWnpYovaKdk72VE14xcIzCH/uFi2PfP+fCOeYcnVbP1a74SUcbzuX+okS1INmQDNFD2tpsIWdK3qbmnZQ4yViNTTHT+Tk5Pwv/AQgl7qB4dwc/gAAAABJRU5ErkJggg==>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGIAAAAUCAYAAAB23ujSAAAC4ElEQVR4Xu2YS6hNURjHv5BX3kVEFAbIm4FkQGbCRTEgLjEgFBkIA2HmEQMpAyRDuR5FErrlHXk/S91bFGVCtwiJ//+sve/+1nf2uXtlZLn7V7866/vW+e4969t7nbO2SElJSUlULIa7bLAGg+Bb+AleglP8dIU+NiDV9TvAzfA5/AL3wv7ejHBm2EAsNMLtcAFcD3/DbXpCDYbDJnELxoWcAz97Mxx3xa9/Tarrn4On4TTYQ9z/wNqD9aRA6mwgFvihtVysEM7AfSa2HPY0sUfi1//ppyt8hevU+LW4uftVLJRoGzHbBgLgdsOFYjM04+FME7tvxnmw1jtxdxa5nMSutM4Ip101gnwUt1i9VIx7u+WeDeRQD0eqcbO42ntULBRugVGyCB6H78V9+DF+uibcljifW88sOFdccyx3xK9/0U/nwnnf4FCbCCDaRvBXym5xVyS3Fi7mKG9GPp3hWcn2/l9wqjfDcVv8+jek7fpjYYu4xv4N0TaiuxmnC8tFa4uB8CmcD59I9r5OehKYZMZdpbg+87dgN5sIINpGWHiHcCFW24SiC3wGL6jxIXHvsz9N8yiqf0rcnBM2EcB/04gmcYuw0yYU9fC7VO/ha+F1E8ujqP4mye6wCSZXRJSNWAKXmhhPyVyAjSauOQ8f2GAC76gU1ufJ22LrH4VD1HieZI3YoOIhRNmIY3CZifHDP5Zsr+cCvYSvktdkazIvb58/qF6z/gs1JnwEoevzVxpr7Wid4bY3xn7AYSoeQpSNmCj+lchTMbccPmpIWSHZ1bkyifUWd1C7mYxT+JxKnytYn1/oKazPxuj6hCfp0WrMxvPv2WdSIUTZCMKtiF+6R+AHONlPywB4NbGfivMZ00NxV/1hcVtQg8qn8GGerv/GT1c4Ke7RRyM8IK4JPBymJ+080nNMqHyg+E/TFy6EW+B0kyuiI1wF18ARJqfR9fmePHhncpvkFjXO5EpKStoFfwDurK1wAZA25QAAAABJRU5ErkJggg==>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAUCAYAAAD2rd/BAAACEElEQVR4Xu2WPUhVYRjHn6EGQxoSl1xufgx+DGVLg6QYgSi0JAjRFrjYVEGTGDT5QaQUZrVlDUVTg0hzIG6l6aBgEDrUUkRKiejzv+/zcv7nOfd6dXAQ7g/+4Pk9z3vuc8/xPeeKlClT5tjyQfNTc1Vz0tWOhN0imeKmffgr2bWcb5rK2Ez+ueaaZtuO+6kHnNWsalY0t7jgP2BHc58bSuDX+7QmrdJi7jG5QXP/NDnyb8zXkMvDJ/+uqU2XS4J1S5q3Eu4KBkA2NH+oDzyS0N9DrsMc8sRctea/uQybXhwSfxXBPfPXnf9ivo1cozlk0VwfuQxbmpuaWc0vzSfNbc0JbjoEOMcPzSnnc5IMcYH8OfJxwJfuOAXkZ82ApluSWzEv2Q89CFg74qXSJMkQ58nnyMcBX7vjFNOa03TM3+4Z+YNQL2HdZV9QmqXwwNgzfuC44QoO7HkoSfNvVyvFOwnrKnxB0lfyIvkG8nHAov8SuHWQH8kNmSu4YB9wVdE/4wvEVwk9neQumUPWzd0gl2LC5HtyY+aQNfLYHNjld8gx2KhYg6tcjFEJPb3kuswhL8xVSXguZwa+ImFHY5jIgiQn4DfMK/LYQB48Q1FDXzHqJPRg8MgDc0g7eTzT4c6QywO5LOFqx7cO8pSbJGzOWONNExmXUJv0BQd68Da9K+H3B/6GG+YmCa/zOQv6yhwpewc5wFa6+QNIAAAAAElFTkSuQmCC>