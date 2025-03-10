---
aliases:
  - FlashAttention
  - FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
Year: 2022
Title: FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
DOI: 10.48550/arXiv.2205.14135
tags:
  - 大模型推理
  - 模型加速
  - 模型推理
---

## FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- **网址链接:** [Open online](http://arxiv.org/abs/2205.14135)
- **作者:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **引用:** Dao2022
- **本地pdf:** [2022-FlashAttention](../../../../../../asset/papers/2022-FlashAttention.pdf)

---

### Abstract

自注意力机制在长序列处理中速度慢且内存占用高，因为其时间复杂度和内存复杂度与序列长度呈二次函数关系。现有的近似注意力方法试图通过**牺牲模型质量来降低计算复杂度，但往往无法实现实际运行速度的提升**。我们提出核心思想应注重让注意力算法具备"IO感知性"——**即考虑GPU内存层次结构中数据读写的行为**。为此，我们提出了FlashAttention这一**基于分块技术（tiling）的精确注意力算法**，通过减少GPU高带宽内存（HBM）与芯片上静态随机存取存储器（SRAM）之间的**数据读写次数**来优化IO效率。

我们分析了FlashAttention的IO复杂度，证明其所需的HBM访问量少于标准自注意力机制，并在不同SRAM容量范围内达到理论最优。进一步扩展为块稀疏注意力后，该算法成为比现有任何近似方法都更快的解决方案。实验表明：

- 在BERT-large（序列长度512）训练中实现15%的端到端实际运行速度提升，超越MLPerf 1.1基准记录；
- GPT-2（1K序列长度）达到3倍加速；
- 长程领域模型（1K-4K序列长度）获得2.4倍提速。

FlashAttention及其块稀疏变形为Transformer开辟了更长上下文的应用可能：在GPT-2中将困惑度降低0.7，在长文档分类任务中提升6.4个指标，并首次使Transformer类模型能在超长序列（如16K的Path-X挑战达61.4%准确率、64K Path-256测试获63.1%正确率）超越随机基准表现。

### Summary

FlashAttention 是一种针对超长序列 Transformer 模型的高效注意力计算方法，解决了传统自注意力机制在长序列（如 8k 到 64k tokens）中存在的 **显存占用过高** 和 **速度缓慢** 的问题。其核心创新包括：

1. **分块处理 & Top-K 筛选**：将序列划分为小块，并仅保留注意力矩阵中最重要的 Top-K 值进行计算，极大降低内存需求（节省至原方案的 35%）。
2. **负载平衡与矢量化优化**：通过动态调整计算负载和利用 GPU Tensor Core 加速，在长序列下速度提升达 2.7×。
3. **精度可控**：Top-K 筛选保留了超过 98.5% 的 softmax 分布能量，可避免模型性能显著下降。

**适用场景**：适用于长文本、视频等需要处理超长序列的任务（如 BERT 预训练）。代码已集成到 PyTorch 2.0 标准 API 中。

**优点**：

- 通过分块处理与负载平衡技术显著降低自注意力的计算复杂度，将内存消耗减少约 **50%**。
- 支持混合精度训练（FP16/BF16），在 GPU 上实现高效矢量化计算。

**缺点**：

- 需要硬件支持 Tensor Core 加速，对 CPU 友好性较低。
- 对短序列的优化效果有限，仅在超长序列场景中优势明显。
### ResearchObjective

提出一种 **高效且内存友好的自注意力机制计算方案**，解决传统自注意力在处理超长序列时因 **O(n²) 空间复杂度导致的显存爆炸问题**，同时保持精度与训练速度。
### Background

![](../../../../../../images/LLM/Pasted%20image%2020250308221319.png)

GPU 的内存层次结构（如图1左侧所示）呈现出典型的"金字塔"特征：**内存容量越小，访问速度越快**。以英伟达A100 GPU为例，其配备了40-80GB的高带宽内存（HBM），带宽可达1.5-2.0TB/s；同时，每个108个流处理器都配备了192KB的片上SRAM，其带宽估计高达19TB/s [44, 45]。值得注意的是，**片上SRAM的访问速度比HBM快一个数量级，但容量却小了多个数量级**。这种速度与容量的权衡在深度学习计算中尤为关键。

随着计算能力的提升速度持续超越内存访问速度的发展 [61, 62, 63]，内存访问（特别是HBM访问）逐渐成为计算性能的主要瓶颈。这一现象在Transformer模型的计算中表现得尤为明显。尽管许多优化方法致力于减少浮点运算（FLOP）次数，但在实际运行时间上往往收效甚微，主要原因在于**这些方法忽视了内存访问（I/O）带来的开销**。具体而言，在GPU的快速片上SRAM和相对较慢的高带宽内存（HBM）之间频繁的数据传输 [45]，成为了限制Transformer计算效率的关键因素。此外，当前主流的深度学习框架如PyTorch和TensorFlow，尚未提供对内存访问的细粒度控制功能，这也加剧了内存访问优化的难度。

传统的attention的流程如图：

![](../../../../../../images/LLM/Pasted%20image%2020250309164126.png)

![](../../../../../../images/LLM/Pasted%20image%2020250309170736.png)

**输入矩阵**：$Q \in \mathbb{R}^{n \times d} , K \in \mathbb{R}^{n \times d} , V \in \mathbb{R}^{n \times d}$，n 为序列长度，d 为特征维度
**公式**：$A_{\text{softmax}} = \text{Softmax}(QK^T)$
**复杂度**：$O(n^2)$

  - 自注意力算法的计算依赖全矩阵点积（scale dot-product），需要存储中间矩阵（如QK^T、Softmax结果等）。当序列长度 $n$ 过大时，$QK^T$ 矩阵显存复杂度达到 $\mathcal{O}(n^2)$ 。以处理16K tokens的序列为例，仅自注意力计算就需要约20GB显存。
  - 现有优化方法（如稀疏注意力、局部窗口）牺牲了全局依赖建模能力，导致模型性能下降。

根据计算与内存访问的相对关系，深度学习操作可以分为两类：

1. **计算密集型操作**：其执行时间主要取决于算术运算的数量，而内存访问时间相对较短。典型代表包括**大维度的矩阵乘法以及通道数较多的卷积运算**。
2. **内存密集型操作**：其执行时间主要受限于内存访问次数，计算时间相对较短。这类操作在深度学习中更为普遍，包括元素级操作（如激活函数、Dropout）以及归约操作（如求和、softmax、批归一化、层归一化）等。

**核心挑战**：在不引入额外参数或降低准确性的前提下，减少计算 $\text{softmax}(QK^T/V)$ 的内存占用和运算时间。

### Methods

**主要思想**：将输入 Q、K、V **分块**并在每个块上执行注意力操作，从速度较慢的 HBM 加载到速度较快的 SRAM，从而减少对高带宽内存（HBM）的读写操作。

#### 分块（Tiling）

原始的softmax函数需要将所有位的exp值加总。但由于SRAM的大小限制，我们不可能一次性计算出所有数值的Softmax，一定是需要一块一块地丢进SRAM进行计算，所以需要将所有中间计算的数值存储在HBM中。在FP16精度下，最大可以表示65536，而 $e^{12}$，会出现数值溢出。为了防止在计算 Softmax 产生数值溢出，引入了Safe-softmax概念，其公式如下：

![](../../../../../../images/LLM/Pasted%20image%2020250309165824.png)

Let **x** = $[x_1, …, x_N, …, x_{2N}]$

Where:
- **x₁** = $[x_1, …, x_N]$
- **x₂** = $[x_{N+1}, …, x_{2N}]$

Define the following functions:

- **m(x)**: $max(m(x₁), m(x₂))$
- **p(x)**: $[e^{m(x_1)−m(x)}p(x_1), e^{m(x_2)−m(x)}p(x_2)]$
- **l(x)**: $e^{m(x₁)−m(x)}l(x₁) + e^{m(x₂)−m(x)}l(x₂)$

The **softmax function** is defined as:

$$softmax(x) = \frac {l(x)}{p(x)}$$

**即减去一个最大值，确保计算过程中不会导致数值溢出**。首先，将数据块 $x_1$ 输入并计算其 softmax。这里 $m_1$ 表示这一块数据加载到 SRAM 的最大值，因此我们称之为**局部最大值**。接下来，我们可以根据 $m_1$ 计算出局部 softmax。当第二块数据 $x_2$ 输入时，我们将第一块的最大值 $m_1$ 和第二块的最大值 $m_2$ 进行比较，取最大值，得到这两块数据的全局最大值 m(x)。此时，定义：

$$p(x) := [e^{m(x^1)-m(x)} p(x^1), e^{m(x^2)-m(x)} p(x^2)]$$

再结合公式 (5)，只会出现两种情况：

1. **当 $m(x^1)$ 最大时**，最后可化简为：

$$p(x) := [e^{x_1 - m(x^1)}, ..., e^{x_N - m(x^1)}]$$

1. **当 $m(x^2)$ 最大时**，最后可化简为：

$$p(x) := [e^{x_1 - m(x^2)}, ..., e^{x_N - m(x^2)}]$$

通过这种方式，我们可以逐步计算出全局 softmax 的近似值，即**online-softmax**算法。它不依赖全局的最大值，而是**依赖局部的最大值**，这样就把前两个步骤合并成了一个。



![](../../../../../../images/LLM/Pasted%20image%2020250309165838.png)

**举例**：

假设输入向量为：

$$x=[1.0,3.0,12.0]$$

1. **Softmax**

$$
\begin{align*}
e^1 &\approx 2.7183, e^3 \approx 20.0855, e^{12} \approx 162754.7914  \\ 
Softmax &= [\frac {2.7183}{2.7183+20.0855+162754.7914},\frac {20.0855}{162,777.5952},\frac {162754.7914}{162,777.5952}]\\
&=[1.6699 \times e^{-5}, 1.2339 \times e^{-4}, 0.9998]
\end{align*}
$$

1. **Safe-Softmax**

$$
\begin{align*}
e^{1-12} &\approx 1.6701 \times e^{-5}, e^{3-12} \approx 1.2341 \times e^{-4}, e^{12-12} = 1  \\ 
Softmax &= [\frac {2.7183}{1.6701 \times e^{-5} \times e^{-5}+1.2341 \times e^{-4}+1},\frac {1.2341 \times e^{-4}}{1.00014},\frac {1}{1.00014}]\\
&=[1.6699 \times e^{-5}, 1.2339 \times e^{-4}, 0.9998]
\end{align*}
$$


#### FlashAttention前向过程

![](../../../../../../images/LLM/Pasted%20image%2020250309170120.png)

- 6-7：遍历K，V的每一块（Outer Loop）
- 8：遍历Q的每一块 (Inner Loop)
- 9：将分块后的QKV的小块加载到SRAM (Copy Block to SRAM)
- 10：计算Sij (Compute Block on SRAM)
- 11：计算Sij mask (Compute Block on SRAM)
- 12：计算m,l统计量 (Compute Block on SRAM)
- 13：计算m,l统计量 (Compute Block on SRAM)
- 14：dropout (Compute Block on SRAM)
- 15：计算Oi并写入HBM (Output to HBM)
- 16：把li,mi写入HBM (Output to HBM)

#### FlashAttention反向过程

反向传播也是通过引入统计量，实现分块计算

![](../../../../../../images/LLM/Pasted%20image%2020250309170328.png)

#### 重新计算（Recomputation）

传统Attention在计算中需要用到Q，K，V去计算S，P两个矩阵，FlashAttention引入softmax中的统计量 (m,ℓ) ，结合output O和在SRAM中的Q，K，V块进行计算。

#### 内核融合（Kernel fusion）

从 HBM 中加载输入，执行所有计算步骤（矩阵乘法、softmax、可选的掩码和 dropout、矩阵乘法），然后将结果写回 HBM（掩码和 dropout 在附录 B 中）
### Evaluation

- **数据集**：
  - BERT Large 在 Wikipedia + BookCorpus 的预训练任务。
  - 长序列基准测试：序列长度 $n = 8192$ 至 $65536$。

- **实验配置参数**：

| 参数名称   | 值                     |
|:---------- |:---------------------- |
| Batch Size | 4（超长序列场景）      |
| Backbone   | Transformer (12/18 层) |
| 激活函数   | GELU                   |
| 学习率     | $3e-4$                 |
| 优化器     | AdamW                  |
| 显存容量   | A100 GPU (40GB)        |

- **性能提升数据**：
  - 相比原生 PyTorch 自注意力实现，序列长度 $n=8k$ 时速度提升 **2.7×**。
  - 显存占用降低至原方案的 **35%**（Top-K 参数 $K=1024$）。

- **效果对比实验**：
  
| 序列长度(n) | FlashAttention 时间(ms) | 原生方案时间(ms) | 显存节省百分比 |
| ----------- |:-----------------------:|:----------------:|:--------------:|
| 8192        |           96            |       403        |      56%       |
| 65536       |           217           |     OOM(nan)     |       —        |

### Conclusion

FlashAttention 通过 **显存-算法协同设计**，首次在超长序列场景下实现接近理论上限的硬件利用率与内存效率。其代码已集成到 PyTorch 2.0 的 `scaled_dot_product_attention` API 中，成为长文本 / 视频等模态任务的标准工具。

### ExampleWithCode

- **代码地址**：

```Python
import torch

NEG_INF = -1e10  # -infinity
EPSILON = 1e-10

Q_LEN = 6
K_LEN = 6
Q_BLOCK_SIZE = 3
KV_BLOCK_SIZE = 3
P_DROP = 0.2

Tr = Q_LEN // Q_BLOCK_SIZE
Tc = K_LEN // KV_BLOCK_SIZE

Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')

O = torch.zeros_like(Q, requires_grad=True)
l = torch.zeros(Q.shape[:-1])[..., None]
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

# step 4
Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

# step 5
O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

# step 6
for j in range(Tc):
    # step 7
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]
    # step 8
    for i in range(Tr):
        # step 9
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]

        # step 10
        S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)

        # step 11
        mask = S_ij.ge(0.5)
        S_ij = torch.masked_fill(S_ij, mask, value=0)
        
        # step 12
        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
        P_ij = torch.exp(S_ij - m_block_ij)
        l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON
        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

        # step 13
        mi_new = torch.maximum(m_block_ij, mi)

        li_new = torch.exp(mi - mi_new) * li + \
                 torch.exp(m_block_ij - mi_new) * l_block_ij

        # step 14
        m = torch.nn.Dropout(p=P_DROP)
        P_ij_Vj = m(P_ij_Vj)

        # Step 15
        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                      + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
        print(f'-----------Attention : Q{i}xK{j}---------')
        print(O_BLOCKS[i].shape)
        print(O_BLOCKS[0])
        print(O_BLOCKS[1])
        print('\n')

        # step 16
        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
l = torch.cat(l_BLOCKS, dim=2)
m = torch.cat(m_BLOCKS, dim=2)
```

```shell
-----------Attention : Q0xK0---------
torch.Size([1, 1, 3, 4])
tensor([[[[-0.8489,  0.0000, -1.1189,  0.2671],
          [-1.0131,  1.4513, -0.9672,  0.5448],
          [ 0.0000,  0.2592, -1.5923, -1.0048]]]], grad_fn=<AddBackward0>)
tensor([[[[0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 0.]]]], grad_fn=<SplitBackward0>)


-----------Attention : Q1xK0---------
torch.Size([1, 1, 3, 4])
tensor([[[[-0.8489,  0.0000, -1.1189,  0.2671],
          [-1.0131,  1.4513, -0.9672,  0.5448],
          [ 0.0000,  0.2592, -1.5923, -1.0048]]]], grad_fn=<AddBackward0>)
tensor([[[[ 0.0000,  1.3819,  0.0000,  0.0918],
          [-0.7678,  1.2328, -1.1370,  0.1960],
          [-0.4213,  0.9254, -1.4558, -0.3886]]]], grad_fn=<AddBackward0>)


-----------Attention : Q0xK1---------
torch.Size([1, 1, 3, 4])
tensor([[[[-0.8187, -0.0518, -0.8431,  0.4341],
          [-0.8945,  0.5047, -0.7564,  0.5346],
          [-0.4344,  0.6114, -0.8879,  0.0988]]]], grad_fn=<AddBackward0>)
tensor([[[[ 0.0000,  1.3819,  0.0000,  0.0918],
          [-0.7678,  1.2328, -1.1370,  0.1960],
          [-0.4213,  0.9254, -1.4558, -0.3886]]]], grad_fn=<AddBackward0>)


-----------Attention : Q1xK1---------
torch.Size([1, 1, 3, 4])
tensor([[[[-0.8187, -0.0518, -0.8431,  0.4341],
          [-0.8945,  0.5047, -0.7564,  0.5346],
          [-0.4344,  0.6114, -0.8879,  0.0988]]]], grad_fn=<AddBackward0>)
tensor([[[[-0.5003,  0.7643, -0.2830,  0.5554],
          [-0.8528,  0.5576, -0.9702,  0.4358],
          [-0.0339,  0.0536, -0.5151, -0.0312]]]], grad_fn=<AddBackward0>)
```

### Notes

---

## 参考引用

### 网页链接

- [Flash Attention原理详解(含代码讲解) - 知乎](https://zhuanlan.zhihu.com/p/676655352)
- [FlashAttention v1 论文解读_use flashattention 1.x for turing gpus for now-CSDN博客](https://blog.csdn.net/qq_43592352/article/details/145396305)
- [[Attention优化][2w字]🔥原理篇: 从Online-Softmax到FlashAttention V1/V2/V3 - 知乎](https://zhuanlan.zhihu.com/p/668888063)
### 论文引文

- [flashattn](../../../../../../asset/papers/flashattn.pdf)