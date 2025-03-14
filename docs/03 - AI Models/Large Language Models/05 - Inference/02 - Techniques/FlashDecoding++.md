---
aliases:
  - FlashDecoding++ Faster Large Language Model Inference on GPUs
  - FlashDecoding++
Authors: Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, Yu Wang
Year: 2024
Title: FlashDecoding++ Faster Large Language Model Inference on GPUs
DOI: 10.48550/arXiv.2311.01282
tags:
  - 大模型推理
---

## FlashDecoding++: Faster Large Language Model Inference on GPUs

- **网址链接:** [Open online](http://arxiv.org/abs/2311.01282)
- **作者:** Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, Yu Wang
- **引用:** Hong2024
- **本地pdf:** [2024-FlashDecoding++](../../../../../asset/papers/2024-FlashDecoding++.pdf)

---

### Abstract


随着大型语言模型（LLM）在各个领域中的重要性日益增加，LLM推理的性能对于大规模LLM应用至关重要。然而，在加速LLM推理方面，以下挑战仍未得到解决：

1. **同步的部分Softmax更新**：Softmax操作需要在每个部分Softmax结果之间进行同步更新，导致LLM中注意力计算的开销约为20%。
2. **扁平GEMM（矩阵乘法）的计算未充分利用**：在LLM推理中，执行GEMM的矩阵形状扁平，导致计算未充分利用，并且在之前的设计（如cuBLAS、CUTLASS等）中填充零后性能损失超过50%。
3. **静态数据流导致的性能损失**：LLM中的内核性能依赖于不同的输入数据特征、硬件配置等。单一的静态数据流可能导致LLM推理中不同形状的GEMM性能损失高达50.25%。

我们提出了**FlashDecoding++**，一个支持主流LLM和硬件后端的快速LLM推理引擎。为了解决上述挑战，FlashDecoding++创新性地提出了以下技术：

1. **带有统一最大值的异步Softmax**：FlashDecoding++引入了统一最大值技术，用于不同的部分Softmax计算以避免同步。基于此，提出了细粒度的流水线技术。
2. **通过双缓冲优化扁平GEMM**：FlashDecoding++指出，不同形状的扁平GEMM面临不同的瓶颈。为此，引入了双缓冲等技术。
3. **基于硬件资源自适应的启发式数据流**：FlashDecoding++根据输入动态性，使用不同的硬件资源（如Tensor Core或CUDA核心）启发式地优化数据流。

由于FlashDecoding++中优化的多样性，其在NVIDIA和AMD GPU上相比Hugging Face实现分别实现了高达4.86倍和3.93倍的加速。此外，FlashDecoding++在主流的LLM上相比最先进的LLM推理引擎平均实现了1.37倍的加速。
### Summary

**优点**：

**缺点**：

### ResearchObjective

### Background

LLM推理中的主要操作如图2中的操作①到⑥所示，包括线性投影（①和⑤）、注意力机制（②、③和④）以及前馈网络（⑥）。为简化起见，图中未展示位置嵌入、非线性激活、掩码等操作。预填充阶段和解码阶段的操作在数据形状上有所不同。由于每次只处理一个标记（批量大小=1）或少量标记（批量大小>1），解码阶段的输入矩阵为平面形状矩阵甚至向量。

![](../../../../../images/LLM/Pasted%20image%2020250312211005.png)


由于soft-softmax需要根据其他部分softmax结果进行更新，不可避免地引入了数据同步操作。根据我们的性能分析结果，这种同步更新操作在NVIDIA Tesla A100 GPU上对输入长度为1024的Llama2-7B推理的注意力计算中导致了18.8%的开销。

$$
\begin{aligned}
m(x) &= \max(m(x'), m(x'')) \\
f(x') &= e^{m(x') - m(x)} f(x') \\
f(x'') &= e^{m(x'') - m(x)} f(x'') \\
l(x) &= f(x') + f(x'') \\
softmax([x', x'']) &= [f(x'), f(x'')] \div l(x)
\end{aligned}
$$
### Methods

为了提升大模型推理速度，在FlashDecoding基础上做了三点主要改进：

1. 作者把online softmax改成一种更容易并行的计算方式，从而增加了softmax算子计算的并行度；
2. 对细长矩阵的[Flat GEMM](https://zhida.zhihu.com/search?content_id=235987579&content_type=Article&match_order=1&q=Flat+GEMM&zhida_source=entity)操作降低zero-padding尺寸并用双Buffer降低访存开销。
3. 启发式地选择Tensor Core或CUDA Core进行GEMM运算。

![](../../../../../images/LLM/Pasted%20image%2020250312212219.png)

#### 异步Softmax

FlashDecoding++最主要的创新点，在于提出了基于统一max值的异步softmax。我们知道，safe-softmax的计算公式中，需要先求每行x的最大值，然后减去这个max(x)之后，再做softmax以防止数值溢出。

$$
\begin{aligned}
\text{softmax}(x) & =\frac{\left[e^{x_{1}-m(x)}, \ldots, e^{x_{d}-m(x)}\right]}{\sum_{i} e^{x_{i}-m(x)}} \\
& =\frac{\left[e^{x_{1}-\phi}, \ldots, e^{x_{d}-\phi}\right]}{\sum_{i} e^{x_{i}-\phi}}, \forall \phi \in \mathbb{R}
\end{aligned}
$$

FlashDecoding++认为，这个max值，不一定需要online计算max(x)，而是可以是一个合理的先验值 ϕ 。我们对上边的公式分子分母提取公因式，可以得到：

$$
\begin{aligned}
\text{softmax}(x) &= \frac{e^{-m(x)} [e^{x_1}, \dots, e^{x_d}]}{e^{-m(x)} \sum_i e^{x_i}} \\
&= \frac{e^{-\phi} [e^{x_1}, \dots, e^{x_d}]}{e^{-\phi} \sum_i e^{x_i}}, \quad \forall \phi \in \mathbb{R} \\
&= \frac{[e^{x_1}, \dots, e^{x_d}]}{\sum_i e^{x_i}}, \quad \forall \phi \in \mathbb{R}
\end{aligned}
$$

可以发现，使用先验值 ϕ 与直接计算max(x)，最终softmax的结果，在数学上是等价的。问题在于如何确定这个先验值 ϕ 以防止数值异常，比如对于一个很小的x，这时如果使用一个非常大的先验值 ϕ，就可能导致概率值异常（结果太大可能会溢出 float32，太小又可能出现精度问题影响效果）。因此作者提出设置一个全局max，达到不需要其他部分softmax计算结果的情况下计算每个部分softmax结果。假设每个xi的a < xi−φ < b，以确保精度并避免溢出。然后，单独处理部分softmax操作。然而，当xi−φ≤a或xi−φ≥b时，对于xi所在的向量x，终止异步部分softmax计算。然后使用同步部分softmax方案(在FlashAttention和FlashDecoding中使用)重新计算softmax。根据图5所示的统计数据，这种重新计算方案避免了溢出，同时引入了可以忽略不计的开销。


![](../../../../../images/LLM/Pasted%20image%2020250312214705.png)

#### 双Buffer Flat GEMM

![](../../../../../images/LLM/Pasted%20image%2020250313211301.png)

```C++
for (int i = 0; i < m; i++) {           // 遍历 A 的行
    for (int j = 0; j < n; j++) {       // 遍历 B 的列
        for (int p = 0; p < k; p++) {   // 遍历 A 的列和 B 的行
            C[i][j] += A[i][p] * B[p][j];
        }
    }
}
```

- 计算操作数：其中M、N、K分别为3层循环次数，2为最内层循环的1次乘法和1次加法；
  
  $$
  M*N*(k+k-1)+N*M=2*M*N*K
  $$

 - 内存访存量：4MNK，其中M、N、K分别为3层循环次数，4为最内层循环中对C、A、B中元素的内存访问次数

假设 GEMM 中两个相乘的矩阵大小分别为 M * K 和 K * N，同时每个 GEMM Tile 会对 K * N 的矩阵进行分块，每块大小为 Bn * Bk （不足则进行填充），那么每个 GEMM Tile的计算量为 $2MB_nB_k$，内存访问量为 $M * B_k + B_n * B_k$，共有 $N * K / B_n * B_k$ 块。算上把乘法结果写入的内存访问，整个 GEMM 过程中计算与内存的比值为：

$$
\frac{2 \ast M \ast B_n \ast B_k \ast \frac{N + K}{B_n \ast B_k}}
{(M \ast B_k + B_n \ast B_k) \ast \frac{N + K}{B_n \ast B_k} + M \ast N}
= \frac{2 \ast M \ast K}{K + \frac{M \ast K}{B_n} + M}
$$



![](../../../../../images/LLM/Pasted%20image%2020250313211244.png)

于是作者发现了：计算和内存比与$B_N$正相关，而并行度与$B_N$负相关。下图展示了GEMM在不同 $B_N$ 和N下的性能（归一化后）。本文总结了两个关键结论：

![](../../../../../images/LLM/Pasted%20image%2020250313213123.png)

1. 当 N 较小时，flat GEMM是parallelism-bounded。NVIDIA Tesla A100中有108个Streaming Multiprocessors (SMs)，于是应该将B_N设置为一个相关的数（128或256）
2. 当 N 较大时，flat GEMM是memory-bounded。通过隐藏memory access latency可以提高性能。

### Evaluation

![](../../../../../images/LLM/Pasted%20image%2020250313213841.png)

![](../../../../../images/LLM/Pasted%20image%2020250313213932.png)



### Conclusion

FlashDecoding++提出了具有统一最大值的异步softmax、具有双缓冲的平面GEMM优化和具有硬件资源自适应的启发式数据流三种新颖设计。FlashDecoding++在主流llm和硬件上均取得了有效的加速效果：

### ExampleWithCode

- **代码地址**：

### Notes

---

## 参考引用

### 网页链接

- [🔥原理&图解FlashDecoding/FlashDecoding++ - 知乎](https://zhuanlan.zhihu.com/p/696075602)
- [大模型推理加速之FlashDecoding++：野生Flash抵达战场 - 知乎](https://zhuanlan.zhihu.com/p/665361668)
- [[论文分享]LLM推理加速——FLASHDECODING++_vllm flash-CSDN博客](https://blog.csdn.net/bmfire/article/details/134599948)
- [GPU语言模型加速：FlashDecoding+的并行softmax优化,-CSDN博客](https://blog.csdn.net/ymk1998/article/details/134397471)
- [FlashDecoding++_flash decoding-CSDN博客](https://blog.csdn.net/yzsjwd/article/details/134403397)

### 论文引文

- 


