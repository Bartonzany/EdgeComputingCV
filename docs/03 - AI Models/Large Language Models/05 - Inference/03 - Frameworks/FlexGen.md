---
aliases:
  - FlexGen
  - FlexGen High-Throughput Generative Inference of Large Language Models with a Single GPU
Authors: Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang
Year: 2023
Title: FlexGen High-Throughput Generative Inference of Large Language Models with a Single GPU
DOI: 10.48550/arXiv.2303.06865
tags:
  - 大模型推理
  - 模型推理
---

## FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

- **网址链接:** [Open online](http://arxiv.org/abs/2303.06865)
- **作者:** Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang
- **引用:** Sheng2023
- **本地pdf:** [2023-FlexGen](../../../../../../asset/papers/2023-FlexGen.pdf)

---

### Abstract

大语言模型（LLM）推理的高计算和内存需求使得其仅能通过多台高端加速器实现。鉴于对延迟不敏感任务（如批处理）的需求日益增长，本文开启了在有限资源（例如单台消费级 GPU）上进行高吞吐量 LLM 推理的研究。我们提出了 FlexGen，一种**在有限 GPU 内存下运行 LLM 的高吞吐量生成引擎**。FlexGen 可以**通过聚合 GPU、CPU 和磁盘的内存和计算资源，灵活地配置在各种硬件资源约束下**。通过求解线性规划问题，它能够搜索高效的模式来存储和访问张量。FlexGen 进一步将权重和注意力缓存压缩至 4 比特，且精度损失可忽略不计。这些技术使 FlexGen 拥有更大的批处理大小选择空间，从而显著提升了最大吞吐量。实验结果表明，在单台 16GB GPU 上运行 OPT-175B 时，FlexGen 相比当前最先进的卸载系统实现了显著更高的吞吐量，首次在有效批处理大小为 144 的情况下达到了 1 token/s 的生成吞吐量。在 HELM 基准测试中，FlexGen 能够利用 16GB GPU 在 21 小时内完成 7 个代表性子场景下 30B 模型的基准测试。

### Summary

FlexGen 提出了一种高效的单GPU生成推理框架，旨在解决大型语言模型（LLMs）在有限硬件资源下的高吞吐量推理问题。通过优化内存使用和计算效率，FlexGen 能够在单GPU上实现接近多GPU集群的性能，显著降低了LLMs推理的硬件门槛。

**优点**：

- 高效的单GPU推理，降低了硬件成本。
- 高吞吐量，接近多GPU集群的性能。
- 灵活的内存和计算优化策略，聚合GPU、CPU和磁盘的内存和计算资源。

**缺点**：

- 在某些极端情况下，性能可能受到内存限制。
- 需要进一步验证在不同硬件配置下的通用性。

### ResearchObjective

在单个普通 GPU 上设计高效的卸载策略，以实现高吞吐量的生成式推理。FlexGen 聚合了来自 GPU、CPU 和磁盘的内存，并高效地调度 I/O 操作，同时结合可能的压缩方法和分布式流水线并行技术

### Background

此前降低大型语言模型（LLM）推理资源需求的努力主要集中在以下三个方向：

1. **模型压缩**：通过模型剪枝、量化和蒸馏等技术，显著减小模型占用的存储空间和计算复杂度（Dettmers et al., 2022; Yao et al., 2022; Frantar et al., 2022; Xiao et al., 2022）。例如，量化技术将模型权重从浮点精度降低到低精度（如INT8），从而减少内存占用和计算开销。
2. **协作推理**：通过去中心化的多卡并行计算，分摊推理成本（Borzunov et al., 2022）。例如，模型并行技术将模型参数分布到多个GPU上，从而解决单个GPU内存不足的问题。
3. **卸载技术**：将模型权重等参数从GPU内存卸载到CPU内存甚至硬盘中，同时结合模型量化技术进一步优化存储和计算效率（Aminabadi et al., 2022; HuggingFace, 2022）。例如，使用分层存储策略将不常用的参数卸载到CPU或硬盘，以减少GPU内存占用。

然而，这些技术虽然显著降低了LLM推理的资源需求，但也存在明显的局限性：

- 模型压缩和协作推理通常假设模型可以完全放入GPU内存中，因此在单个普通GPU上运行超大规模模型（如175B参数）时面临困难。
- 基于卸载的最先进系统由于I/O调度和张量放置效率低下，在单个GPU上无法达到可接受的吞吐量。例如，这些系统可能因批处理大小过小而受到瓶颈（在某些情况下，OPT-175B的批处理大小仅为1或2）。

### Methods

FlexGen 提出了一种基于三级内存层次结构（GPU、CPU、磁盘）的推理优化方法，旨在处理大语言模型（LLM）无法完全装入 GPU 内存的场景。通过将部分权重卸载到二级存储（CPU 或磁盘），并按需加载到 GPU 中，FlexGen 能够逐步完成推理任务。

计算图由多个方块组成，每个方块对应某一层的一个批次计算，相同颜色的方块共享相同的权重。为了实现高效的推理，FlexGen 需要找到一条满足以下约束的有效路径：

1. **顺序性** ：每个方块只能在其所在行左侧的所有方块计算完成后才能被计算。
2. **内存一致性** ：计算某个方块时，其所有输入（权重、激活值和 KV 缓存）必须已加载到同一设备。
3. **输出生命周期管理** ：
    - 激活值需保留至其右侧的兄弟方块被计算完毕。
    - KV 缓存需保留至同一行中最右侧的方块被计算完毕。
4. **内存容量限制** ：任何时刻，设备上的张量总大小不能超过其内存容量。

最终目标是设计一条有效路径，遍历并计算所有方块，同时最小化总执行时间。总执行时间由两部分组成：

- **计算时间** ：在 GPU 或 CPU 上执行计算所需的时间。
- **I/O 时间** ：在设备之间传输张量（如权重、激活值和缓存）所需的时间。

![](../../../../../images/LLM/Pasted%20image%2020250310210021.png)

根据上面抽象出来的推理计算图，要设法在图中找出一条能够最小化执行时间的路径，其中包括在设备之间移动张量时的计算成本和 I/O 成本。

#### zig-zag并行策略

直观上，遍历图 2 中的图有两种顺序：逐行遍历和逐列遍历。这是合理的，因为这是最快完成一个批次生成的方式，并且 KV 缓存可以在处理完一行后立即释放。然而，由于每两个连续的方块不共享权重，此调度必须重复加载权重，导致巨大的 I/O 成本。为了减少权重的 I/O 成本，我们可以逐列遍历图。一列中的所有方块共享权重，因此我们可以将权重保留在 GPU 上以便重用，仅加载/卸载激活和 KV 缓存。然而，我们不能一直遍历到列的末尾，因为激活和 KV 缓存仍需要存储。因此，我们必须在它们填满 CPU 和磁盘内存时停止。考虑到这些因素，我们收敛到一种锯齿形块调度，如图 3(b) 所示。

![](../../../../../images/LLM/Pasted%20image%2020250310212408.png)

![](../../../../../images/LLM/Pasted%20image%2020250310212322.png)


### Conclusion

FlexGen 提供了一种高效的单GPU生成推理框架，能够在资源有限的环境下实现高吞吐量的LLMs推理。实验结果验证了其在内存和计算优化方面的有效性。

### ExampleWithCode

- **代码地址**：[FMInference/FlexLLMGen: Running large language models on a single GPU for throughput-oriented scenarios.](https://github.com/FMInference/FlexLLMGen)

### Notes

---

## 参考引用

### 网页链接

### 论文引文

- [LLM推理论文精读1 -- FlexGen[ICML'23] | Qingwei Ji](https://qingweiji.github.io/post/10_flexgen/)
- [Flexgen LLM推理 CPU Offload计算架构到底干了什么事情？ - 知乎](https://zhuanlan.zhihu.com/p/615021309)
- [FlexGen论文笔记 - 知乎](https://zhuanlan.zhihu.com/p/664164593)
- 

