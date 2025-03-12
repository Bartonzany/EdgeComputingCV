---
aliases:
  - Continuous Batching
title: How continuous batching enables 23x throughput in LLM inference while reducing p50 latency
date: 2025-03-11 20:35:58
excerpt: 大模型推理
tags:
  - 大模型推理
  - 模型加速
---

## Continuous Batching

- **网址链接:** [Open online](https://www.anyscale.com/blog/continuous-batching-llm-inference)

---

### Abstract

由于 LLM 巨大的 GPU 内存开销和计算成本，在大多数应用中，机器学习工程师通常通过 **内部调整（如量化和对 CUDA 核的定制）** 来优化。然而，由于 LLM 通过迭代生成其输出，并且 LLM 推理通常涉及内存而不是计算，因此在很多实践中，优化系统级批处理可以使性能差异达到10倍甚至更多。

一种最近提出的优化方法是**连续批处理（Continuous batching）**，也称为**动态批处理**或基于迭代级的批处理。其具有如下惊人的效果：

- 基于 vLLM，使用连续批处理和连续批处理特定的内存优化，可以实现多达23倍的吞吐量提升；
- 对于 HuggingFace 本地生成推理，使用连续批处理，可以实现8倍的吞吐量提升；
- 基于 NVIDIA 的 FasterTransformer，使用优化过的模型实现，可以实现4倍的吞吐量提升。

### Summary

Continuous Batching 打破了传统静态批处理的固有模式。传统的静态批处理，就像是把一批货物整齐打包好，凑够一定数量或者等一段时间后，才统一进行运输处理。而 Continuous Batching 则不同，它允许在运输过程中，随时装卸货物，也就是实时地处理请求。简单来说，它是一种动态调整批次的技术，能够根据请求的实际情况，灵活地安排计算资源。

**优点**：

**缺点**：

### ResearchObjective

### Background

#### prefill和generation

大模型推理分为两部分 prefill 和 generation，如下图所示：

![](../../../../../../images/LLM/Pasted%20image%2020250311211408.png)

- Prefill 阶段：在进行生成之前大模型需要输入的 prompt，所以第一步 T1 是 Prefill 阶段，黄色部分即为模型的输入
- Generation 阶段：T2，T3，T4，END 红色为模型的结束序列标记。

在 Prefill 阶段需要对 "Hello", "World", "!" 三个 Token 进行 Attention 计算，而在 generation 阶段每次只需要对最后一个 Token 进行 Attention 计算。但是模型会有以下问题：

1. **LLM 推理是内存 IO 限制，而不是计算限制**。换句话说，目前加载 1MB 的数据到 GPU 所需的时间比 1MB 的数据在GPU上计算所需的时间长。这意味着 LLM 推理的吞吐量很大程度上取决于您能将多少批数据装入到高速GPU 内存中；
2. **GPU 内存的消耗量随着基本模型大小和标记长度的增加而增加**。如果我们将序列长度限制为 512，那么在一个批处理中，我们最多只能处理28个序列；一个序列长度为 2048 则批处理大小最多只能为7个序列。需要注意的是，这只是一个上限，因为中间计算结果没有留下存储的空间。

### Methods

#### 朴素批处理 Naive batching

![](../../../../../../images/LLM/Pasted%20image%2020250311213217.png)

如上图所示，在第一遍迭代（左）中，每个序列从Prompt（黄）中生成一个标记（蓝色），经过几轮迭代完成所有序列生成。这种方式会很明显造成内存碎片，白色部分的显存虽然可能当前阶段不会用到甚至可能永远生成结束也不会用到，但是仍然会申请这部分显存。

#### 连续批处理 continuous batching

![](../../../../../../images/LLM/Pasted%20image%2020250311220146.png)

每次 prefill 或者 generation 之前都会进行一次 batching， 图中阴影部分为未使用的显存。  

左上图中Prefill阶段因为不生成新的Token，所以只需要按照最长的S4来确定batch的序列长度多长，右上图中蓝色部分为在第一轮 generation 后生成的 Token，此时我们再次形成的batch长度会加一，左下图在第二轮 generation 中生成了“END”，那么代表 S2 和 S4 已经结束不再，可以直接将结果收集起来不再参与后续的batching 中，右下图中展示了 S2，S4被清除之后的内存占用状态，之后的状态取决于时候有新的请求加入。

### Evaluation

#### 1、实验设置

**数据集**：
**Batch Size**：
**优化器**：
**Epoch**：
**数据增强**：
**Baseline**：
**激活函数**：
**学习率**：
**衰减率**：
**框架**：
**平台**：

#### 2、实验结果

1. **吞吐量大幅提升**：经过实际测试，Continuous Batching 能够让 GPU 的利用率大幅提升，达到 80% 以上。在使用 NVIDIA A100 GPU 时，吞吐量更是能提升 3 - 5 倍。这意味着在相同的时间内，大模型可以处理更多的请求，就像一条原本只能通行少量车辆的道路，通过优化后可以容纳更多的车辆同时行驶，大大提高了系统的处理能力，尤其在高并发的场景下，优势更加明显。
2. **降低延迟**：从数据上看，Continuous Batching 可以使平均响应时间减少 30% - 60%。而且，它对长尾延迟（P99 延迟）的改善效果也非常显著。这对于用户体验来说至关重要，想象一下，当我们使用智能客服时，原本可能需要等待好几秒才能得到回复，现在瞬间就能收到答案，这种即时响应的感觉是不是很棒
3. **资源效率优化**：在显存占用方面，Continuous Batching 能够减少约 40% 的显存占用。这意味着在同样的硬件条件下，可以支持更高的并发量。对于云服务提供商来说，这就像是用同样大小的仓库，能够存放更多的货物，大大降低了运营成本，提高了资源的使用效率


### Conclusion


### ExampleWithCode

- **代码地址**：

### Notes

---

## 参考引用

### 网页链接

- [Continuous Batching：一种提升 LLM 部署吞吐量的利器](https://www.high-flyer.cn/blog/continuous-batching/)
- [从continuous batching到vLLM中的batching - 知乎](https://zhuanlan.zhihu.com/p/688551989)
- [大模型推理黑科技：Continuous Batching，你了解多少？ - 知乎](https://zhuanlan.zhihu.com/p/25634151188)