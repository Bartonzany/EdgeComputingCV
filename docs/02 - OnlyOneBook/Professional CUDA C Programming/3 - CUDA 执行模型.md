---
aliases:
  - CUDA 执行模型
title: CUDA 执行模型 CUDA Execution Model
date: 2025-02-27 12:58:19
excerpt: CUDA 入门，给新手一些指引
tags:
  - CUDA
---

## 3 - CUDA 执行模型 CUDA Execution Model

---

### 0. 前言

从本篇开始，我们将深入探讨 CUDA 编程最核心的领域——**硬件架构与程序执行模型**。需要明确的是，学习 CUDA 的终极目标并非掌握某种特定语法，而是通过理解其底层运行机制来实现**极致的性能优化**。毕竟在 GPU 计算领域，效率的提升才是真正最能体现性能优化的地方，没有谁会满足于仅仅实现一个 Hello World 级别的程序。

在之前的章节中，我们学习了如何编写和启动核函数、进行计时以及统计时间，也初步理解了线程模型与内存模型的基本概念，后者作为 CUDA 编程的重中之重，将在后续几篇中在进行系统性分析。本文我们聚焦于对程序优化具有决定性指导意义的**底层架构原理**。

理解 CUDA 的设计哲学至关重要：**如果我们能够按照硬件设计的思路来设计程序，往往能够获得成功；相反，一旦偏离了这种设计思路，程序的表现就可能不尽如人意。** 这种硬件与软件的高度契合性，正是 GPU 编程区别于传统 CPU 编程的重要特征。

### 1. 概述 Introducing the CUDA Execution Model

**CUDA 执行模型本质上是对 GPU 并行硬件架构的抽象映射**。在芯片研发流程中，架构师首先确定硬件计算单元的功能特性与执行逻辑，随后由硬件工程师完成物理电路设计。当执行模型的设计规范与晶体管级实现存在矛盾时，需通过跨团队协同迭代优化，直至达成架构设计与制程工艺之间的妥协。这种深度协同产生的硬件特性（如 SIMT 执行机制、内存子系统结构等），最终构成了编程模型的设计基准，使得 CUDA 编程接口能够精准反映底层硬件的实际行为特征。

以内存体系为例，其层次化的 **线程组织结构（线程->线程块->网格）** 直接对应着 GPU 芯片内计算核心的物理排布方式。架构团队在定义多级并行硬件计算单元后，集成电路工程师据此设计 **流式多处理器(SM)** 及其调度单元，驱动开发者随后封装出对应层级的API接口。这种自底向上的设计哲学，使得开发者能够通过编程模型直观地操控硬件资源。

深入理解这种硬件与编程模型的对应关系（CUDA 执行模型），开发者可针对性地实施相应优化。通过**指令流水线排布提升计算吞吐量**，运用内存局部性原理**降低延迟**，最终实现趋近于硬件理论最大值的执行效率。

#### 1.1. GPU 架构概述 GPU Architecture Overview

GPU 架构是围绕一个 **流式多处理器（SM）** 的扩展阵列搭建的。通过复制这种结构来实现 GPU 的硬件并行。

![Streaming Multiprocessors](/images/Professional%20CUDA%20C%20Programming/Streaming%20Multiprocessors.png)

上图展示了CUDA架构中的关键组件，包括：

-   **CUDA 核心**：执行并行计算任务的基本计算单元。
-   **共享内存/一级缓存**：用于线程之间的数据共享和快速访问，显著提高数据传输效率。
-   **寄存器文件**：存储每个线程的局部数据，提供更快的访问速度。
-   **加载/存储单元**：负责数据的输入和输出，确保内存与处理单元之间的高效交互。
-   **特殊功能单元**：处理特定类型的计算任务，比如浮点运算或整数运算，提升计算的灵活性和速度。
-   **线程束调度器**：管理和调度执行的线程束，优化资源利用和提高线程并行度。

这些组件协同工作，使得CUDA能够实现高效的并行计算。

##### 1.1.1. 流式处理器 Streaming Multiprocessors

在 GPU 架构中，**流式处理器（SM）** 是实现并行计算的核心单元。每个 SM 拥有同时支持数百个线程并发执行的能力，而现代 GPU 通常集成数十个 SM 以形成强大的并行计算阵列。当 CUDA 核函数被调用时，线程网格中的多个线程块将根据硬件资源状态动态分配到可用的 SM 上。

需特别注意两个关键特性：
1. 线程块一旦分配给特定 SM 后即形成绑定关系，整个执行周期内不会迁移至其他 SM。也就是一旦一个线程块被分配给某个 SM，它就**只能**在这个 SM 上执行，并不能重新分配到其他 SM；
2. 多个线程块可以同时被分配到**同一个** SM 上进行执行。
   
在 SM 内部，线程块内的多个线程通过 **单指令多线程（SIMT）** 架构展开**线程级并行**，而每个线程内部则通过指令**流水线化**（如指令预取、乱序执行等技术）实现**指令级并行（ILP）**。这种"粗粒度线程并行+细粒度指令优化"的双重加速机制，使得 GPU 能够在处理大规模数据时表现出卓越的性能

##### 1.1.2. 线程束 Warp

CUDA 采用 **单指令多线程（SIMT）** 架构来管理执行线程，其核心调度单元称为**线程束（Warp）**。在不同的设备中，线程束的大小可能有所不同，但目前主流GPU设备普遍采用32位宽的线程束架构，即每个 SM 上有 32 个线程。在每个 SM上，会有多个线程块（block），而每个线程块又包含多个线程（通常有几百个，但不会超过某个最大值）。

从机器的角度来看，在某一时刻 T，SM 上实际上只会同时执行一个线程束，**即 32 个线程会并行地执行同一条指令**。这种同步执行的机制包括处理分支条件时的情况：线程束中的每个线程都遵循相同的指令路径，即使在有条件分支的情况下也如此。通过这种方式，单指令多线程（SIMT） 架构能够有效地利用GPU的并行计算能力，提升整体性能。

##### 1.1.3. SIMD vs SIMT

在并行计算架构中，**单指令多数据（SIMD）** 和 **单指令多线程（SIMT）** 是两种重要的模式。单指令多数据的执行属于**向量机**，例如当我们需要对四个数字执行加法运算时，SIMD 指令可一次性完成原本需要四次的运算。然而这种机制的问题就是过于死板，不允许每个分支有不同的操作，所有分支必须同时执行相同的指令，**无法根据数据特征执行差异化的计算逻辑**。

相比之下，**单指令多线程（SIMT）** 架构则更加灵活。尽管两者都以相同的指令广播给多个执行单元，SIMT 允许某些线程选择不执行。在同一时刻，所有线程接收相同的指令，但不是所有线程都必须执行。这种灵活性使得 SIMT 能够实现**线程级别的并行**，而 SIMD 则更倾向于实现**指令级别的并行**。

SIMT 相较于 SIMD 具备以下关键特性：

1.  **每个线程都有自己的指令地址计数器**：线程可以独立控制执行的位置，可实现代码分支的动态跳转。
2.  **每个线程都有自己的寄存器状态**：保证每个线程的上下文独立性，支持差异化的数据存取。
3.  **每个线程可以有一个独立的执行路径**：允许线程根据条件执行不同的指令，这种设计使得同一批线程可以根据数据特征选择性地**激活**或**挂起**，既保持了硬件层面的执行效率，又在编程层面实现了逻辑分支的灵活性。

而上面这三个特性通过**编程模型**所介绍的，为每个线程分配唯一的标识符（`blockIdx` 和 `threadIdx`）得以实现。这些特性确保了线程之间的独立性，使得在执行过程中各线程能够根据实际需求灵活运行。

##### 1.1.4. 数字 32 Number 32

数字 32 在 CUDA 架构中被视作一个“神奇的数字”，它的由来根源于硬件系统设计，是由集成电路工程师决定的，因此，软件工程师只能对此适应和接受。

从概念上讲，32 代表的是流式处理器（SM）在以 SIMD 方式同时处理工作的粒度。可以这样理解：在某一时刻，某个 SM 上的 32 个线程在执行同一条指令，这 32 个线程可以执行也可以不执行，但不能执行其他指令。只有当所有需要执行该指令的线程都完成后，才能继续处理下一条指令。这可以用一个形象的老师分苹果的比喻来说明：

-   **第一次分苹果**：老师把苹果分给所有32个小朋友。你可以选择不吃苹果，但不吃的同时也没有其他选择，只能在旁边看着。等到所有的小朋友都吃完后，老师会收回没有吃的苹果，以防浪费。
-   **第二次分橘子**：尽管你很喜欢橘子，但有的小朋友可能不喜欢吃。在这种情况下，那些不爱吃的小朋友同样不能做其他事情，只能继续看你吃。待你吃完后，老师会继续收回那些没有吃的橘子。
-   **第三次分桃子**：大家都很喜欢桃子，这次大家一起享用。等到所有人都吃完桃子后，老师发现没有剩下的，便继续发放其他种类的水果，直到所有水果都分发完毕，才可以结束这一天的课程。

这个比喻很好地说明了 SIMT 架构中线程之间的同步与选择性执行的关系。通过这种方式，CUDA 能高效地利用 GPU 的并行计算能力，同时保证每个线程在执行时的独立性。

##### 1.1.5. CUDA 编程的组件与逻辑 CUDA Programming Components and Logic

下图从逻辑角度和硬件角度描述了CUDA编程模型对应的组件

![logical view and hardware view of CUDA](/images/Professional%20CUDA%20C%20Programming/logical%20view%20and%20hardware%20view%20of%20CUDA.png)

共享内存和寄存器是在 CUDA 的流式处理器（SM）中不可或缺的关键资源。线程块内的线程通过共享内存和寄存器进行相互通信和协作。它们的分配和使用会显著影响程序的性能。

需要注意的是，虽然在编程模型的层面上，所有线程看起来像是并行执行的，但从微观角度来看，线程块实际上是以批次在物理硬件上进行调度的，即**同一线程块内的不同线程可能在执行进度上不一致，但同一线程束内的线程始终拥有相同的执行状态**（即上文提到的部分线程处于活跃执行状态，部分可能因内存访问延迟而处于等待挂起状态，而上下文切换由硬件自动管理且无额外开销，使得资源的利用更加高效）。这种并行性可能导致资源竞争，多个线程以不确定的顺序访问同一数据，可能导致不可预测的行为。

针对这个问题，CUDA 提供了**块内同步**的机制，通过 `__syncthreads()` 实现块内同步，但在不同块之间则无法实现同步。通过这种机制，CUDA能够在确保高并发的同时降低了上下文切换的成本，提高计算性能。

#### 1.2. Fermi 架构 The Fermi Architecture

Fermi 架构是第一个完整的 GPU 架构，所以了解这个架构是非常有必要的

![Fermi Architecture](/images/Professional%20CUDA%20C%20Programming/Fermi%20Architecture.png)

Fermi架构逻辑图如上，具体数据如下:

1. 512 个加速核心 CUDA 核
2. 每个 CUDA 核心都有一个全流水线的整数算数逻辑单元 ALU，和一个浮点数运算单元 FPU
3. CUDA 核被组织到16个 SM 上
4. 6 个 384-bits 的 GDDR5 的内存接口
5. 支持 6G 的全局共享内存
6. GigaThread 引擎，分配线程块到 SM 线程束调度器上
7. 768KB 的二级缓存，被所有 SM 共享

而 SM 则包括下面这些资源：

- 执行单元（CUDA核）
- 调度线程束的调度器和调度单元
- 共享内存，寄存器文件和一级缓存

每个流式处理器（SM）配备有 16 个加载/存储单元，因此在每个时钟周期内，可以进行 16 个线程（相当于半个线程束）的源地址和目标地址的计算。同时，**特殊功能单元（SFU）** 负责执行一些固有指令，例如**正弦、余弦、平方根**以及**插值**等。在每个时钟周期中，SFU 能够针对每个线程执行一条固有指令。

**每个 SM 还包含两个线程束调度器和两个指令调度单元**。当一个线程块被分配给某个SM时，线程块内的所有线程会被划分成多个线程束。然后，两个线程束调度器会从中选择两个线程束，并利用指令调度单元存储这两个线程束将要执行的指令（两个班级的老师，即指令调度器各自掌控着自己班级的水果，负责指挥分发水果给学生）

像第一张图上的显示一样，每 16 个 CUDA 核心为一个组，还有 16 个加载/存储单元或 4 个特殊功能单元。当某个线程块被分配到一个 SM 上的时候，会被分成多个线程束，线程束在 SM 上交替执行：

![Fermi Execution](/images/Professional%20CUDA%20C%20Programming/SM%20Execution.png)

上面曾经说过，每个线程束在同一时间执行同一指令，同一个块内的线程束互相切换是没有时间消耗的。Fermi 上支持同时并发执行内核。并发执行内核允许执行一些小的内核程序来充分利用 GPU，如图：

![Fermi Execution](/images/Professional%20CUDA%20C%20Programming/Fermi%20Execution.png)

#### 1.3. Kepler 架构 The Kepler Architecture

Kepler 架构作为 Fermi 架构的后代，有以下技术突破：

- 强化的 SM
- 动态并行
- Hyper-Q技术

技术参数也提高了不少，比如单个 SM 上 CUDA 核的数量，SFU 的数量，LD/ST 的数量等：

![Kepler Architecture1](/images/Professional%20CUDA%20C%20Programming/Kepler%20Architecture1.png)

![Kepler Architecture2](/images/Professional%20CUDA%20C%20Programming/Kepler%20Architecture2.png)

首先是动态并行（Dynamic Parallelism），这也是 CUDA 发展史上首次允许 **GPU 内核启动内核**。这种特性突破了传统 GPU 编程的线性执行限制，使得**递归算法和复杂任务调度完全能够在GPU端自主完成**，流程如下:

![Dynamic Parallelism](/images/Professional%20CUDA%20C%20Programming/Dynamic%20Parallelism.png)

此外，Kepler 架构还引入了 Hyper-Q 技术，这是一种用于增强 CPU 和 GPU 之间同步的硬件机制。通过Hyper-Q，CPU 可以在 GPU 执行任务的同时继续处理更多的工作。Fermi 架构下 CPU 控制 GPU 只有一个队列，Kepler 架构下可以通过 Hyper-Q 技术实现多个队列，如下图所示：

![Hyper-Q](/images/Professional%20CUDA%20C%20Programming/Hyper-Q.png)

这种改进显著提升了 GPU 的利用率和灵活性，使得多个 CPU 线程可以同时向 GPU 提交任务，减少了等待时间和资源闲置率。

两种不同架构计算能力概览：

![Compute Capability1](/images/Professional%20CUDA%20C%20Programming/Compute%20Capability1.png)

![Compute Capability2](/images/Professional%20CUDA%20C%20Programming/Compute%20Capability2.png)

#### 1.4. 使用 Profile 进行优化 Profile-Driven Optimization

《Professional CUDA C Programming》原文翻译这个标题叫“配置文件驱动优化”，驱动这个词在这里应理解为动词，更合适的翻译应该是“根据 profile 文件的信息进行优化”，从而更准确地反映其内容。

性能分析可以通过以下几个方面进行：：

1.  应用程序代码的**空间(内存)或时间复杂度**
2.  特殊指令的使用
3.  函数调用的**频率**和**持续时间**

程序优化必须建立在对硬件特性和算法过程充分理解的基础上。如果对这些缺乏理解，仅依靠试验，那么结果往往不尽如人意。因此，**深入理解平台的执行模型和硬件特性是优化性能的基础**。

开发高性能计算程序的两个关键步骤：

1.  确保结果的正确性和程序的健壮性
2.  优化程序的执行速度

Profile 性能分析工具可以帮助我们深入观察程序的内部行为：

-   一般而言，一个原生的核函数应用并不能发挥出最佳效果。我们不能期望一开始就编写出最优的核函数，而是需要通过性能分析工具来发现性能瓶颈。
-   CUDA 将 SM 中的计算资源在多个驻留线程块之间进行分配。这种分配方式可能导致某些资源成为性能的限制因素，而性能分析工具能够帮助我们识别这些资源的具体使用情况。
-   CUDA 提供了对硬件架构的抽象，使用户能够有效控制线程的并发性。性能分析工具不仅可以监测性能，还能优化和可视化这个过程。

总结一句话：**要优化速度，首先需要掌握如何使用性能分析工具**。

可供使用的性能分析工具包括：

-   nvvp（NVIDIA Visual Profiler）
-   nvprof

限制内核性能的主要因素包括但不限于以下几点：

-   存储带宽
-   计算资源
-   指令和内存延迟

### 2. 理解线程束执行的本质 Understanding the Nature of Warp Execution

在前面的讨论中，我们已经大致介绍了 CUDA 执行模型的基本流程，包括线程网格、线程束、线程之间的关系，以及硬件的基本结构，例如流式处理器（SM）的架构。对硬件而言， CUDA 执行的实质是线程束的执行，因为硬件并不关心每个块内线程的具体情况和执行顺序。SM 只依据机器码进行操作，而先后执行的顺序则是硬件设计所直接决定的。

从表面上看，CUDA 执行所有线程是并行的，似乎没有明确的执行顺序。然而，硬件资源往往是有限的，无法同时处理百万个线程。因此，从硬件的角度来看，**物理上执行的实际上只是线程的一部分，而每次实际执行的这一部分正是我们之前提到的线程束**。这意味着在任何给定的时刻，只有一部分线程被调度并执行，硬件以线程束为单位高效地管理和分配计算资源，以实现最大的并行度和性能。

#### 2.1. 线程束和线程块 Warps and Thread Blocks

**线程束是流式处理器（SM）中基本的执行单元**。当一个网格被启动时（网格启动等同于核函数启动，每个核函数对应一个独立的网格），其包含的线程块会根据硬件资源分配到特定 SM 上，每个线程块在 SM 内部会被划分为若干个连续线程束。当前主流的 GPU 每个线程束通常包括 32 个线程（尽管现在的 GPU 是 32 个线程，但不保证未来还是 32 个，可能会变为 64）。

在一个线程束内，所有线程以单指令多线程（SIMT）的方式执行，每个线程在同一时间执行相同的指令，但处理的数据则是各自独立的私有数据。下图反应的就是逻辑，实际，和硬件的图形化：

![logical view and hardware view of a thread block](/images/Professional%20CUDA%20C%20Programming/view%20of%20a%20thread%20block.png)

CUDA 执行模型中的线程块本质上是一种**逻辑抽象机制**。因为在计算机中，内存的物理寻址是以一维线性方式存在的，所以访问线程块中的线程也是按一维方式进行的。但 CUDA 通过多维编址的方式为开发者提供了更符合直觉的编程接口。在编写程序时，我们可以以二维或三维的方式表示线程块，这样做是为了方便处理图像或其他三维数据。使用二维或三维的线程块符合直觉，更容易理解和管理。

- 在一个线程块中，每个线程都有一个唯一的编号（可能是一个三维编号），通常表示为 threadIdx。
- 在一个网格中，每个线程块也有一个唯一的编号（可能是一个三维编号），通常表示为 blockIdx。

因此，每个线程都有一个在其所在网格中唯一的编号。通过这种方式，程序员可以方便地组织和管理不同线程的工作，使代码更具可读性和可维护性。

当一个线程块中有 128 个线程的时候，其分配到 SM 上执行时，会分成4个块：

```shell
warp0: thread  0, ... , thread 31
warp1: thread 32, ... , thread 63
warp2: thread 64, ... , thread 95
warp3: thread 96, ... , thread 127
```

**当编号使用三维编号时，x 位于最内层，y 位于中层，z 位于最外层**。例如 C 语言的数组，如果把上面这句话写成c语言，假设三维数组 t 保存了所有的线程，那么 $(threadIdx.x, threadIdx.y, threadIdx.z)$表示为：

```C
t[z][y][x];
```

计算出三维对应的线性地址是：

$$
tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x
$$

一个线程块包含多少个线程束呢？

$$
WarpsPerBlock = ceil(\frac {ThreadsPerBlock}{warpSize})
$$

`ceil` 函数是用于向上取整的数学函数，如下图所示的 $ceil(\frac{80}{32}) = 3$。在这个例子中，即使最后的半个线程束（即最后一个线程束中没有充分填满的线程）不活跃，它们仍然消耗 SM 的资源，例如寄存器和共享内存。这类似于前文提到的吃苹果的例子。

![allocate warps of threads](/images/Professional%20CUDA%20C%20Programming/allocate%20warps%20of%20threads.png)

线程束和线程块，一个是**硬件层面的线程集合**，一个是**逻辑层面的线程集合**，我们编程时为了程序正确，必须从逻辑层面计算清楚，但是为了得到更快的程序，硬件层面是我们应该注意的。

#### 2.2. 线程束分化 Warp Divergence

在执行线程束时，这些线程被分配到相同的指令，同时处理各自的私有数据。就像之前提到的分水果的例子，每次分配的水果都是相同的，但每个线程可以选择是否“吃”这些水果，而这个“吃”与“不吃”就对应了程序中的分支。在 CUDA 编程中，支持 C 语言中的循环结构，例如 `if…else`、`for` 和 `while`。然而，如果一个线程束中的不同线程包含不同的控制条件，当执行到这些条件时，就会面临不同的选择。

值得一提的是，CPU 在处理大量分支判断时，其程序逻辑会显得复杂。每个分支都可能产生两种执行路径，例如，如果有 10 个分支，可能会出现 1024 种不同的执行路径。CPU 采用流水线技术来提高执行效率，但如果每次都等到分支判断执行完再继续执行后续指令，会造成显著的延迟。为了解决这个问题，现代处理器引入了**分支预测技术**，以提前猜测可能的执行路径，从而减少延迟。

与此不同，GPU 的设计初衷是为了应对大量计算且逻辑相对简单的任务，因此在处理复杂分支时并不如 CPU 高效。CPU 更适合处理**逻辑复杂度高但计算量小**的应用，如操作系统和控制系统，而GPU则是为进行**大规模、并行计算**而设计，特别适合执行**数学运算**等任务。因此，在编程时，开发者需要根据具体应用的特性，合理选择使用 CPU 还是 GPU。

如下一段代码：

```C
if (cond) {
    //do something
} else {
    //do something
}
```

假设这段代码是核函数的一部分，当一个线程束中的 32 个线程执行这段代码时，可能会出现这样的情况：16 个线程执行 `if` 中的代码段，而另外 16 个线程执行 `else` 中的代码段。**同一个线程束中的线程，执行不同的指令，这叫做线程束的分化**。在线程束中，所有线程在同一指令周期内应当执行相同的指令，但由于执行路径的不同，线程束实际上出现了分化，这看似矛盾，但实际上并不矛盾。

解决这种矛盾的方式是让每个线程都执行所有的 `if` 和 `else` 部分。当满足条件的线程（即 `cond` 成立时）会执行 `if` 块内的代码，而不满足条件的线程（即 `cond` 不成立时）将会被“挂起”，等待其他线程完成执行。这就好比分水果的例子：如果你不爱吃某种水果，你只能旁观其他人吃。**如果条件分支还没有执行完毕，所有的线程都必须等待，直到线程束中的所有线程都执行完当前的指令（即下一次指令的执行）**。

**线程束的分化会导致显著的性能下降**，尤其是在条件分支越多时，整体的并行性会受到更加严重的削弱。需要注意的是，线程束分化只影响同一线程束中的线程，不同的线程束之间的分支执行是相互独立的。

执行过程如下：

![warp divergence](/images/Professional%20CUDA%20C%20Programming/warp%20divergence.png)

为了有效应对线程束分化导致的性能下降，我们可以采用线程束的方法来解决。其根本思路是**避免同一个线程束内的线程分化**。由于线程块中线程分配到线程束是有规律的而非随机分配，这一特点使得我们可以**通过线程编号来设计分支**，优化代码执行。

需要特别说明的是，当线程束中的所有线程都执行 `if`，或者都执行 `else` 时，不会产生性能下降。**只有在线程束内出现分歧时**，即部分线程执行 `if`，而其他线程执行 `else` 时，性能才会急剧下降。

既然线程束内的线程是可控的，我们就可以通过将所有执行 `if` 的线程组合在一个线程束中，或者将所有执行 `else` 的线程组合在另一个线程束中。这种方法有助于确保线程束内的线程始终执行相同的指令，从而有效提高执行效率。

下面说明线程束分化是如何导致性能下降的 `chapter03/simpleDivergence.cu`：

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/common.h"

__global__ void warmup(float* c) {
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a   = 0.0;
    float b   = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    // printf("%d %d %f \n",tid,warpSize,a+b);
    c[tid] = a + b;
}

__global__ void mathKernel1(float* c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float a = 0.0;
    float b = 0.0;

    if (tid % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }

    c[tid] = a + b;
}

__global__ void mathKernel2(float* c) {
    int   tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a   = 0.0;
    float b   = 0.0;

    if ((tid / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }

    c[tid] = a + b;
}

__global__ void mathKernel3(float* c) {
    int   tid   = blockIdx.x * blockDim.x + threadIdx.x;
    float a     = 0.0;
    float b     = 0.0;
    bool  ipred = (tid % 2 == 0);

    if (ipred) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }

    c[tid] = a + b;
}

int main(int argc, char** argv) {
    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size      = 64;
    int blocksize = 64;
    if (argc > 1)
        blocksize = atoi(argv[1]);
    if (argc > 2)
        size = atoi(argv[2]);
    printf("Data size: %d\n", size);

    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float* C_dev;
    size_t nBytes = size * sizeof(float);
    float* C_host = (float*)malloc(nBytes);
    CHECK(cudaMalloc((float**)&C_dev, nBytes));

    // run a warmup kernel to remove overhead
    double iStart = 0.0;
    double iElaps = 0.0;
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    warmup<<<grid, block>>>(C_dev);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("warmup			%d, %d	elapsed %lf sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(C_dev);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1		%d, %d	elapsed %lf sec \n", grid.x, block.x, iElaps);
    CHECK(cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaGetLastError());

    // run kernel 2
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(C_dev);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2		%d, %d	elapsed %lf sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // run kernel 3
    iStart = cpuSecond();
    mathKernel3<<<grid, block>>>(C_dev);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel3		%d, %d	elapsed %lf sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    CHECK(cudaFree(C_dev));
    free(C_host);
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
```

下面这个核函数会产生一个比较低效的分支：

```C
__global__ void mathKernel1(float *c) {
	int tid = blockIdx.x* blockDim.x + threadIdx.x;
	
	float a = 0.0;
	float b = 0.0;

	if (tid % 2 == 0) {
		a = 100.0f;
	} else {
		b = 200.0f;
	}

	c[tid] = a + b;
}
```

在这种情况下，我们假设只配置了一个一维线程块，大小为 x=64。此时，只有两个线程束（warp）。在每个线程束内，**奇数索引的线程（即 threadIdx.x 为奇数的线程）将执行 else 分支，而偶数索引的线程则会执行 if 分支**。这种情况导致了线程执行路径的严重分化。

线程束的执行逻辑如下： 

| **线程 ID (`tid`)** | **`tid % 2`** | **`a` 值** | **`b` 值** | **`c[tid]` 值** |
|---------------------|---------------|------------|------------|------------------|
| 0                  | 0             | 100.0      | 0.0        | 100.0           |
| 1                  | 1             | 0.0        | 200.0      | 200.0           |
| 2                  | 0             | 100.0      | 0.0        | 100.0           |
| 3                  | 1             | 0.0        | 200.0      | 200.0           |
| 4                  | 0             | 100.0      | 0.0        | 100.0           |
| 5                  | 1             | 0.0        | 200.0      | 200.0           |
| ...                | ...           | ...        | ...        | ...              |

然而，如果我们采用另一种方法来获取相同但顺序错乱的结果（其实这个顺序并不重要，因为我们可以在后期进行调整），那么下面代码就会很高效，即条件 `(tid / warpSize) % 2 == 0` 。这样可以**确保分支粒度是线程束大小的倍数**。具体而言，偶数编号的线程会执行 `if` 语句，而奇数编号的线程则执行 `else` 语句。通过这种方式，我们可以有效降低线程的分支分化，提高程序的整体执行效率。

```C
__global__ void mathKernel2(float *c) {
	int tid = blockIdx.x* blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;

	if ((tid/warpSize) % 2 == 0) {
		a = 100.0f;
	} else {
		b = 200.0f;
	}

	c[tid] = a + b;
}
```

-   在第一个线程束内，线程编号 tid 从 0 到 31，因此 `tid / warpSize` 的值都等于 0，因此所有线程都将执行 `if` 语句。
-   在第二个线程束内，线程编号 tid 从 32 到 63，`tid / warpSize` 的值都等于 1，因此所有线程将执行 `else `语句。在这种情况下，线程束内没有出现分支分化，从而使得程序的执行效率得以提高。
  
线程束的执行逻辑如下：

| **线程 ID (`tid`)** | **Warp ID (`tid / warpSize`)** | **Warp ID % 2** | **`a` 值** | **`b` 值** | **`c[tid]` 值** |
| ------------------- | ------------------------------ | --------------- | ---------- | ---------- | --------------- |
| 0                   | 0                              | 0               | 100.0      | 0.0        | 100.0           |
| 1                   | 0                              | 0               | 100.0      | 0.0        | 100.0           |
| 2                   | 0                              | 0               | 100.0      | 0.0        | 100.0           |
| ...                 | ...                            | ...             | ...        | ...        | ...             |
| 31                  | 0                              | 0               | 100.0      | 0.0        | 100.0           |
| 32                  | 1                              | 1               | 0.0        | 200.0      | 200.0           |
| 33                  | 1                              | 1               | 0.0        | 200.0      | 200.0           |
| 34                  | 1                              | 1               | 0.0        | 200.0      | 200.0           |
| ...                 | ...                            | ...             | ...        | ...        | ...             |
| 63                  | 1                              | 1               | 0.0        | 200.0      | 200.0           |

执行结果：

```shell
./simpleDivergence using Device 0: NVIDIA GeForce GTX 1060 6GB
Data size: 64
Execution Configure (block 64 grid 1)
warmup                  <<<1, 64>>>     elapsed 0.000034 sec 
mathKernel1             <<<1, 64>>>     elapsed 0.000009 sec 
mathKernel2             <<<1, 64>>>     elapsed 0.000008 sec 
mathKernel3             <<<1, 64>>>     elapsed 0.000007 sec 
```

代码中的 warmup 部分是为了提前启动 GPU，因为第一次启动 GPU 时的性能通常低于后续运行的速度。具体原因虽然尚不明确，但可以查阅 CUDA 相关技术文档以获取更多信息。为了更深入地分析程序的执行过程，我们可以使用 `nvprof` 工具。通过 `nvprof`，我们可以获取详细的性能数据，帮助我们了解程序的运行效率和潜在的性能瓶颈：

```shell
nvprof --metrics branch_efficiency ./simpleDivergence
```

然后得到下面这些参数。编写的 CMakeLists 禁用了分支预测功能，这样 kernel1 和 kernel3 的效率是相近的。即用 kernel3 的编写方式，会得到没有优化的结果如下：

```shell
==39332== NVPROF is profiling process 39332, command: ./simpleDivergence
./simpleDivergence using Device 0: NVIDIA GeForce GTX 1060 6GB
Data size: 64
Execution Configure (block 64 grid 1)
warmup                  <<<1, 64>>>     elapsed 0.006334 sec 
mathKernel1             <<<1, 64>>>     elapsed 0.003533 sec 
mathKernel2             <<<1, 64>>>     elapsed 0.003518 sec 
mathKernel3             <<<1, 64>>>     elapsed 0.003493 sec 
==39332== Profiling application: ./divergence
==39332== Profiling result:
==39332== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: mathKernel1(float*)
          1                         branch_efficiency                         Branch Efficiency      83.33%      83.33%      83.33%
    Kernel: mathKernel2(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
    Kernel: mathKernel3(float*)
          1                         branch_efficiency                         Branch Efficiency      83.33%      83.33%      83.33%
    Kernel: warmup(float*)
          1                         branch_efficiency                         Branch Efficiency     100.00%     100.00%     100.00%
```

分支效率值的计算是这样的：

$$
Branch Efficiency = \frac {Branches − DivergentBranches}{Branches}
$$

以下是 kernel3 编译器不会优化的代码：

```C
__global__ void mathKernel3(float *c) {
	int tid = blockIdx.x* blockDim.x + threadIdx.x;
	float a = 0.0;
	float b = 0.0;
	bool ipred = (tid % 2 == 0);

	if (ipred) {
		a = 100.0f;
	} else {
		b = 200.0f;
	}

	c[tid] = a + b;
}
```

考察一下事件计数器：

```shell
nvprof --events branch,divergent_branch ./simpleDivergence
```

```shell
==39513== NVPROF is profiling process 39513, command: ./simpleDivergence
./simpleDivergence using Device 0: NVIDIA GeForce GTX 1060 6GB
Data size: 64
Execution Configure (block 64 grid 1)
warmup                  <<<1, 64>>>     elapsed 0.004556 sec 
mathKernel1             <<<1, 64>>>     elapsed 0.003272 sec 
mathKernel2             <<<1, 64>>>     elapsed 0.004756 sec 
mathKernel3             <<<1, 64>>>     elapsed 0.003206 sec 
==39513== Profiling application: ./divergence
==39513== Profiling result:
==39513== Event result:
Invocations                                Event Name         Min         Max         Avg       Total
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: mathKernel1(float*)
          1                                    branch          12          12          12          12
          1                          divergent_branch           2           2           2           2
    Kernel: mathKernel2(float*)
          1                                    branch          11          11          11          11
          1                          divergent_branch           0           0           0           0
    Kernel: mathKernel3(float*)
          1                                    branch          12          12          12          12
          1                          divergent_branch           2           2           2           2
    Kernel: warmup(float*)
          1                                    branch          11          11          11          11
          1                          divergent_branch           0           0           0           0
```

nvcc 在 1 和 3 上优化有限，但是也超过了 50% 以上的利用率，`divergent_branch` 即线程分岔的数量。

#### 2.3. 资源分配 Resource Partitioning

我们前面提到，每个 SM 上执行的基本单位是线程束，这意味着单条指令会通过指令调度器广播给该线程束中所有线程，这些线程会在同一时刻执行相同的命令。尽管我们已经讨论了分支情况，但仍然有许多线程束尚未执行。那么，这些未执行的线程束应该如何分类呢？我将它们分为两类，尽管这种分类可能不是官方的说法。

首先，我们可以从线程束的角度来观察：一类是**已经激活的线程束**，这类线程束已经在 SM 上准备就绪，只是尚未轮到它们执行。在这种情况下，它们的状态被称为“阻塞”；另一类是**未激活的线程束**，尽管这些线程束已经分配到 SM，但尚未被激活执行。这样通过这种分类，我们能够更好地理解线程束在 GPU 上的行为。

而每个 SM 上有多少个线程束处于激活状态，取决于以下资源：

- 程序计数器
- 寄存器
- 共享内存

一旦线程束被激活并进入 SM，它会一直保持在该 SM 中，直到执行完成为止。每个 SM 具有一组 32 位寄存器，具体的寄存器数量取决于 GPU 的架构，而这些寄存器存储在寄存器文件中，并为每个线程进行分配。同时，每个 SM 还拥有固定数量的共享内存，这部分内存则在线程块之间进行分配。

**一个 SM 上可分配的线程块和线程束的数量，取决于 SM 中可用的寄存器和共享内存，以及内核所需的寄存器和共享内存大小。** 这实际上是一个平衡问题，就像一个固定大小的坑，其能容纳多少萝卜，取决于坑的大小以及萝卜的大小。相比于一个大坑，小坑可以容纳十个小萝卜或者两个大萝卜；同样的道理，SM 上的资源也遵循这一原则。当内核占用的资源较少时，能够激活更多的线程（这意味着线程束的数量也会增加）；相反，当资源需求较高时，活跃的线程数量就会减少。

关于寄存器资源的分配：

![allocate of register](/images/Professional%20CUDA%20C%20Programming/allocate%20of%20register.png)

关于共享内存资源的分配：

![allocate of shared memory](/images/Professional%20CUDA%20C%20Programming/allocate%20of%20shared%20memory.png)

上述内容主要集中在线程束上，从逻辑角度来看，线程块的可用资源分配同样会影响常驻线程块的数量。特别是在 SM 内的资源不足以处理一个完整的线程块时，程序就无法启动。因此，这种情况往往是我们需要关注的一个问题。

为了避免这种现象，我们需要仔细考虑核函数的设计，包括核函数的大小以及每个线程块中线程的数量。这些参数的配置直接影响到资源的有效利用。因此，**优化核函数的编写和合理设置线程块的规模，将有助于确保程序能够顺利启动并充分利用 GPU 的计算能力**。

以下是资源列表：

![Compute Capability3](/images/Professional%20CUDA%20C%20Programming/Compute%20Capability3.png)

当寄存器和共享内存分配给线程块后，该线程块便处于活跃状态，其内部包含的线程束被称为**活跃线程束**。这些活跃线程束根据执行状态可分为三类：

1.  **执行态线程束**（Selected Warp）：当前正在流处理器（SM）上执行的线程束
2.  **候选态线程束**（Eligible Warp）：已满足执行条件等待调度的线程束
3.  **阻塞态线程束**（Stalled Warp）：因资源未就绪暂时无法执行的线程束

线程束进入候选态需同时满足两个关键条件：

- 可用的 32 个CUDA计算单元（对应线程束的 32 个线程并行执行能力）
- 所需的执行资源全部准备就绪（如寄存器、共享内存）

以 Kepler 架构为例，其设计规范要求单个 SM 中活跃线程束总数不超过 64 个，同时每个时钟周期最多只能调度 4 个线程束执行。由于计算资源是在线程束之间分配的，且线程束的整个生命周期都处于 SM 内，因此线程束之间的上下文切换非常迅速。接下来，我们将介绍如何通过大量活跃线程束的切换来有效降低延迟。

#### 2.4. 延迟隐藏 Latency Hiding

延迟是什么？延迟就是当你让**计算机处理一个任务时所需的时间**。可以通过一个宏观的例子来说明这一概念。例如，当你将一个算法验证任务交给计算机时，计算机会让某个特定的计算单元完成这个任务，整个计算过程需要十分钟。在这十分钟内，你只能等待，直到计算完成才能开始下一个任务。在这段时间内，计算机的利用率可能并非达到 100%，即某些计算资源可能处于空闲状态。

你就想，能不能在这十分钟内同时运行相同的程序但数据集不同（这种情况在机器学习中并不陌生，大家通常会同时运行多个版本）。如果你继续给计算机添加任务，你可能会发现，在十分钟内，你已经插入了十个任务，而第一个任务仍在运行。当第一个任务完成时，如果你选择不再添加任务并等待后续任务执行，则总耗时为 20分钟，平均每个任务所需的时间为 $\frac{20}{10}=2$ 分钟/任务 。

然而还有一种情况，假设在第十分钟，第一个任务完成时你继续向计算机添加任务，那么这个任务的处理循环会持续进行。如果在二十分钟时你停止添加任务并等待所有任务完成，那么最终的平均任务时间将变为 $\frac{30}{20}=1.5$ 分钟/任务。如果你一直持续添加任务，理想情况下这种处理速度会趋近于 $lim_{n\to\infty}\frac{n+10}{n}=1$， 也就是**极限速度**每分钟处理一个任务，从而实现了隐藏了 9 分钟的延迟。

当然，这里还有另一个重要参数，即每十分钟你能添加多少个任务。如果每十分钟允许你添加 100 个任务，那么在二十分钟内你可以处理 100 个任务，此时每个任务的平均耗时为 $\frac{20}{100}=0.2$ 分钟/任务，三十分钟就是 $\frac{30}{200}=0.15$ 分钟/任务。如果你持续添加任务，理论上的极限速度将为 $lim_{n\to\infty}\frac{n+10}{n\times 10}=0.1$ 分钟/任务。

然而，这仍然是一个理想情况，有一个必须考虑的就是虽然你十分钟添加了100个任务，可是没准添加50个计算机就满载了，这样的话 极限速度只能是：因为尽管你可以在十分钟内添加 100 个任务，实际上可能在添加 50 个后，计算机就会满载，这时极限速度将降至 $lim_{n\to\infty}\frac{n+10}{n\times 5}=0.2$ 分钟/任务 了。

因此，**最优化的目标是充分利用硬件资源，尤其是核心计算部分的硬件，使其始终保持高负载状态**。当某些资源处于闲置状态时，整体利用率将显著降低，因此关键在于最大限度地提高每个单元的利用率，这与活跃线程束的数量直接相关。

线程调度器负责管理线程束的调度。当每个时刻都有可用的线程束供其调度时，便能够实现计算资源的完全利用。这种状态可以确保在其他常驻线程束中发布其他指令，从而降低每个指令的延迟，提升整体计算性能。通过优化线程束的调度，可以有效降低延迟，从而提高计算的总体效率。

与其他类型的编程相比，GPU 的延迟隐藏显得格外重要。指令延迟通常分为两类：

-   **算术指令延迟**：一个算术操作的开始与产生结果之间的时间。在这个时间段内，只有部分计算单元处于工作状态，而其他逻辑计算单元则处于空闲状态。
-   **内存指令延迟**：当计算单元需要访问内存时，它必须等待数据从内存加载到寄存器，这一过程的周期往往非常漫长。

具体而言：

-   **算术指令延迟**大约在 10 到 20 个时钟周期之间。
-   **内存指令延迟**则高达 400 到 800 个时钟周期。

下图就是阻塞线程束到候选态线程束的过程逻辑图：

![Warp Scheduler](/images/Professional%20CUDA%20C%20Programming/Warp%20Scheduler.png)

其中线程束 0 在经历了两段时间的阻塞后恢复到可调度模式，但值得注意的是，在这段等待时间中，SM 并没有处于闲置状态。

为了最小化延迟，至少需要多少个线程或线程束呢？**Little 法则**能够帮助我们进行计算: 

$$
\text{所需线程束} = \text{延迟} \times \text{吞吐量}
$$

> 注意带宽和吞吐量的区别，带宽一般指的是理论峰值，最大每个时钟周期能执行多少个指令，吞吐量是指实际操作过程中每分钟处理多少个指令。

这个可以想象成一个瀑布，像这样，绿箭头是线程束，只要线程束足够多，吞吐量是不会降低的：

![Throughput](/images/Professional%20CUDA%20C%20Programming/Throughput.png)

下面表格给出了 Fermi 和 Kepler 执行某个简单计算时需要的并行操作数：

![Full Arithmetic Utilization](/images/Professional%20CUDA%20C%20Programming/Full%20Arithmetic%20Utilization.png)

另外有两种方法可以提高并行：

-   **指令级并行（ILP）:** **在单个线程中包含多条独立的指令**，这些指令可以同时执行。通过充分利用不同的运算单元，ILP 能够帮助提升单个线程的执行效率，减少因指令依赖性造成的延迟。
-   **线程级并行（TLP）:** **在同一时间内并发执行多个线程**。通过启动大量并发线程，GPU 可以高效利用其多个计算核心，从而降低等待时间和整体延迟。TLP 特别适合那些可以独立执行且互不依赖的任务。

同时，内存延迟的隐藏与指令延迟隐藏的原理类似。内存延迟的隐藏依赖于并发内存读取操作。需要注意的是，**指令隐藏的主要目标是充分利用计算资源，而内存读取的延迟隐藏则目的是为充分利用内存带宽**。在内存延迟发生时，计算资源可能正被其他线程束占用，因此在计算内存读取延迟时，我们不考虑计算资源的活动情况。

可以将这两种延迟视为不同的方面，但它们遵循相同的原则。我们的最终目标是充分利用计算资源和内存带宽资源，从而实现理论上的最大效率。

同样，下表基于 Little 法则给出了最小化内存读取延迟所需的线程束数量。在此之前，需要进行一定的单位换算。机器的性能指标中，内存读取速度通常以 GB/s 为单位，而为了计算所需的字节数，我们需要将其转换为**每个时钟周期读取的字节数**。这可以通过将内存带宽除以其工作频率来实现，例如，对 NVIDIA 2070 显卡，其内存带宽为 144 GB/s，转换为每个时钟周期的读取字节数可以表示为：

$$\frac{144GB/s}{1.566GHz}=92 B/t$$

我们就得到了单位时间内的内存带宽，从而可以更准确地评估所需的线程束数量，以优化内存读取的延迟。

![Full Memory Utilization](/images/Professional%20CUDA%20C%20Programming/Full%20Memory%20Utilization.png)

需要说明的是，提到的内存带宽并不是针对单个 SM，而是整个 GPU 设备的带宽。具体来说，Fermi 架构需要并行读取约 74 条数据才能充分利用 GPU 的内存带宽。如果每个线程读取 4 个字节，我们大约需要 18500 个线程，也就是约 579 个线程束，以达到这个峰值。因此，**延迟的隐藏效果取决于活动线程束的数量，数量越多，延迟隐藏效果越好**。然而，线程束的数量又受到上面说的资源的限制。因此，我们需要寻找最优的执行配置，以实现最佳的延迟隐藏。

那么，如何确定一个线程束数量的下界，使得在这个数量之上，SM 的延迟能够得到充分的隐藏呢？其实，这个公式相对简单且易于理解：即 **SM 的计算核心数量乘以单条指令的延迟**。例如，如果有 32 个单精度浮点计算器，而且每次计算的延迟为 20 个时钟周期，那么我们至少需要 $32 \times 20 = 640$，即至少需要 640 个线程来确保设备始终处于忙碌状态。通过计算，我们可以设定合理的线程束数量，从而有效地降低延迟影响，优化 GPU 的性能。

#### 2.5. 占用率 Occupancy

占用率是一个SM种活跃的线程束的数量，占SM最大支持线程束数量的比。前面写的程序`chapter02/checkDeviceInfor.cu` 中添加几个成员的查询就可以帮我们找到这个值（`simpleDeviceQuery.cu`）。

```C
#include <stdio.h>
#include <cuda_runtime.h>
#include "../common/common.h"

/*
 * Fetches basic information on the first device in the current CUDA platform,
 * including number of SMs, bytes of constant memory, bytes of shared memory per
 * block, etc.
 */

int main(int argc, char* argv[]) {
    int            iDev = 0;
    cudaDeviceProp iProp;
    CHECK(cudaGetDeviceProperties(&iProp, iDev));

    printf("Device %d: %s\n", iDev, iProp.name);
    printf("  Number of multiprocessors:                     %d\n",
           iProp.multiProcessorCount);
    printf("  Total amount of constant memory:               %4.2f KB\n",
           iProp.totalConstMem / 1024.0);
    printf("  Total amount of shared memory per block:       %4.2f KB\n",
           iProp.sharedMemPerBlock / 1024.0);
    printf("  Total number of registers available per block: %d\n",
           iProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           iProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n",
           iProp.maxThreadsPerBlock);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           iProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of warps per multiprocessor:    %d\n",
           iProp.maxThreadsPerMultiProcessor / 32);
    return EXIT_SUCCESS;
}
```

输出如下：

```shell
  Device 0: NVIDIA GeForce RTX 3090
  Number of multiprocessors:                     82
  Total amount of constant memory:               64.00 KB
  Total amount of shared memory per block:       48.00 KB
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum number of threads per multiprocessor:  1536
  Maximum number of warps per multiprocessor:    48
```

占用率是**每个 SM 中活跃的线程束占最大线程束数量的比值**：

$$
\text{占用率} = \frac{\text{活动线程束数量}}{\text{最大线程束数量}}
$$

CUDA 工具包中提供一个叫做 CUDA 占用率计算器的电子表格，填上相关数据可以帮你自动计算网格参数：

![Occupancy Calculator](/images/Professional%20CUDA%20C%20Programming/Occupancy%20Calculator.png)

上面我们已明确内核使用的寄存器数量会影响每个流式处理器（SM）内线程束的数量。同时，`nvcc` 编译选项也允许我们手动控制寄存器的使用。

为了提高占用率，还可以通过调整每个线程块内的线程数量来达到这一目的，但必须合理控制，避免极端情况：

-   **小的线程块**：如果每个线程块中的线程数量过少，可能在所有资源未被充分利用的情况下便达到了线程束的最大限制。
-   **大的线程块**：如果每个线程块内的线程数量过多，则可能导致每个 SM 中每个线程可用的硬件资源减少。

为了使程序能够适应当前和未来的设备，建议最好遵循以下指导原则：

-   确保每个块中的线程数是**线程束大小（32）的倍数**。
-   避免使用过小的线程块，**每个块至少应包含 128 或 256 个线程**。
-   根据内核的资源需求灵活调整块的大小。
-   确保**线程块的数量远远多于 SM 的数量**，以确保设备中有足够的并行度。
-   通过实验确定最佳的执行配置和资源使用策略。

#### 2.6. 同步 Synchronization

在并发编程中，Synchronization（同步）机制至关重要，例如在 `pthread` 中使用的锁以及在 `OpenMP` 中的同步机制，其主要目的是为了防止内存竞争。在 CUDA 中，我们的讨论将集中在两种主要的同步机制上：

-   **线程块内同步**：CUDA 提供了线程块内的同步机制，允许同一个线程块中的线程协同工作。使用 `__syncthreads()` 函数，线程可以在需要时等待其他线程完成其操作，确保数据的一致性和正确性。这种机制特别适用于需要共享数据的场景。
-   **系统级别同步**：在 CUDA 编程中，系统级别的同步较少使用，但在多块并行执行的情况下，可能需要通过管理主机和设备之间的数据传输来实现。此类同步通常需要使用 CUDA 提供的事件（Events）和流（Streams）机制，以确保不同的 GPU 计算任务按顺序执行或进行协调。

可以调用 CUDA API 实现线程同步：

```c
cudaError_t cudaDeviceSynchronize(void);
```

该函数会使主机端等待设备上的所有线程执行完毕，确保 GPU 上的所有任务都已经完成。

在块级别，同一个线程块内的线程可以通过调用以下函数来进行同步：

```C
__syncthread();
```

这个函数会使同一线程块内的所有线程在调用点处阻塞，直到该线程块内的所有线程都执行到该点为止。需要注意的是，**这个函数仅能在同一个线程块内的线程之间进行同步，不能跨块同步**。如果需要同步不同块内的线程，则必须等待整个核函数执行完成，通过控制程序在主机端和设备之间进行数据交换来实现。

**内存竞争是一种非常严重的错误，可能导致程序的不确定性和难以追踪的 bug**。编写并发程序时，一定要小心管理对共享数据的访问，确保在对数据进行读写操作时不会引发竞争条件。这是 CUDA 编程中常见的错误之一，需要保持警惕。

#### 2.7. 可扩展性 Scalability

可扩展性通常是相对于不同硬件设备而言的。当某个程序在设备 1 上执行时，所需时间为 T。当我们将该程序转移至设备 2，而设备 2 的资源量是设备 1 的两倍时，我们期望程序的运行时间减少至 T/2。这种性能提升的特性正是 CUDA 驱动程序所提供的功能之一。目前，NVIDIA 正积极致力于在这方面进行优化，具体情况如下图所示：

![Scalability](/images/Professional%20CUDA%20C%20Programming/Scalability.png)

### 3. 并行性表现 Exposing Parallelism

本节的主要内容是进一步理解线程束在硬件上执行的过程。结合前几节关于执行模型的学习，我们通过调整核函数的参数配置，观察核函数的执行速度，并分析硬件的利用率与实际性能。本节将重点研究核函数配置如何影响执行效率，即通过不同的网格和块配置，来获得不同的执行性能。

本节只用到下面的核函数:

```C
__global__ void sumMatrix(float * MatA, float * MatB, float * MatC, const int num_x, const int num_y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * num_x + row;

    if (row < num_x && col < num_y) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
```

没有任何优化的最简单的二维矩阵加法，代码在 `chapter03/sumMatrix2D.cu` 中。

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void sumMatrix2D_CPU(float* MatA, float* MatB, float* MatC, const int num_x, const int num_y) {
    float* a = MatA;
    float* b = MatB;
    float* c = MatC;

    for (int j = 0; j < num_y; j++) {
        for (int i = 0; i < num_x; i++) {
            c[i] = a[i] + b[i];
        }

        c += num_x;
        b += num_x;
        a += num_x;
    }
}

__global__ void sumMatrix(float* MatA, float* MatB, float* MatC, const int num_x, const int num_y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * num_x + row;

    if (row < num_x && col < num_y) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char** argv) {
    printf("strating...\n");
    int dev = 0;
    cudaSetDevice(dev);
    int row    = 1 << 12;    // 2^12, 16384
    int col    = 1 << 12;    // 2^12, 16384
    int sum    = row * col;
    int nBytes = sum * sizeof(float);

    // Malloc
    float* A_host     = (float*)malloc(nBytes);
    float* B_host     = (float*)malloc(nBytes);
    float* C_host     = (float*)malloc(nBytes);
    float* C_from_gpu = (float*)malloc(nBytes);

    // 为输入矩阵赋值
    initialData(A_host, sum);
    initialData(B_host, sum);

    // 输出矩阵，分配设备端内存
    float* A_dev = NULL;
    float* B_dev = NULL;
    float* C_dev = NULL;
    CHECK(cudaMalloc((void**)&A_dev, nBytes));
    CHECK(cudaMalloc((void**)&B_dev, nBytes));
    CHECK(cudaMalloc((void**)&C_dev, nBytes));

    // 将输入数据从主机端拷贝到设备端
    CHECK(cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_dev, B_host, nBytes, cudaMemcpyHostToDevice));

    int    dim_x  = argc > 2 ? atoi(argv[1]) : 32;
    int    dim_y  = argc > 2 ? atoi(argv[2]) : 32;
    double iStart = 0.0;
    double iElaps = 0.0;

    // cpu compute
    cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost);
    iStart = cpuSecond();
    sumMatrix2D_CPU(A_host, B_host, C_host, row, col);
    iElaps = cpuSecond() - iStart;
    printf("CPU Execution Time elapsed %f sec\n", iElaps);

    // 2d block and 2d grid
    dim3 blockDim_0(dim_x, dim_y);
    dim3 gridDim_0((row + blockDim_0.x - 1) / blockDim_0.x, (col + blockDim_0.y - 1) / blockDim_0.y);
    iStart = cpuSecond();
    sumMatrix<<<gridDim_0, blockDim_0>>>(A_dev, B_dev, C_dev, row, col);    // 调用 CUDA 核函数
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
           gridDim_0.x, gridDim_0.y, blockDim_0.x, blockDim_0.y, iElaps);
    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, sum);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);

    free(A_host);
    free(B_host);
    free(C_host);
    free(C_from_gpu);
    
    cudaDeviceReset();
    return 0;
}
```

这里用两个 $8192×8192$ 的矩阵相加来测试效率。注意一下这里的 GPU 内存，一个矩阵是 $2^{14}×2^{14}×2^2=2^{30}$ 字节 也就是 1G，三个矩阵就是 3G。 

#### 3.1. 用 nvprof 检测活跃的线程束 Checking Active Warps with nvprof

对比性能要控制变量，上面的代码只用两个变量，也就是块的x和y的大小，所以，调整x和y的大小来产生不同的效率，结果如下：

```shell
(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 32 32
CPU Execution Time elapsed 0.538640 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.090911 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 32 16
CPU Execution Time elapsed 0.548685 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.086876 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 16 32
CPU Execution Time elapsed 0.544791 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.056706 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 16 16
CPU Execution Time elapsed 0.548078 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.056472 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 16 8
CPU Execution Time elapsed 0.546093 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.086659 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 8 16
CPU Execution Time elapsed 0.545576 sec
GPU Execution configuration<<<(2048, 1024),(8, 16)>>> Time elapsed 0.056402 sec
```

汇总成表格:

|  gridDim   | blockDim | CPU Time (s) | GPU Time (s) |
|:----------:|:--------:|:------------:|:------------:|
|  512, 512  |  32, 32  |   0.538640   |   0.090911   |
| 512, 1024  |  32, 16  |   0.548685   |   0.086876   |
| 1024, 512  |  16, 32  |   0.544791   |   0.056706   |
| 1024, 1024 |  16, 16  |   0.548078   |   0.056472   |
| 1024, 2048 |  16, 8   |   0.546093   |   0.086659   |
| 2048, 1024 |  8, 16   |   0.545576   |   0.056402   |

当块大小超过硬件的极限，并没有报错，而是返回了错误值，这个值得注意。

另外，每个机器执行此代码效果可能定不一样，所以大家要根据自己的硬件分析数据。书上给出的 M2070 就和我们的结果不同，2070 的 (32,16) 效率最高，而我们的 (16, 16) 效率最高，毕竟架构不同，而且 CUDA 版本不同导致了优化后的机器码差异很大，所以我们还是来看看活跃线程束的情况，使用

```shell
nvprof --metrics achieved_occupancy ./sumMatrix2D 
```

得出结果

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 32 32 
==43939== NVPROF is profiling process 43939, command: ./sumMatrix2D 32 32
CPU Execution Time elapsed 0.550530 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.096127 sec
==43939== Profiling application: ./sumMatrix2D 32 32
==43939== Profiling result:
==43939== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.728469    0.728469    0.728469

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 32 16
==44053== NVPROF is profiling process 44053, command: ./sumMatrix2D 32 16
CPU Execution Time elapsed 0.551584 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.089149 sec
==44053== Profiling application: ./sumMatrix2D 32 16
==44053== Profiling result:
==44053== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.904511    0.904511    0.904511

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 16 32
==44187== NVPROF is profiling process 44187, command: ./sumMatrix2D 16 32
CPU Execution Time elapsed 0.547609 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.070035 sec
==44187== Profiling application: ./sumMatrix2D 16 32
==44187== Profiling result:
==44187== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.817224    0.817224    0.817224

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 16 16
==44285== NVPROF is profiling process 44285, command: ./sumMatrix2D 16 16
CPU Execution Time elapsed 0.550066 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.062846 sec
==44285== Profiling application: ./sumMatrix2D 16 16
==44285== Profiling result:
==44285== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.885973    0.885973    0.885973

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 16 8
==44394== NVPROF is profiling process 44394, command: ./sumMatrix2D 16 8
CPU Execution Time elapsed 0.548652 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.092749 sec
==44394== Profiling application: ./sumMatrix2D 16 8
==44394== Profiling result:
==44394== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.968459    0.968459    0.968459

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 8 16
==44547== NVPROF is profiling process 44547, command: ./sumMatrix2D 8 16
CPU Execution Time elapsed 0.549166 sec
GPU Execution configuration<<<(2048, 1024),(8, 16)>>> Time elapsed 0.062462 sec
==44547== Profiling application: ./sumMatrix2D 8 16
==44547== Profiling result:
==44547== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.870483    0.870483    0.870483
```

汇总成表格:

|  gridDim   | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy |
|:----------:|:--------:|:------------:|:------------:|:------------------:|
|  512, 512  |  32, 32  |   0.550530   |   0.096127   |      0.728469      |
| 512, 1024  |  32, 16  |   0.551584   |   0.089149   |      0.904511      |
| 1024, 512  |  16, 32  |   0.547609   |   0.070035   |      0.817224      |
| 1024, 1024 |  16, 16  |   0.550066   |   0.062846   |      0.885973      |
| 1024, 2048 |  16, 8   |   0.548652   |   0.092749   |      0.968459      |
| 2048, 1024 |  8, 16   |   0.549166   |   0.062462   |      0.870483      |

活跃线程束比例的定义是：在每个周期内，活跃的线程束的平均数量与一个流式处理器（SM）支持的最大线程束数量之间的比值。这个比例用于衡量线程束的活跃程度，进而反映出程序在硬件资源利用上的效率。可见**活跃线程束比例高的未必执行速度快**，但实际上从原理出发，应该是利用率越高效率越高，但是还受到其他因素制约。

#### 3.2. 用 nvprof 检测内存操作 Checking Active Warps with nvprof

下面我们继续用 nvprof 来看看内存利用率如何

```C
nvprof --metrics gld_throughput ./sumMatrix2D
```

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sumMatrix2D 32 32 
==44801== NVPROF is profiling process 44801, command: ./sumMatrix2D 32 32
CPU Execution Time elapsed 0.544097 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.273369 sec
==44801== Profiling application: ./sumMatrix2D 32 32
==44801== Profiling result:
==44801== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  61.836GB/s  61.836GB/s  61.836GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sumMatrix2D 32 16
==44878== NVPROF is profiling process 44878, command: ./sumMatrix2D 32 16
CPU Execution Time elapsed 0.545615 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.247466 sec
==44878== Profiling application: ./sumMatrix2D 32 16
==44878== Profiling result:
==44878== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  68.650GB/s  68.650GB/s  68.650GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sumMatrix2D 16 32
==44973== NVPROF is profiling process 44973, command: ./sumMatrix2D 16 32
CPU Execution Time elapsed 0.553040 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.244212 sec
==44973== Profiling application: ./sumMatrix2D 16 32
==44973== Profiling result:
==44973== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  34.835GB/s  34.835GB/s  34.835GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sumMatrix2D 16 16
==45123== NVPROF is profiling process 45123, command: ./sumMatrix2D 16 16
CPU Execution Time elapsed 0.545451 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.240271 sec
==45123== Profiling application: ./sumMatrix2D 16 16
==45123== Profiling result:
==45123== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  35.409GB/s  35.409GB/s  35.409GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sumMatrix2D 16 8
==45182== NVPROF is profiling process 45182, command: ./sumMatrix2D 16 8
CPU Execution Time elapsed 0.543101 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.246472 sec
==45182== Profiling application: ./sumMatrix2D 16 8
==45182== Profiling result:
==45182== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  34.444GB/s  34.444GB/s  34.444GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sumMatrix2D 8 16
==45295== NVPROF is profiling process 45295, command: ./sumMatrix2D 8 16
CPU Execution Time elapsed 0.545891 sec
GPU Execution configuration<<<(2048, 1024),(8, 16)>>> Time elapsed 0.240333 sec
==45295== Profiling application: ./sumMatrix2D 8 16
==45295== Profiling result:
==45295== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  17.701GB/s  17.701GB/s  17.701GB/s
```

汇总成表格:

|  gridDim   | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy | GLD Throughput (GB/s) |
|:----------:|:--------:|:------------:|:------------:|:------------------:|:---------------------:|
|  512, 512  |  32, 32  |   0.544097   |   0.273369   |      0.728469      |        61.836         |
| 512, 1024  |  32, 16  |   0.545615   |   0.247466   |      0.904511      |        68.650         |
| 1024, 512  |  16, 32  |   0.553040   |   0.244212   |      0.817224      |        34.835         |
| 1024, 1024 |  16, 16  |   0.545451   |   0.240271   |      0.885973      |        35.409         |
| 1024, 2048 |  16, 8   |   0.543101   |   0.246472   |      0.968459      |        34.444         |
| 2048, 1024 |  8, 16   |   0.545891   |   0.240333   |      0.870483      |        17.701         |

可以看出综合第二种配置的线程束吞吐量最大。所以可见吞吐量和线程束活跃比例一起都对最终的效率有影响。

接下来，我们来讨论全局加载效率。全局加载效率的定义是：**被请求的全局加载吞吐量与所需全局加载吞吐量之间的比值**。换句话说，它反映了应用程序在执行加载操作时对设备内存带宽的利用程度。需要注意的是，吞吐量和全局加载效率是两个不同的概念，这一点在前面的内容中已经进行了详细解释。

```C
nvprof --metrics gld_efficiency ./sumMatrix2D
```

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sumMatrix2D 32 32
==45602== NVPROF is profiling process 45602, command: ./sumMatrix2D 32 32
CPU Execution Time elapsed 0.544926 sec
==45602== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45602== Profiling application: ./sumMatrix2D 32 32Time elapsed 1.298604 sec
==45602== Profiling result:
==45602== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sumMatrix2D 32 16
==45728== NVPROF is profiling process 45728, command: ./sumMatrix2D 32 16
CPU Execution Time elapsed 0.546795 sec
==45728== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45728== Profiling application: ./sumMatrix2D 32 16 Time elapsed 1.258507 sec
==45728== Profiling result:
==45728== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sumMatrix2D 16 32
==45829== NVPROF is profiling process 45829, command: ./sumMatrix2D 16 32
CPU Execution Time elapsed 0.549460 sec
==45829== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45829== Profiling application: ./sumMatrix2D 16 32 Time elapsed 1.238372 sec
==45829== Profiling result:
==45829== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sumMatrix2D 16 16
==45926== NVPROF is profiling process 45926, command: ./sumMatrix2D 16 16
CPU Execution Time elapsed 0.548614 sec
==45926== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45926== Profiling application: ./sumMatrix2D 16 16> Time elapsed 1.219676 sec
==45926== Profiling result:
==45926== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sumMatrix2D 16 8
==46017== NVPROF is profiling process 46017, command: ./sumMatrix2D 16 8
CPU Execution Time elapsed 0.548084 sec
==46017== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==46017== Profiling application: ./sumMatrix2D 16 8> Time elapsed 1.277124 sec
==46017== Profiling result:
==46017== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sumMatrix2D 8 16
==46086== NVPROF is profiling process 46086, command: ./sumMatrix2D 8 16
CPU Execution Time elapsed 0.545527 sec
==46086== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==46086== Profiling application: ./sumMatrix2D 8 16> Time elapsed 1.219265 sec
==46086== Profiling result:
==46086== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      50.00%      50.00%      50.00%
```

汇总成表格:

|   gridDim    | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy | GLD Throughput (GB/s) | GLD Efficiency |
|:------------:|:--------:|:------------:|:------------:|:------------------:|:---------------------:| -------------- |
|  (512, 512)  | (32, 32) |   0.544097   |   0.273369   |      0.728469      |        61.836         | 12.50%         |
| (512, 1024)  | (32, 16) |   0.545615   |   0.247466   |      0.904511      |        68.650         | 12.50%         |
| (1024, 512)  | (16, 32) |   0.553040   |   0.244212   |      0.817224      |        34.835         | 25.00%         |
| (1024, 1024) | (16, 16) |   0.545451   |   0.240271   |      0.885973      |        35.409         | 25.00%         |
| (1024, 2048) | (16, 8)  |   0.543101   |   0.246472   |      0.968459      |        34.444         | 25.00%         |
| (2048, 1024) | (8, 16)  |   0.545891   |   0.240333   |      0.870483      |        17.701         | 50.00%         |

可见，如果线程块中内层的维度（blockDim.x）过小，**小于线程束会影响加载效率**。有效加载效率是指在全部的内存请求中（当前在总线上传递的数据）有多少是我们要用于计算的。

#### 3.3. 增大并行性 Exposing More Parallelism

线程块中内层的维度（blockDim.x）过小是否对现在的设备还有影响，我们来看下面的试验：

```shell
(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 64 2
CPU Execution Time elapsed 0.544023 sec
GPU Execution configuration<<<(256, 8192),(64, 2)>>> Time elapsed 0.356677 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 64 4
CPU Execution Time elapsed 0.544404 sec
GPU Execution configuration<<<(256, 4096),(64, 4)>>> Time elapsed 0.174845 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 64 8
CPU Execution Time elapsed 0.544168 sec
GPU Execution configuration<<<(256, 2048),(64, 8)>>> Time elapsed 0.091977 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 128 2
CPU Execution Time elapsed 0.545258 sec
GPU Execution configuration<<<(128, 8192),(128, 2)>>> Time elapsed 0.355204 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 128 4
CPU Execution Time elapsed 0.547236 sec
GPU Execution configuration<<<(128, 4096),(128, 4)>>> Time elapsed 0.176689 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 128 8
CPU Execution Time elapsed 0.545464 sec
GPU Execution configuration<<<(128, 2048),(128, 8)>>> Time elapsed 0.089984 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 256 2
CPU Execution Time elapsed 0.545916 sec
GPU Execution configuration<<<(64, 8192),(256, 2)>>> Time elapsed 0.363761 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 256 4
CPU Execution Time elapsed 0.548850 sec
GPU Execution configuration<<<(64, 4096),(256, 4)>>> Time elapsed 0.190659 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sumMatrix2D 256 8
CPU Execution Time elapsed 0.547406 sec
GPU Execution configuration<<<(64, 2048),(256, 8)>>> Time elapsed 0.000030 sec
```

汇总成表格:

|   gridDim   | blockDim | CPU Time (s) | GPU Time (s) |
|:-----------:|:--------:|:------------:|:------------:|
| (256, 8192) | (64, 2)  |   0.544023   |   0.356677   |
| (256, 4096) | (64, 4)  |   0.544404   |   0.174845   |
| (256, 2048) | (64, 8)  |   0.544168   |   0.091977   |
| (128, 8192) | (128, 2) |   0.545258   |   0.355204   |
| (128, 4096) | (128, 4) |   0.547236   |   0.176689   |
| (128, 2048) | (128, 8) |   0.545464   |   0.089984   |
| (64, 8192)  | (256, 2) |   0.545916   |   0.363761   |
| (64, 4096)  | (256, 4) |   0.548850   |   0.190659   |
| (64, 2048)  | (256, 8) |   0.547406   |     fail     |

通过这个表我们发现，**块最小的反而获得最低的效率，即数据量大可能会影响结果**。当数据量较大时，影响执行时间的因素可能会发生变化。以下是我们可以观察到的一些具体结论：

-   尽管 (64, 4) 和 (128, 2) 的块大小相同，但它们的执行效率却有所不同，这说明内层线程块的尺寸会影响整体效率。
-   最后一个块的参数是无效的，因为所有线程的总数超过了 GPU 的最大限制（1024 个线程）。
-   尽管 (64, 2) 的线程块是最小的，它启动的线程块数量却是最多的，但其速度并不是最快的。
-   综合考虑线程块的大小和数量，(128, 8) 的配置实现了最快的执行速度。

调整块的尺寸，还是为了增加并行性，或者说增加活跃的线程束，看看线程束的活跃比例：

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03#  nvprof --metrics achieved_occupancy ./sumMatrix2D 64 2
==47210== NVPROF is profiling process 47210, command: ./sumMatrix2D 64 2
CPU Execution Time elapsed 0.549154 sec
GPU Execution configuration<<<(256, 8192),(64, 2)>>> Time elapsed 0.363687 sec
==47210== Profiling application: ./sumMatrix2D 64 2
==47210== Profiling result:
==47210== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.941718    0.941718    0.941718

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 64 4
==47520== NVPROF is profiling process 47520, command: ./sumMatrix2D 64 4
CPU Execution Time elapsed 0.554265 sec
GPU Execution configuration<<<(256, 4096),(64, 4)>>> Time elapsed 0.182942 sec
==47520== Profiling application: ./sumMatrix2D 64 4
==47520== Profiling result:
==47520== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.939658    0.939658    0.939658

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 64 8
==47609== NVPROF is profiling process 47609, command: ./sumMatrix2D 64 8
CPU Execution Time elapsed 0.552905 sec
GPU Execution configuration<<<(256, 2048),(64, 8)>>> Time elapsed 0.100848 sec
==47609== Profiling application: ./sumMatrix2D 64 8
==47609== Profiling result:
==47609== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.912401    0.912401    0.912401

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 128 2
==47706== NVPROF is profiling process 47706, command: ./sumMatrix2D 128 2
CPU Execution Time elapsed 0.554928 sec
GPU Execution configuration<<<(128, 8192),(128, 2)>>> Time elapsed 0.361216 sec
==47706== Profiling application: ./sumMatrix2D 128 2
==47706== Profiling result:
==47706== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.842183    0.842183    0.842183

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 128 4
==47822== NVPROF is profiling process 47822, command: ./sumMatrix2D 128 4
CPU Execution Time elapsed 0.555749 sec
GPU Execution configuration<<<(128, 4096),(128, 4)>>> Time elapsed 0.182397 sec
==47822== Profiling application: ./sumMatrix2D 128 4
==47822== Profiling result:
==47822== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.833157    0.833157    0.833157

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 128 8
==47928== NVPROF is profiling process 47928, command: ./sumMatrix2D 128 8
CPU Execution Time elapsed 0.550801 sec
GPU Execution configuration<<<(128, 2048),(128, 8)>>> Time elapsed 0.099784 sec
==47928== Profiling application: ./sumMatrix2D 128 8
==47928== Profiling result:
==47928== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.732285    0.732285    0.732285

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 256 2
==48042== NVPROF is profiling process 48042, command: ./sumMatrix2D 256 2
CPU Execution Time elapsed 0.550500 sec
GPU Execution configuration<<<(64, 8192),(256, 2)>>> Time elapsed 0.369576 sec
==48042== Profiling application: ./sumMatrix2D 256 2
==48042== Profiling result:
==48042== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.804247    0.804247    0.804247

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 256 4
==48122== NVPROF is profiling process 48122, command: ./sumMatrix2D 256 4
CPU Execution Time elapsed 0.538097 sec
GPU Execution configuration<<<(64, 4096),(256, 4)>>> Time elapsed 0.197963 sec
==48122== Profiling application: ./sumMatrix2D 256 4
==48122== Profiling result:
==48122== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.791321    0.791321    0.791321

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sumMatrix2D 256 8
==48214== NVPROF is profiling process 48214, command: ./sumMatrix2D 256 8
CPU Execution Time elapsed 0.549278 sec
GPU Execution configuration<<<(64, 2048),(256, 8)>>> Time elapsed 0.000024 sec
==48214== Profiling application: ./sumMatrix2D 256 8
==48214== Profiling result:
No events/metrics were profiled.
```

汇总成表格:

|   gridDim   | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy |
|:-----------:|:--------:|:------------:|:------------:|:------------------:|
| (256, 8192) | (64, 2)  |   0.549154   |   0.363687   |      0.941718      |
| (256, 4096) | (64, 4)  |   0.554265   |   0.182942   |      0.939658      |
| (256, 2048) | (64, 8)  |   0.552905   |   0.100848   |      0.912401      |
| (128, 8192) | (128, 2) |   0.554928   |   0.361216   |      0.842183      |
| (128, 4096) | (128, 4) |   0.555749   |   0.182397   |      0.833157      |
| (128, 2048) | (128, 8) |   0.550801   |   0.099784   |      0.732285      |
| (64, 8192)  | (256, 2) |   0.550500   |   0.369576   |      0.804247      |
| (64, 4096)  | (256, 4) |   0.538097   |   0.197963   |      0.791321      |
| (64, 2048)  | (256, 8) |   0.549278   |     fail     |        Nan         |

可以明显看出，**最高的利用率并不一定对应最优的效率**。没有单一因素能够直接决定最终的效率，实际上，多个因素共同作用才会影响到最终结果。这是“多因一效”的一个典型例子。因此，在进行优化时，我们应该优先确保测试时间的准确性、客观性和稳定性。

-   在大多数情况下，单一指标无法实现最优性能的优化。
-   整体性能与内核代码的本质直接相关（核函数的设计才是关键）。
-   应该在指标和性能之间寻找一个合理的平衡点。
-   从不同角度探索指标的平衡，以最大化系统的效率。
-   网格和块的尺寸为性能调节提供了良好的起点。

总而言之，**使用 CUDA 的主要目的是为了实现高效，而研究和优化这些指标是提高效率的快速途径**（当然，内核算法的改进潜力更大）。

### 4. 避免分支分化 Avoiding Branch Divergence

#### 4.1. 并行规约问题 The Parallel Reduction Problem

在串行编程中，一个非常常见的问题是**将一组大量数字通过计算变成一个单一的结果，例如求和或乘积**。在满足以下两个条件的情况下，我们可以使用并行归约的方法来处理这些计算：

-   结合性
-   交换性

加法和乘法都符合交换律和结合律，因此对于所有具有这两个性质的计算，都可以应用归约方法。归约是一种常见的计算方式（无论是串行还是并行），其过程是每次迭代都采用相同的计算方法，从一组多个数据最终得到一个数（即归约）。归约的基本步骤如下：

1.  **将输入向量分割成更小的数据块**。
2.  **每个线程计算一个数据块的部分和**。
3.  **将所有数据块的部分和再汇总以得到最终结果**。

数据分块的设计确保每个线程块可以处理一个数据块。每个线程负责处理更小的部分，因此一个线程块能够处理一个较大的数据块，为整个数据集的处理提供了灵活性。最终，所有线程块得到的结果将在 CPU 上进行相加，以获得最终结果。

在归约问题中，最常见的加法计算方法是将向量的数据成对分组，使用不同的线程计算每一对元素的和。得到的结果将作为新的输入，再次分成对进行迭代，直到最终只剩下一个元素。成对划分的常用方法有以下两种：

1. **相邻配对：** 元素与他们相邻的元素配对

![Neighbored pair](/images/Professional%20CUDA%20C%20Programming/Neighbored%20pair.png)

2. **交错配对：** 元素与一定距离的元素配对

![Interleaved pair](/images/Professional%20CUDA%20C%20Programming/Interleaved%20pair.png)

在图中清晰地展示了这两种方式的实现方法，接下来，我们将提供对应的代码，首先是 CPU 版本的交错配对归约计算的实现代码：

```C
int recursiveReduce(int *data, int const size) {
	// terminate check
	if (size == 1) return data[0];
	// renew the stride
	int const stride = size / 2;
	
    if (size % 2 == 1) {
		for (int i = 0; i < stride; i++) {
			data[i] += data[i + stride];
		}
		data[0] += data[size - 1];
	} else {
		for (int i = 0; i < stride; i++) {
			data[i] += data[i + stride];
		}
	}
	// call
	return recursiveReduce(data, stride);
}
```

与书中提供的代码有些不同，因为书中代码并未考虑数组长度为非 2 的整数幂次的情况。因此，添加了对奇数数组中最后一个无人配对元素的处理。这个加法运算可以替换为任何满足结合律和交换律的操作，例如乘法或寻找最大值等。通过不同的配对方式和数据组织形式，可以更全面地评估CUDA的执行效率。

#### 4.2. 并行规约中的分化 Divergence in Parallel Reduction

**线程束分化**已经明确说明了，有判断条件的地方就会产生分支，比如 if 和 for 这类关键词。如下图表示，对相邻元素配对进行内核实现的流程描述：

![Parallel Reduction](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction.png)

-   **第一步：** 将数组分块，每一块仅包含部分数据，如上图所示（图中数据较少，但我们假设每块上只有这些数据）。我们假设这是一个线程块的全部数据。
-   **第二步：** 每个线程执行的任务标示在橙色圆圈内。可以看到，线程 `threadIdx.x = 0` 参与了三次计算，而奇数编号的线程则处于等待状态，没有进行任何计算。然而，正如在1.1.4 中所讨论的，这些线程虽然未执行任务，但仍不能执行其他指令。线程编号为 4 的线程进行了两次计算，而线程编号为2和6则各进行了一次计算。
-   **第三步：** 最后，将所有线程块的计算结果相加，得到最终结果。

这个计算划分体现了最简单的并行规约算法，完美遵循我们上面提到的三步走策略。在每次进行一轮计算时（如黄色框所示，这些操作是并行执行的），部分全局内存会被修改，但只有一部分数据会被替换，而未被替换的数据在后续的计算中不会再被使用到。例如，蓝色框中标注的内存仅被读取了一次，之后就不再被处理了。

接下来是相关的核函数代码：

```C
__global__ void reduceNeighbored(int * g_idata,int * g_odata,unsigned int n) {
	//set thread ID
	unsigned int tid = threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the 
	int *idata = g_idata + blockIdx.x * blockDim.x;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if ((tid % (2 * stride)) == 0) {
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}
```

这里面唯一要注意的地方就是同步指令

```C
__syncthreads();
```

原因可以从图中找到：虽然我们的每一轮操作都是并行进行的，但并不保证所有线程能够同时完成任务。因此，执行快的线程需要等待执行慢慢的线程，这样可以避免块内线程对内存的竞争。被操作的两个对象之间的距离称为**跨度**，也就是变量 `stride`。完整的执行逻辑如下：

![stride](/images/Professional%20CUDA%20C%20Programming/stride.png)

以下是具体的计算过程：

1. **第一轮** ：步长 `stride = 1`，每个线程 `tid` 检查 `(tid % (2 * stride)) == 0`，即 `tid % 2 == 0`，只有偶数索引的线程会执行加法操作：
   -   `data[0] += data[1]` → `data[0] = 3 + 1 = 4`
   -   `data[2] += data[3]` → `data[2] = 7 + 0 = 7`
   -   `data[4] += data[5]` → `data[4] = 4 + 1 = 5`
   -   `data[6] += data[7]` → `data[6] = 6 + 3 = 9`
2. **第二轮** ：步长 `stride = 2`，每个线程 `tid` 检查 `(tid % (2 * stride)) == 0`，即 `tid % 4 == 0`，只有索引为 0 和 4 的线程会执行加法操作：
   -   `data[0] += data[2]` → `data[0] = 4 + 7 = 11`
   -   `data[4] += data[6]` → `data[4] = 5 + 9 = 14`
3. **第三轮** ：步长 `stride = 4`，只有索引为 0 的线程会执行加法操作：
   -   `data[0] += data[4]` → `data[0] = 11 + 14 = 25`

注意主机端和设备端的分界，注意设备端的数据分块，完整代码在 `chapter03/reduceInteger.cu`：

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

int recursiveReduce(int* data, int const size) {
    // terminate check
    if (size == 1)
        return data[0];
    // renew the stride
    int const stride = size / 2;

    if (size % 2 == 1) {
        for (int i = 0; i < stride; i++) {
            data[i] += data[i + stride];
        }
        data[0] += data[size - 1];
    } else {
        for (int i = 0; i < stride; i++) {
            data[i] += data[i + stride];
        }
    }
    // call
    return recursiveReduce(data, stride);
}

__global__ void warmup(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int* idata = g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighbored(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int* idata = g_idata + blockIdx.x * blockDim.x;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredLess(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned     idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local point of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;
    if (idx > n)
        return;
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    // write result for this block to global men
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int* g_idata, int* g_odata, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned     idx = blockIdx.x * blockDim.x + threadIdx.x;
    // convert global data pointer to the local point of this block
    int* idata = g_idata + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // write result for this block to global men
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);

    int size = 1 << 24;
    printf("	with array size %d  ", size);

    // execution configuration
    int blocksize = 1024;

    if (argc > 1) {
        blocksize = atoi(argv[1]);
    }

    dim3 block(blocksize, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    printf("grid %d block %d \n", grid.x, block.x);

    // allocate host memory
    size_t bytes      = size * sizeof(int);
    int*   idata_host = (int*)malloc(bytes);
    int*   odata_host = (int*)malloc(grid.x * sizeof(int));
    int*   tmp        = (int*)malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; ++i) {
        idata_host[i] = (int)( rand() & 0xFF );
    }

    memcpy(tmp, idata_host, bytes);
    double iStart, iElaps;
    int    gpu_sum = 0;

    // device memory
    int* idata_dev = NULL;
    int* odata_dev = NULL;
    CHECK(cudaMalloc((void**)&idata_dev, bytes));
    CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

    // cpu reduction
    int cpu_sum = 0;
    iStart      = cpuSecond();
    // cpu_sum = recursiveReduce(tmp, size);
    for (int i = 0; i < size; i++)
        cpu_sum += tmp[i];
    printf("cpu sum:%d \n", cpu_sum);
    iElaps = cpuSecond() - iStart;
    printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // kernel 1:reduceNeighbored
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = cpuSecond();
    warmup<<<grid, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu warmup                 elapsed %lf ms gpu_sum: %d   <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // kernel 1:reduceNeighbored
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighbored       elapsed %lf ms gpu_sum: %d   <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // kernel 2:reduceNeighboredLess
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d   <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);

    // kernel 3:reduceInterleaved
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    
    iStart = cpuSecond();
    reduceInterleaved<<<grid, block>>>(idata_dev, odata_dev, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    
    for (int i = 0; i < grid.x; i++)
        gpu_sum += odata_host[i];
    printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d   <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);
    // free host memory

    free(idata_host);
    free(odata_host);
    cudaFree(idata_dev);
    cudaFree(odata_dev);

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    if (gpu_sum == cpu_sum) {
        printf("Test success!\n");
    }
    
    return EXIT_SUCCESS;
}
```

结果如下：

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                 elapsed 0.006651 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.002833 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.002634 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.001813 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.001667 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
```

下面是经过优化的两种写法。`warmup` 目的是为了启动 GPU，以避免首次计算时 GPU 启动过程导致的延迟，从而影响效率测试。`warmup` 的代码实际上与 `reduceneighbored` 的代码相似，尽管两者之间仍存在一些微小的差别。

#### 4.3. 改善并行规约的分化 Improving Divergence in Parallel Reduction

上述的归约实现显然是最基础的，未经过优化的版本并不适合直接用于实际应用。可以说，一个普遍的真理是：很难在第一次编写代码时就达到满意的效果。

```C
if ((tid % (2 * stride)) == 0)
```

这个条件判断给内核造成了极大的分支，如图所示：

![Parallel Reduction](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction2.png)

- 第一轮有 $\frac {1}{2}$ 的线程没用
- 第二轮有 $\frac {3}{4}$ 的线程没用
- 第三轮有 $\frac {7}{8}$ 的线程没用

对于上面的低利用率，我们想到了下面这个方案来解决：

![Parallel Reduction3](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction3.png)

**注意橙色圆形内的标号是线程符号**，这样的计算线程的利用率是高于原始版本的，核函数如下：

```C
__global__ void reduceNeighboredLess(int * g_idata,int *g_odata,unsigned int n) {
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx > n)
		return;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		//convert tid into local array index
		int index = 2 * stride * tid;
		if (index < blockDim.x) {
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}
```

最关键的一步就是

```C
int index = 2 * stride * tid;
```

这一步保证索引 index 能够向后移动到实际含有数据的内存位置，而不是简单地将线程 ID 与内存地址一一对应，这样做可以避免大量线程处于空闲状态。那么，这样做的效率提升具体体现在哪里呢？

以修改后的代码为例，在 `reduceNeighbored` 中，每个线程 `tid` 都会检查 `(tid % (2 * stride)) == 0` 来决定是否参与加法操作，即使某些线程不需要执行加法，这些线程仍然会被调度并执行检查操作。例如，在第一轮归约时，步长 `stride = 1`，只有偶数索引的线程会执行加法操作，但所有奇数索引的线程仍然会运行代码并进行条件判断，尽管它们实际上没有做任何有用的工作（线程闲置），导致**线程利用率低**的同时造成大量线程束分化。

而 `reduceNeighboredLess` 中，每个线程 `tid` 计算 `index = 2 * stride * tid`，并检查 `index < blockDim.x` 来决定是否参与加法操作。这种方式直接跳过了**后半部分**不会参与加法操作的线程，减少了不必要的线程调度和条件判断。例如，在第一轮归约时，步长 `stride = 1`，只有 `tid = 0, 1, 2, 3` 的线程会计算 `index` 并执行加法操作，而`tid = 4, 5, 6, 7` **虽然还是会遍历一遍但不参与计算过程**，减少了线程束分化。随着 `stride` 增大，活跃线程数逐渐减少，但始终保持 **连续分配**，避免线程浪费。

以下是具体的计算过程：

1. **第一轮** ：步长 `stride = 1`，每个线程 `tid` 计算 `index = 2 * stride * tid`，只有满足 `index < blockDim.x` 的线程会执行加法操作：
	-   `tid = 0`，`index = 0` → `data[0] += data[1]` → `data[0] = 3 + 1 = 4`
	-   `tid = 1`，`index = 2` → `data[2] += data[3]` → `data[2] = 7 + 0 = 7`
	-   `tid = 2`，`index = 4` → `data[4] += data[5]` → `data[4] = 4 + 1 = 5`
	-   `tid = 3`，`index = 6` → `data[6] += data[7]` → `data[6] = 6 + 3 = 9`
1. **第二轮** ：步长 `stride = 2`，每个线程 `tid` 计算 `index = 2 * stride * tid`，即 `index = 4 * tid`，只有满足 `index < blockDim.x` 的线程会执行加法操作：
   -   `data[0] += data[2]` → `data[0] = 4 + 7 = 11`
   -   `data[4] += data[6]` → `data[4] = 5 + 9 = 14`
2. **第三轮** ：步长 `stride = 4`，每个线程 `tid` 计算 `index = 2 * stride * tid`，即 `index = 8 * tid`，只有 `tid = 0` 满足条件，执行加法操作：
   -   `data[0] += data[4]` → `data[0] = 11 + 14 = 25`

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                 elapsed 0.006651 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.002833 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.002634 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.001813 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.001667 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
```

这个效率提升惊人，直接降了一位！大约差了一半。

我们现在就来看看，每个线程束上执行指令的平均数量

```shell
nvprof --metrics inst_per_warp ./reduceInteger
```

```shell
        with array size 16777216  grid 16384 block 1024 
==57663== NVPROF is profiling process 57663, command: ./reduceInteger
cpu sum:2139353471
cpu reduce                 elapsed 0.003717 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.074615 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.069011 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.035108 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.017052 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
==57663== Profiling application: ./reduceInteger
==57663== Profiling result:
==57663== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  361.656250  361.656250  361.656250
    Kernel: warmup(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  886.562500  886.562500  886.562500
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  886.562500  886.562500  886.562500
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  386.906250  386.906250  386.906250
```

指标结果显示，原始核函数的表现为 886.6，而新核函数的表现为 386.9。这表明原始核函数中存在大量被执行但没有实际意义的分支指令。

分化程度越高，`inst_per_warp` 这一指标通常会相应增加。这个概念非常重要，建议大家在未来评估程序效率时，可以经常参考该指标。

接着看一下内存加载吞吐：

```shell
nvprof --metrics gld_throughput ./reduceInteger
```

```shell
        with array size 16777216  grid 16384 block 1024 
==57742== NVPROF is profiling process 57742, command: ./reduceInteger
cpu sum:2139353471
cpu reduce                 elapsed 0.003933 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.182649 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.162657 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.069491 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.060214 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
==57742== Profiling application: ./reduceInteger
==57742== Profiling result:
==57742== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  2.2792GB/s  2.2792GB/s  2.2792GB/s
    Kernel: warmup(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  2.9760GB/s  2.9760GB/s  2.9760GB/s
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  3.2833GB/s  3.2833GB/s  3.2833GB/s
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  7.6661GB/s  7.6661GB/s  7.6661GB/s
```

新核函数的内存效率显著提高，几乎达到了原来的两倍。原因是我们上面分析的，在一个线程块中，前面的几个线程束都在积极工作，而后面的几个线程束则不干活。这种情况下，不执行的线程被自动优化掉，从而集中利用了处于活跃状态的线程的内存请求，实现了带宽的最大化利用。

相比于新核函数中，原来的核函数即便有一些线程不在执行，它们仍然在相同的线程束内运行，但并不发起内存请求。这样一来，内存访问的效率就会被打散，理论上只能达到原有内存效率的一半，而实际测试结果显示，这一数值非常接近。

#### 4.4. 交错配对的规约 Reducing with Interleaved Pairs

上文提到的策略是通过修改线程处理的数据，使得部分线程束能够最大程度地利用数据。接下来，我们将采用相似的思想，但方法有所不同。这次，我们将调整数据的跨度，也就是说，每个线程仍然处理对应的内存位置，但这些内存位置不再是相邻的，而是隔开了一定的距离。

![Parallel Reduction4](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction4.png)

我们依然把上图当做一个完整的线程块，那么前半部分的线程束依然是最大负载在跑，而后半部分的线程束不执行。

```C
__global__ void reduceInterleaved(int * g_idata, int *g_odata, unsigned int n) {
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx >= n)
		return;
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride > 0; stride >>=1) {
		if (tid <stride) {
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}
```

**`reduceInterleaved` 的改进**：
- 采用 **从大到小的跨步（stride）迭代**（例如，初始 `stride=32`，之后逐步减半）。在第一次迭代中，线程 0 访问 `idata[0]` 和 `idata[32]`，线程 1 访问 `idata[1]` 和 `idata[33]`，以此类推。这种模式虽然初始跨度大，但后续迭代中数据逐渐向内存前部集中，**提升缓存局部性**，且在全局内存中更易触发 **合并访问**。而 `reduceNeighboredLess` 在每次迭代中，`index = 2 * stride * tid` 导致线程访问的地址跨度较大（例如，当 `stride=1` 时，线程 0 访问 `idata[0]` 和 `idata[1]`，线程 1 访问 `idata[2]` 和 `idata[3]`）。虽然这种模式在早期迭代中是连续的，但随着 `stride` 增大（如 `stride=2` 时，线程 0 访问 `idata[0]` 和 `idata[2]`），会导致 **非连续的全局内存访问**，降低合并访问效率。
- 条件 `if (tid < stride)` 确保了同一线程束内的线程要么全部活跃，要么全部闲置。例如，当 `stride=32` 时，前 32 个线程（第一个 Warp）全部活跃，后 32 个线程（第二个 Warp）全部闲置。这种模式 **完全避免了线程束分化**，提高了线程束的执行效率。`reduceNeighboredLess` 则是当 `stride` 增大时，`index < blockDim.x` 的条件可能导致同一线程束（Warp）中的部分线程满足条件，另一部分不满足。例如，当 `blockDim.x=64` 且 `stride=16` 时，`index = 2*16*tid`，当 `tid=2` 时 `index=64`，超过块大小，导致部分线程失效，触发线程束分化。

以下是具体的计算过程：

1. **第一轮** ：步长 `stride = blockDim.x / 2 = 4`，只有 `tid < stride` 的线程会执行加法操作：
	-   `tid = 0` → `data[0] += data[4]` → `data[0] = 3 + 4 = 7`
	-   `tid = 1` → `data[1] += data[5]` → `data[2] = 1 + 1 = 2`
	-   `tid = 2` → `data[2] += data[6]` → `data[4] = 7 + 6 = 13`
	-   `tid = 3` → `data[3] += data[7]` → `data[6] = 0 + 3 = 3`
1. **第二轮** ：步长 `stride = 2`，只有 `tid < stride` 的线程会执行加法操作：
	-   `tid = 0` → `data[0] += data[2]` → `data[0] = 7 + 13 = 20`
	-   `tid = 1` → `data[1] += data[3]` → `data[1] = 2 + 3 = 5`
1. **第三轮** ：步长 `stride = 1`，只有 `tid < stride` 的线程会执行加法操作：
	-   `tid = 0` → `data[0] += data[1]` → `data[0] = 20 + 5 = 25`

执行结果：

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                 elapsed 0.006651 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.002833 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.002634 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.001813 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.001667 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
```

从优化原理的角度来看，这个新的核函数与先前的核函数在理论上应该具有相同的效率。然而，实际测试结果显示，这个新核函数的运行速度显著提升。因此，我们有必要进一步考察几个关键指标，以深入理解性能提高的原因。

```C
nvprof --metrics inst_per_warp ./reduceInteger
```

```shell
        with array size 16777216  grid 16384 block 1024 
==58133== NVPROF is profiling process 58133, command: ./reduceInteger
cpu sum:2139353471 
cpu reduce                 elapsed 0.004141 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.074924 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.066874 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.035046 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.016936 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
==58133== Profiling application: ./reduceInteger
==58133== Profiling result:
==58133== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  361.656250  361.656250  361.656250
    Kernel: warmup(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  886.562500  886.562500  886.562500
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  886.562500  886.562500  886.562500
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                             inst_per_warp                     Instructions per warp  386.906250  386.906250  386.906250
```

```C
nvprof --metrics gld_throughput ./reduceInteger
```

```shell
        with array size 16777216  grid 16384 block 1024 
==58189== NVPROF is profiling process 58189, command: ./reduceInteger
cpu sum:2139353471 
cpu reduce                 elapsed 0.003949 ms cpu_sum: 2139353471
gpu warmup                 elapsed 0.181888 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.164936 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.072196 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.060335 ms gpu_sum: 2139353471   <<<grid 16384 block 1024>>>
Test success!
==58189== Profiling application: ./reduceInteger
==58189== Profiling result:
==58189== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceInterleaved(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  2.2776GB/s  2.2776GB/s  2.2776GB/s
    Kernel: warmup(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  2.9780GB/s  2.9780GB/s  2.9780GB/s
    Kernel: reduceNeighbored(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  3.2343GB/s  3.2343GB/s  3.2343GB/s
    Kernel: reduceNeighboredLess(int*, int*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  7.4154GB/s  7.4154GB/s  7.4154GB/s
```

在我们的测试中，reduceInterleaved 的内存吞吐量是最低的，尽管线程束内的分化程度却是最小的。书中提到 reduceInterleaved 的优势主要体现在内存读取效率，而非线程束分化。右前面的分析可知，`reduceInterleaved` 实际上是**减少了不必要的计算**，从大步长开始逐步减小步长，避免了大量线程在早期阶段执行无效的操作。随着步长的减小，参与加法操作的线程数量逐渐减少，减少了线程调度和同步的开销。`reduceInterleaved` 因缓存命中率高，实际从全局内存加载的数据量减少，导致 `gld_throughput` 数值较低。

**此处需要查看机器码，确定两个内核的实际不同**。

### 5. 展开循环 Unrolling Loops

在前面关于执行模型和线程束的讲解中，我们已经明确指出，GPU 不具备分支预测能力。即在执行过程中，每一个分支都会被实际执行。因此，在编写内核时，尽量避免使用分支语句，包括 `if` 语句以及 `for` 等循环结构。这样可以有效减少不必要的计算，提高内核的整体性能。

举例：

```C
for (itn i = 0; i < tid; i++) {  
    // to do something
}
```

如果上面提到的代码出现在内核中，就会引发分支问题。对于一个线程束来说，首个线程和最后一个线程的线程 ID（tid）之间相差 32（假设线程束大小为 32），这意味着在执行 `for` 循环时，每个线程的计算量可能会有所不同。当某些线程完成工作而其他线程仍在运行时，系统必须等待所有线程完成，这就导致了分支的产生。

> 循环展开是一个尝试通过减少分支出现的频率和循环维护指令来优化循环的技术。

上面这句属于书上的官方说法，我们来看看例子，不止并行算法可以展开，传统串行代码展开后效率也能一定程度的提高，因为省去了判断和分支预测失败所带来的迟滞。

```C
for (itn i = 0; i < tid; i++) {  
    a[i] = b[i] + c[i];
}
```

这个是最传统的写法，如果进行循环展开呢？

```C
for (int i = 0; i < 100; i += 4) {
    a[i + 0] = b[i + 0] + c[i + 0];
    a[i + 1] = b[i + 1] + c[i + 1];
    a[i + 2] = b[i + 2] + c[i + 2];
    a[i + 3] = b[i + 3] + c[i + 3];
}
```

我们可以通过手动展开循环来提高性能，将原本由循环完成的任务提前列出。具体而言，我们可以使循环每次增加 4，并在每次迭代中处理 4 个元素。这样做的好处是从串行执行的角度来看，可以**减少条件判断的次数，从而提升性能**。然而，如果将这段代码在机器上运行，可能并不会显著体现出性能差异，因为现代编译器通常会将这两种不同的写法编译为相似的机器代码，也就是说，即使我们不手动展开循环，编译器也可能会进行相应的优化。

不过需要注意的是，**当前的 CUDA 编译器尚未实现这种优化，因此手动展开内核中的循环可以显著提升内核的性能**。

在 CUDA 中展开循环的主要目的是：

1.  **减少指令消耗**
2.  **增加更多的独立调度指令**

如果以下指令被添加到 CUDA 流水线上：

```C
a[i + 0] = b[i + 0] + c[i + 0];
a[i + 1] = b[i + 1] + c[i + 1];
a[i + 2] = b[i + 2] + c[i + 2];
a[i + 3] = b[i + 3] + c[i + 3];
```

这样做是非常受欢迎的，因为它能够**最大限度地提高指令的执行效率和内存带宽的利用率**。通过同时操作多个元素，这种方式能够有效减少内存访问的延迟，同时增加计算的并行性，确保 GPU 资源得到充分利用，从而提升整体性能。

#### 5.1. 展开的归约 Reducing with Unrolling

在 [4. 避免分支分化 Avoiding Branch Divergence](#4.%20避免分支分化%20Avoiding%20Branch%20Divergence) 中，核函数 `reduceInterleaved` 每个线程块通常只处理对应的数据片段。我们有一个新的想法：**是否可以让一个线程块处理多块数据？** 这实际上是可行的。在对这部分数据进行求和之前，我们可以让每个线程执行一次加法操作，将来自其他线程块的数据相加，这相当于先进行一次向量加法，然后再进行归约操作。通过这种方式，我们能够在一条指令中完成之前通常需要多次计算的工作，这样的性价比确实非常诱人。

```C
__global__ void reduceUnroll2(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int* idata = g_idata + blockIdx.x * blockDim.x * 2;
    if (idx + blockDim.x < n) {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

这里面的第二句，第四句，在确定线程块对应的数据的位置的时候有个乘2的偏移量

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250126220921.png)

这就是第二句，第四句指令的意思，我们只处理红色的线程块，而对相邻的白色线程块，我们使用以下代码来处理：

```C
if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
}
```

这里我们采用的是一维线程配置，也就是说我们使用的线程块数量只有原来的二分之一。在仅增加一小句指令的情况下，我们就完成了原本需要全部线程块执行的计算量，这样的效果应该是显而易见的。

首先来看一下调用核函数的部分：

```C
    CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnroll2<<<grid.x / 2, block>>>(idata_dev, odata_dev, size);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for (int i = 0; i < grid.x / 2; i++)
        gpu_sum += odata_host[i];
    printf("reduceUnrolling2            elapsed %lf ms gpu_sum: %d<<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x / 2, block.x);
```

这里需要注意由于合并了一半的线程块，这里的网格个数都要对应的减少一半，来看效率：

```C
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.019267 ms cpu_sum: 2139353471
gpu warmup                  elapsed 0.002203 ms 
reduceUnrolling2            elapsed 0.001801 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
reduceUnrolling4            elapsed 0.001630 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
reduceUnrolling8            elapsed 0.002190 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceUnrollingWarp8        elapsed 0.001327 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnrollWarp8   elapsed 0.000481 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnroll        elapsed 0.000400 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
```

与上一节的效率相比，这种优化的性能提升令人瞩目。相较于最简单的归约算法，它的速度快了三倍。至于 warmup 相关的代码，我们可以暂时不予考虑。

在上面的框中，我们展示了三种不同规模的展开：2、4、8。这分别对应于一个块处理两个、四个和八个块的数据。针对这些变化，对应的调用代码也需要进行相应的修改（见 `chapter03/reduceUnrolling.cu`）。

可以明显看出，直接展开对性能的影响非常显著。这不仅节省了多余线程块的运算时间，还减少了内存加载/存储操作的开销，从而提升了总体性能，进一步提高了延迟的隐藏效果。接下来，让我们来看看它们的吞吐量：

```C
nvprof --metrics dram_read_throughput ./reduceUnrolling
```

```shell
==8102== NVPROF is profiling process 8102, command: ./reduceUnrolling
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.004233 ms cpu_sum: 2139353471
gpu warmup                  elapsed 0.036615 ms 
reduceUnrolling2            elapsed 0.010564 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
reduceUnrolling4            elapsed 0.006182 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
reduceUnrolling8            elapsed 0.004932 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceUnrollingWarp8        elapsed 0.003932 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnrollWarp8   elapsed 0.004921 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnroll        elapsed 0.003486 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
==8102== Profiling application: ./reduceUnrolling
==8102== Profiling result:
==8102== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceCompleteUnrollWarp8(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  28.133GB/s  28.133GB/s  28.133GB/s
    Kernel: warmup(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  987.92MB/s  987.92MB/s  987.92MB/s
    Kernel: reduceUnrollWarp8(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  23.327GB/s  23.327GB/s  23.327GB/s
    Kernel: void reduceCompleteUnroll<unsigned int=1024>(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  28.251GB/s  28.251GB/s  28.251GB/s
    Kernel: reduceUnroll8(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  20.867GB/s  20.867GB/s  20.867GB/s
    Kernel: reduceUnroll4(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  12.719GB/s  12.719GB/s  12.719GB/s
    Kernel: reduceUnroll2(int*, int*, unsigned int)
          1                      dram_read_throughput             Device Memory Read Throughput  7.2209GB/s  7.2209GB/s  7.2209GB/s
```

可见执行效率是和内存吞吐量是呈正相关的

#### 5.2. 展开线程的归约 Reducing with Unrolled Warps

接下来，我们的目标是处理最后那 32 个线程。由于归约运算的特性呈现出倒金字塔形状，最终的结果只有一个数。因此，在最后 64 个线程进行计算得到数字结果的过程中，每执行一步，线程的利用率就会降低一半：从 64 降到 32，接着是 16 …… 依此类推，直到最终只有 1 个线程在工作。

为了提升效率，我们希望展开最后的6步迭代（即从 64 到 32、16、8、4、2，再到 1）。为此，我们可以使用以下核函数来实现这最后6步的分支计算：

```C
__global__ void reduceUnrollWarp8(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;
    // unrolling 8;
    if (idx + 7 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int a5       = g_idata[idx + 4 * blockDim.x];
        int a6       = g_idata[idx + 5 * blockDim.x];
        int a7       = g_idata[idx + 6 * blockDim.x];
        int a8       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid < 32) {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

在 `unrolling8` 的基础上，我们将针对线程 `tid` 在 范围内的情况使用以下代码进行展开：

```C
volatile int* vmem = idata;
vmem[tid] += vmem[tid + 32];
vmem[tid] += vmem[tid + 16];
vmem[tid] += vmem[tid + 8];
vmem[tid] += vmem[tid + 4];
vmem[tid] += vmem[tid + 2];
vmem[tid] += vmem[tid + 1];
```

首先，我们在这里定义了一个 `volatile int` 类型的变量，具体的意义稍后再讲解。现在我们理清一下最后的展开过程。当只剩下底部的三角部分时，我们需要将 64 个数合并为一个数。第一步是针对前 32 个数，按照步长为 32 进行并行加法，得到两个数的和并存储在前 32 个数字中。

接着，我们将这 32 个数与步长为 16 的变量相加，得出这 16 个数的结果。这 16 个数的和便是该块的归约结果。然而，根据之前的 `tid < 32` 判断条件，线程 `tid` 为 16 到 31 的线程仍在运行，但此时它们的结果已经没有实际意义。这个步骤非常关键。

另一个可能引发疑惑的点在于：**由于线程是同步执行的，会不会出现线程 17 在加上线程 33 的结果后再写入 17 号内存，这样线程 1 在后续加法时就会导致错误呢**？实际上，在 CUDA 内核中，线程从内存中读取数据到寄存器以及执行加法操作都是同步的。因此，线程 17 和线程 1 会同时读取 33 号和 17 号的内存，尽管线程 17 在下一步可能会进行更改，但这并不影响寄存器中存储的值。

虽然 32 以内的 `tid` 线程都在运行，但每执行一步，后面一半线程的结果将失去意义。这样，继续进行计算的结果最终将保留在 中，成为最后有效的结果。

上述过程有些复杂，但如果我们仔细思考每一步，从数据读取到计算，每一步都分析一下，可以助我们理解实际的结果。

关于 `volatile int` 类型变量，它的作用是**保证变量的值在写回至内存后能够立即反映变化，而不是存在共享内存或缓存中**。这是因为在后续计算中，我们需要使用到这些值，如果它们被缓存在寄存器中，可能会导致读取错误的数据。

```C
vmem[tid] += vmem[tid + 32];
vmem[tid] += vmem[tid + 16];
```

这里 `tid + 16` 的计算依赖于 `tid + 32` 的结果，那么是否会有其他线程导致内存竞争呢？答案是不会的，因为在 CUDA 中，线程束（warp）内的线程执行进度是完全一致的。当执行 `tid + 32` 的操作时，这 32 个线程都在并行进行这一指令，不会有任何线程会在同一个线程束内提前执行到下一条语句。由于 CUDA 编译器的激进优化策略，我们必须使用 `volatile` 关键字来确保对数据的传输不被优化而打乱执行顺序。详情请看看一下 CUDA 的执行模型以加深理解。

接下来，我们观察到的结果如下所示：

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.019267 ms cpu_sum: 2139353471
gpu warmup                  elapsed 0.002203 ms 
reduceUnrolling2            elapsed 0.001801 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
reduceUnrolling4            elapsed 0.001630 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
reduceUnrolling8            elapsed 0.002190 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceUnrollingWarp8        elapsed 0.001327 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnrollWarp8   elapsed 0.000481 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnroll        elapsed 0.000400 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
```

从结果中可以看到，性能表现依然十分优异。采用循环展开的策略，我们不仅优化了计算速度，还显著减少了线程束之间的同步开销。具体来说，我们减少了 5 次对 `__syncthreads()` 指令的调用，这一调整在性能上产生了显著的正面影响。

我们一起来分析一下，究竟减少了多少阻塞时间，以及这些改进对整体性能的贡献。

使用命令

```shell
nvprof --metrics stall_sync ./reduceUnrolling
```

```shell
==8144== NVPROF is profiling process 8144, command: ./reduceUnrolling
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.004224 ms cpu_sum: 2139353471
==8144== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(int*, int*, unsigned int)" (done)
gpu warmup                  elapsed 0.278691 ms 
Replaying kernel "reduceUnroll2(int*, int*, unsigned int)" (done)
reduceUnrolling2            elapsed 0.078157 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
Replaying kernel "reduceUnroll4(int*, int*, unsigned int)" (done)
reduceUnrolling4            elapsed 0.057438 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
Replaying kernel "reduceUnroll8(int*, int*, unsigned int)" (done)
reduceUnrolling8            elapsed 0.062355 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Replaying kernel "reduceUnrollWarp8(int*, int*, unsigned int)" (done)
reduceUnrollingWarp8        elapsed 0.043762 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Replaying kernel "reduceCompleteUnrollWarp8(int*, int*, unsigned int)" (done)
reduceCompleteUnrollWarp8   elapsed 0.041221 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Replaying kernel "void reduceCompleteUnroll<unsigned int=1024>(int*, int*, unsigned int)" (done)
reduceCompleteUnroll        elapsed 0.044732 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
==8144== Profiling application: ./reduceUnrolling
==8144== Profiling result:
==8144== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceCompleteUnrollWarp8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      29.88%      29.88%      29.88%
    Kernel: warmup(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      38.75%      38.75%      38.75%
    Kernel: reduceUnrollWarp8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      26.33%      26.33%      26.33%
    Kernel: void reduceCompleteUnroll<unsigned int=1024>(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      30.16%      30.16%      30.16%
    Kernel: reduceUnroll8(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      34.81%      34.81%      34.81%
    Kernel: reduceUnroll4(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      41.72%      41.72%      41.72%
    Kernel: reduceUnroll2(int*, int*, unsigned int)
          1                                stall_sync     Issue Stall Reasons (Synchronization)      47.24%      47.24%      47.24%
```

可以看出，优化后的 `reduceUnrollWarp8` 阻塞率为 `26.33%`，低于 `reduceUnroll` 的 `34.81%`，说明优化后效果还是非常明显的。

#### 5.3. 完全展开的归约 Reducing with Complete Unrolling

根据上面展开最后 64 个数据，我们可以直接就展开最后 128 个，256 个，512 个，1024 个：  

```C
__global__ void reduceCompleteUnrollWarp8(int* g_idata, int* g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    // boundary check
    if (tid >= n)
        return;
    // convert global data pointer to the
    int* idata = g_idata + blockIdx.x * blockDim.x * 8;
    if (idx + 7 * blockDim.x < n) {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int a5       = g_idata[idx + 4 * blockDim.x];
        int a6       = g_idata[idx + 5 * blockDim.x];
        int a7       = g_idata[idx + 6 * blockDim.x];
        int a8       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();
    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
    // write result for this block to global mem
    if (tid < 32) {
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
        g_odata[blockIdx.x] = idata[0];
}
```

核函数代码如上，在这里，我们需要注意 `tid` 的大小。与最后 32 个线程不同，如果完全计算这些 `tid` 的值，会有一半的计算结果是冗余的。这是因为在实际计算中，许多线程的输出并不会被后续的处理阶段使用。

然而，最后的 32 个线程由于是线程束（warp）中最小的单元，它们的执行特点确保了无论后续的数据是否具有实际意义，正在执行的线程进程都不会中断。因此，即使存在多余的计算，线程的执行效率也能够得到保证。这种设计使得 CUDA 编程能够在并行计算中最大化性能，避免了不必要的停顿。

然后我们看结果：

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.019267 ms cpu_sum: 2139353471
gpu warmup                  elapsed 0.002203 ms 
reduceUnrolling2            elapsed 0.001801 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
reduceUnrolling4            elapsed 0.001630 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
reduceUnrolling8            elapsed 0.002190 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceUnrollingWarp8        elapsed 0.001327 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnrollWarp8   elapsed 0.000481 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnroll        elapsed 0.000400 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
```

速度优化到原来的3倍左右，效果非常明显。

#### 5.4. 模板函数的归约 Reducing with Template Functions

我们看上面这个完全展开的函数：

```C
    if (blockDim.x >= 1024 && tid < 512)
        idata[tid] += idata[tid + 512];
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
```

在这里，某些条件判断显得有些多余。因为在核函数启动后，`blockDim.x` 的值是固定的，无法再发生改变。因此，使用模板函数可以有效地解决这个问题。在编译时，编译器会对 `blockDim.x` 进行检查，如果它的值已知且固定，那么不必要的条件部分将会被自动删除。

例如，当 `blockDim.x` 的值为 512 时，最终生成的机器码会是如下部分，这样就去除了不必要的条件判断：

```C
    if (blockDim.x >= 512 && tid < 256)
        idata[tid] += idata[tid + 256];
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
        idata[tid] += idata[tid + 128];
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
        idata[tid] += idata[tid + 64];
    __syncthreads();
```

通过这样的优化，代码不仅更加简洁，而且运行效率也得到了提升。 

我们来看下模板函数的效率：

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.019267 ms cpu_sum: 2139353471
gpu warmup                  elapsed 0.002203 ms 
reduceUnrolling2            elapsed 0.001801 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
reduceUnrolling4            elapsed 0.001630 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
reduceUnrolling8            elapsed 0.002190 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceUnrollingWarp8        elapsed 0.001327 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnrollWarp8   elapsed 0.000481 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
reduceCompleteUnroll        elapsed 0.000400 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
```

很明显，是这几个核函数中速度最快的那一个。

加载效率存储效率：

```C
nvprof --metrics gld_efficiency,gst_efficiency ./reduceUnrolling
```

```shell
==9099== NVPROF is profiling process 9099, command: ./reduceUnrolling
        with array size 16777216  grid 16384 block 1024 
cpu sum:2139353471 
cpu reduce                  elapsed 0.004055 ms cpu_sum: 2139353471
==9099== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "warmup(int*, int*, unsigned int)" (4 of 4)... 
Replaying kernel "warmup(int*, int*, unsigned int)" (done)
Replaying kernel "reduceUnroll2(int*, int*, unsigned int)" (done)
reduceUnrolling2            elapsed 0.140167 ms gpu_sum: 2139353471<<<grid 8192 block 1024>>>
Replaying kernel "reduceUnroll4(int*, int*, unsigned int)" (done)
reduceUnrolling4            elapsed 0.100995 ms gpu_sum: 2139353471<<<grid 4096 block 1024>>>
Replaying kernel "reduceUnroll8(int*, int*, unsigned int)" (done)
reduceUnrolling8            elapsed 0.079325 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Replaying kernel "reduceUnrollWarp8(int*, int*, unsigned int)" (done)
reduceUnrollingWarp8        elapsed 0.065995 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Replaying kernel "reduceCompleteUnrollWarp8(int*, int*, unsigned int)" (done)
reduceCompleteUnrollWarp8   elapsed 0.063203 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Replaying kernel "void reduceCompleteUnroll<unsigned int=1024>(int*, int*, unsigned int)" (done)
reduceCompleteUnroll        elapsed 0.072666 ms gpu_sum: 2139353471<<<grid 2048 block 1024>>>
Test success!
==9099== Profiling application: ./reduceUnrolling
==9099== Profiling result:
==9099== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: reduceCompleteUnrollWarp8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.71%      99.71%      99.71%
          1                            gst_efficiency            Global Memory Store Efficiency      99.68%      99.68%      99.68%
    Kernel: warmup(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.01%      25.01%      25.01%
          1                            gst_efficiency            Global Memory Store Efficiency      25.00%      25.00%      25.00%
    Kernel: reduceUnrollWarp8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.71%      99.71%      99.71%
          1                            gst_efficiency            Global Memory Store Efficiency      99.68%      99.68%      99.68%
    Kernel: void reduceCompleteUnroll<unsigned int=1024>(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.71%      99.71%      99.71%
          1                            gst_efficiency            Global Memory Store Efficiency      99.68%      99.68%      99.68%
    Kernel: reduceUnroll8(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.75%      99.75%      99.75%
          1                            gst_efficiency            Global Memory Store Efficiency      99.71%      99.71%      99.71%
    Kernel: reduceUnroll4(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.50%      99.50%      99.50%
          1                            gst_efficiency            Global Memory Store Efficiency      99.42%      99.42%      99.42%
    Kernel: reduceUnroll2(int*, int*, unsigned int)
          1                            gld_efficiency             Global Memory Load Efficiency      99.01%      99.01%      99.01%
          1                            gst_efficiency            Global Memory Store Efficiency      98.84%      98.84%      98.84%
```

可以看出，通过一步步的优化，后面三个优化的核函数都达到了 99% 的效率。

### 6. 动态并行

本文作为第三章“CUDA执行模型”的最后一部分，将介绍**动态并行（Dynamic Parallelism）**。书中提到的动态并行示例使用了嵌套归约的方式，我认为这个示例对于我们的实际应用帮助不大。首先，它并没有显著降低代码的复杂度，其次，它的运行效率也未必提高。**动态并行可以被看作是串行编程中的递归调用**，而递归调用如果能够被转换为迭代循环，通常出于性能考虑会选择这种转换。只有在不太关注效率而更注重代码简洁性的情况下，我们才会选择使用递归。因此，在本文中，我们将仅介绍一些基础知识。对于希望深入了解动态并行的读者，建议查阅相关的文档或专业博客。

到目前为止，我们的所有内核都是在主机线程中调用的。那么，我们是否可以在内核内部调用其他内核，甚至是自身呢？这便需要动态并行的支持。需要注意的是，早期的设备并不支持这一功能。

动态并行的一个主要优点是**能够使复杂的内核结构变得层次分明**。然而，这也带来了一个劣势，即编写的程序复杂度更高，因为并行行为的控制本就不易。动态并行的另一个优点是**在执行时灵活配置网格和块的数量，这样可以更好地利用GPU的硬件调度器和负载平衡机制**。通过动态调整，可以有效适应不同的计算负载。此外，在内核中启动其他内核还有助于减少部分数据传输开销。

#### 6.1. 嵌套执行 Nested Execution
 
前面我们详细了解了网格、块和启动配置，以及一些与线程束相关的知识。现在，我们将探讨在内核中启动内核的概念。

在 CUDA 编程中，内核中启动内核的机制与 CPU 并行计算中的父线程与子线程的概念相似。**子线程是由父线程启动的**，但在 GPU 的上下文中，这些术语变得更加具体化，包括**父网格**、**父线程块**和**父线程**，以及对应的**子网格**、**子线程块**和**子线程**。

在这个模型中，子网格是由父线程启动的，并且必须在对应的父线程、父线程块和父网格完成之前结束。换句话说，父线程、父线程块和父网格将等待其下属的所有子网格完成执行后才能结束。这种结构允许我们利用动态并行性来创建更复杂的并发执行模型，使得 GPU 在处理复杂计算时能够更加灵活和高效。

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250126225006.png)

上图清晰地表明了父网格和子网格的使用情况，一种典型的执行方式：

> 主机启动一个网格（即一个内核）-> 此网格（父网格）在执行的过程中启动新的网格（子网格们）-> 所有子网格运行结束后 -> 父网格才能结束，否则将需要等待

**如果调用的线程没有显式地同步启动子网格，运行时系统会保证父网格和子网格之间的隐式同步**。图中通过栅栏设置进行了显式同步，以确保父网格和子网格的协调。

在父网格中，不同线程可以启动不同的子网格，这些子网格共享相同的父线程块，从而可以相互同步。父线程块中的所有线程创建的全部子网格完成后，线程块的执行才会完成。如果块中的所有线程在子网格完成前退出，则会触发隐式同步。隐式同步的特点在于，即使没有使用同步指令，父线程块的所有线程执行完毕后，依然会等待所有子网格完成执行才会退出。

我们之前提到过隐式同步的应用，比如 `cudaMemcpy` 可以起到隐式同步的作用。然而，在主机内启动的网格中，如果没有显式同步，也没有隐式同步指令，那么 CPU 线程可能会提前退出，而 GPU 程序仍在运行，这会导致不希望的情况发生。**因此，父线程块在启动子网格时需要进行显式同步**。这意味着不同的线程束需要同时执行到子网格调用那一行，这样同一线程块下的所有子网格才能同步执行完成。

接下来，我们讨论内存管理，这是动态并行中最棘手的部分。内存竞争问题在普通并行中已经比较复杂了，而在动态并行中更加麻烦，主要存在以下几点注意事项：

1.  父网格和子网格共享相同的全局和常量内存。
2.  父网格和子网格各自拥有不同的局部内存。
3.  利用子网格和父网格之间的弱一致性，允许父网格和子网格对全局内存进行并发访问。
4.  父网格和子网格之间存在一致性的两个时刻：子网格启动时和子网格结束时。
5.  共享内存和局部内存分别对线程块和线程私有。
6.  局部内存对线程私有且对外不可见。

#### 6.2. 在GPU上嵌套 Hello World  Nested Hello World on the GPU

为了研究初步动态并行，我们先来写个 Hello World 进行操作，代码如下：

```C
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void nesthelloworld(int iSize, int iDepth) {
    unsigned int tid = threadIdx.x;
    printf("depth : %d blockIdx: %d,threadIdx: %d\n", iDepth, blockIdx.x, threadIdx.x);
    if (iSize == 1)
        return;
    int nthread = (iSize >> 1);
    if (tid == 0 && nthread > 0) {
        nesthelloworld<<<1, nthread>>>(nthread, ++iDepth);
        printf("-----------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char* argv[]) {
    int  size    = 64;
    int  block_x = 2;
    
    dim3 block(block_x, 1);
    dim3 grid((size - 1) / block.x + 1, 1);
    nesthelloworld<<<grid, block>>>(size, 0);
    cudaGetLastError();
    cudaDeviceReset();
    
    return 0;
}
```

此程序的功能如下：

-   **第一层**：多个线程块将执行输出。每个线程块的 `tid == 0` 线程启动一个子网格，子网格的配置为当前线程数的一半，以及输入参数 `iSize`。
-   **第二层**：在每个子网格中，将执行输出。与第一层相似，`tid == 0` 的子线程将启动新的子网格，这些子网格同样将其线程数和输入参数 `iSize` 配置为当前值的一半。
-   **第三层**：递归调用将继续进行，直到 `iSize` 减小到 1 为止。在此情况下，程序将停止执行递归，达到结束条件。

编译的命令与之前有些不同，工程中使用 cmake 管理，可在 CMakeLists.txt 中查看：

```CMake
add_executable(nestedHelloWorld nestedHelloWorld.cu)
target_compile_options(nestedHelloWorld PRIVATE -g -G -O3 -rdc=true)
set_target_properties(nestedHelloWorld PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
install(TARGETS nestedHelloWorld DESTINATION chapter03)
```

-rdc=true 是前面没有的，指的是生成可重新定位的代码，第十章将会讲解更多重新定位设备代码的内容。CUDA_SEPARABLE_COMPILATION ON 是动态并行需要的一个库。

执行结果如下，有点长，但是能看出一些问题：

```shell
-----------> nested execution depth: 6
depth : 4 blockIdx: 0, threadIdx: 0
depth : 4 blockIdx: 0, threadIdx: 1
depth : 4 blockIdx: 0, threadIdx: 2
depth : 4 blockIdx: 0, threadIdx: 3
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 6
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 5
-----------> nested execution depth: 4
-----------> nested execution depth: 6
-----------> nested execution depth: 5
-----------> nested execution depth: 5
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 5
-----------> nested execution depth: 6
-----------> nested execution depth: 6
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 5
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 5
-----------> nested execution depth: 5
-----------> nested execution depth: 6
-----------> nested execution depth: 5
-----------> nested execution depth: 6
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 5
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 5
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 5
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
-----------> nested execution depth: 6
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 5 blockIdx: 0, threadIdx: 0
depth : 5 blockIdx: 0, threadIdx: 1
-----------> nested execution depth: 5
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
-----------> nested execution depth: 6
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
-----------> nested execution depth: 6
depth : 6 blockIdx: 0, threadIdx: 0
```

可以看出，在进行多层调用子网格时，同一父线程的子网格之间会实现隐式同步。这意味着，属于同一“家族”——即拥有相同祖先线程的子网格——会根据父线程的执行状态进行协调，确保它们的执行是同步的。而不同“家族”——即具有不同父线程的子网格——则各自独立地运行，不会互相干扰。

### 6. 总结 Summary

至此，本篇系统性地介绍了 CUDA 的 **执行模型、线程组织、资源分配及优化方法**，这些知识是构建高效 CUDA 程序的基础。下一篇将进一步深入研究 **GPU 全局内存架构与数据访问模式**，探索如何优化 **内存带宽、缓存利用、全局内存**，以进一步提升 CUDA 计算性能。

---

## 参考引用

### 书籍出处


- [CUDA C编程权威指南](../../../asset/CUDA%20&%20GPU%20Programming/CUDA%20C编程权威指南.pdf)
- [Professional CUDA C Programming](../../../asset/CUDA%20&%20GPU%20Programming/Professional%20CUDA%20C%20Programming.pdf)

### 网页链接

- [人工智能编程 | 谭升的博客](https://face2ai.com/program-blog/)