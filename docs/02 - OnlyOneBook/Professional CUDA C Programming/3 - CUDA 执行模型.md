## 3 - CUDA 执行模型 CUDA Execution Model

---

这一篇开始我们开始接近CUDA最核心的部分，就是有关硬件，和程序的执行模型，用CUDA的目的其实说白了就是为计算速度快，所以压榨性能，提高效率其实就是CUDA学习的最终目的，没人学CUDA为了去显示 Hello world。  

前面几篇我们学了编写，启动核函数，计时，统计时间，然后学习了线程，内存模型，线程内存部分我们会在后面用几章的篇幅进行大书特书，而本章，我们介绍最底层最优理论指导意义的知识。  

什么时候我们沿着硬件设计的思路设计程序，我们就会得到百战百胜；什么时候我们背离了硬件设计的思路去设计程序，我们就会得不到好结果。

### 1. 概述 Introducing the CUDA Execution Model

CUDA执行模型揭示了GPU并行架构的抽象视图，再设计硬件的时候，其功能和特性都已经被设计好了，然后去开发硬件，如果这个过程模型特性或功能与硬件设计有冲突，双方就会进行商讨妥协，直到最后产品定型量产，功能和特性算是全部定型，而这些功能和特性就是变成模型的设计基础，而编程模型又直接反应了硬件设计，从而反映了设备的硬件特性。

比如最直观的一个就是内存，线程的层次结构帮助我们控制大规模并行，这个特性就是硬件设计最初设计好，然后集成电路工程师拿去设计，定型后程序员开发驱动，然后在上层可以直接使用这种执行模型来控制硬件。

所以了解CUDA的执行模型，可以帮助我们优化指令吞吐量，和内存使用来获得极限速度。

#### 1.1. GPU 架构概述 GPU Architecture Overview

GPU架构是围绕一个流式多处理器（SM）的扩展阵列搭建的。通过复制这种结构来实现GPU的硬件并行

![Streaming Multiprocessors](/images/Professional%20CUDA%20C%20Programming/Streaming%20Multiprocessors.png)

上图包括关键组件：

- CUDA 核心
- 共享内存/一级缓存
- 寄存器文件
- 加载/存储单元
- 特殊功能单元
- 线程束调度器

##### 1.1.1. 流式处理器 Streaming Multiprocessors

GPU中每个SM都能支持数百个线程并发执行，每个GPU通常有多个SM，当一个核函数的网格被启动的时候，多个block会被同时分配给可用的SM上执行。当一个blcok被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。在SM上同一个块内的多个线程进行线程级别并行，而同一线程内，指令利用指令级并行将单个线程处理成流水线。

##### 1.1.2. 线程束 Threads

CUDA 采用单指令多线程SIMT架构管理执行线程，不同设备有不同的线程束大小，但是到目前为止基本所有设备都是维持在32，也就是说每个SM上有多个block，一个block有多个线程（可以是几百个，但不会超过某个最大值），但是从机器的角度，在某时刻T，SM上只执行一个线程束，也就是32个线程在同时同步执行，线程束中的每个线程执行同一条指令，包括有分支的部分

##### 1.1.3. SIMD vs SIMT

单指令多数据的执行属于向量机，比如我们有四个数字要加上四个数字，那么我们可以用这种单指令多数据的指令来一次完成本来要做四次的运算。这种机制的问题就是过于死板，不允许每个分支有不同的操作，所有分支必须同时执行相同的指令，必须执行没有例外。

相比之下单指令多线程SIMT就更加灵活了，虽然两者都是将相同指令广播给多个执行单元，但是SIMT的某些线程可以选择不执行，也就是说同一时刻所有线程被分配给相同的指令，SIMD规定所有人必须执行，而SIMT则规定有些人可以根据需要不执行，这样SIMT就保证了线程级别的并行，而SIMD更像是指令级别的并行。

SIMT包括以下SIMD不具有的关键特性：

1. 每个线程都有自己的指令地址计数器
2. 每个线程都有自己的寄存器状态
3. 每个线程可以有一个独立的执行路径

而上面这三个特性在编程模型可用的方式就是给每个线程一个唯一的标号（blckIdx,threadIdx），并且这三个特性保证了各线程之间的独立

##### 1.1.4. 数字32 Number 32

32是个神奇数字，他的产生是硬件系统设计的结果，也就是集成电路工程师搞出来的，所以软件工程师只能接受。

从概念上讲，32是SM以SIMD方式同时处理的工作粒度，这句话这么理解，可能学过后面的会更深刻的明白，一个SM上在某一个时刻，有32个线程在执行同一条指令，这32个线程可以选择性执行，虽然有些可以不执行，但是他也不能执行别的指令，需要另外需要执行这条指令的线程执行完，然后再继续下一条，就像老师给小朋友们分水果：

- 第一次分苹果，分给所有32个人，你可以不吃，但是不吃也没别的，你就只能在那看别人吃，等别人吃完了，老师会把没吃的苹果回收，防止浪费。
- 第二次分橘子，你很爱吃，可是有别的小朋友不爱吃，当然这时候他也不能干别的，只能看你吃完。吃完后老师继续回收刚才没吃的橘子。
- 第三次分桃子，你们都很爱吃，大家一起吃，吃完了老师发现没有剩下的，继续发别的水果，一直发到所有种类的水果都发完了。今天就可以放学了。

简单的类比，但过程就是这样

##### 1.1.5. CUDA 编程的组件与逻辑 CUDA Programming Components and Logic

下图从逻辑角度和硬件角度描述了CUDA编程模型对应的组件

![logical view and hardware view of CUDA](/images/Professional%20CUDA%20C%20Programming/logical%20view%20and%20hardware%20view%20of%20CUDA.png)

SM中共享内存，和寄存器是关键的资源，线程块中线程通过共享内存和寄存器相互通信协调。寄存器和共享内存的分配会严重影响性能

因为SM有限，虽然编程模型层面看所有线程都是并行执行的，但是在微观上看，所有线程块也是分批次的在物理层面的机器上执行，线程块里不同的线程可能进度都不一样，但是同一个线程束内的线程拥有相同的进度。并行就会引起竞争，多线程以未定义的顺序访问同一个数据，就导致了不可预测的行为，CUDA只提供了一种块内同步的方式，块之间没办法同步！

同一个SM上可以有不止一个常驻的线程束，有些在执行，有些在等待，他们之间状态的转换是不需要开销的。

#### 1.2. Fermi 架构 The Fermi Architecture

Fermi架构是第一个完整的GPU架构，所以了解这个架构是非常有必要的

![Fermi Architecture](/images/Professional%20CUDA%20C%20Programming/Fermi%20Architecture.png)

Fermi架构逻辑图如上，具体数据如下:

1. 512个加速核心，CUDA核
2. 每个CUDA核心都有一个全流水线的整数算数逻辑单元ALU，和一个浮点数运算单元FPU
3. CUDA核被组织到16个SM上
4. 6个384-bits的 GDDR5 的内存接口
5. 支持6G的全局机栽内存
6. GigaThread引擎，分配线程块到SM线程束调度器上
7. 768KB的二级缓存，被所有SM共享

而SM则包括下面这些资源：

- 执行单元（CUDA核）
- 调度线程束的调度器和调度单元
- 共享内存，寄存器文件和一级缓存

每个多处理器SM有16个加载/存储单元，所以每个时钟周期内有16个线程（半个线程束）计算源地址和目的地址。特殊功能单元SFU执行固有指令，如正弦，余弦，平方根和插值，SFU在每个时钟周期内的每个线程上执行一个固有指令

每个SM有两个线程束调度器，和两个指令调度单元，当一个线程块被指定给一个SM时，线程块内的所有线程被分成线程束，两个线程束选择其中两个线程束，在用指令调度器存储两个线程束要执行的指令（就像上面例子中分水果的水果一样，我们这里有两个班，两个班的老师各自控制的自己的水果，老师就是指令调度器）

像第一张图上的显示一样，每16个CUDA核心为一个组，还有16个加载/存储单元或4个特殊功能单元。当某个线程块被分配到一个SM上的时候，会被分成多个线程束，线程束在SM上交替执行：

![Fermi Execution](/images/Professional%20CUDA%20C%20Programming/SM%20Execution.png)

上面曾经说过，每个线程束在同一时间执行同一指令，同一个块内的线程束互相切换是没有时间消耗的。Fermi上支持同时并发执行内核。并发执行内核允许执行一些小的内核程序来充分利用GPU，如图：

![Fermi Execution](/images/Professional%20CUDA%20C%20Programming/Fermi%20Execution.png)

#### 1.3. Kepler 架构 The Kepler Architecture

Kepler架构作为Fermi架构的后代，有以下技术突破：

- 强化的SM
- 动态并行
- Hyper-Q技术

技术参数也提高了不少，比如单个SM上CUDA核的数量，SFU的数量，LD/ST的数量等：

![Kepler Architecture1](/images/Professional%20CUDA%20C%20Programming/Kepler%20Architecture1.png)

![Kepler Architecture2](/images/Professional%20CUDA%20C%20Programming/Kepler%20Architecture2.png)

kepler架构的最突出的一个特点就是内核可以启动内核了，这使得我们可以使用GPU完成简单的递归操作，流程如下:

![Dynamic Parallelism](/images/Professional%20CUDA%20C%20Programming/Dynamic%20Parallelism.png)

Hyper-Q技术主要是CPU和GPU之间的同步硬件连接，以确保CPU在GPU执行的同时做更多的工作。Fermi架构下CPU控制GPU只有一个队列，Kepler架构下可以通过Hyper-Q技术实现多个队列如下图

![Hyper-Q](/images/Professional%20CUDA%20C%20Programming/Hyper-Q.png)

计算能力概览：

![Compute Capability1](/images/Professional%20CUDA%20C%20Programming/Compute%20Capability1.png)

![Compute Capability2](/images/Professional%20CUDA%20C%20Programming/Compute%20Capability2.png)

#### 1.4. 使用Profile进行优化Profile-Driven Optimization

中文翻译的这个标题是配置文件驱动优化，驱动这个词在这里应该是个动词，或者翻译的人直接按照字面意思翻译的，其实看完内容以后的意思是根据profile这个文件内的信息对程序进行优化。  

性能分析通过以下方法来进行：

1.  应用程序代码的空间(内存)或时间复杂度
2.  特殊指令的使用
3.  函数调用的频率和持续时间

程序优化是建立在对硬件和算法过程理解的基础上的，如果都不了解，靠试验，那么这个结果可想而知。理解平台的执行模型也就是硬件特点，是优化性能的基础。  

开发高性能计算程序的两步：

1.  保证结果正确，和程序健壮性
2.  优化速度

Profile可以帮助我们观察程序内部。

-   一个原生的内核应用一般不会产生最佳效果，也就是我们基本不能一下子就写出最好最快的内核，需要通过性能分析工具分析性能。找出性能瓶颈
-   CUDA将SM中的计算资源在该SM中的多个常驻线程块之间进行分配，这种分配方式可能导致一些资源成为性能限制因素，性能分析工具可以帮我们找出来这些资源是如何被使用的
-   CUDA提供了一个硬件架构的抽象。它能够让用户控制线程并发。性能分析工具可以检测和优化，并肩优化可视化

总结起来一句话，想优化速度，先学好怎么用性能分析工具。

-   nvvp
-   nvprof

限制内核性能的主要包括但不限于以下因素

-   存储带宽
-   计算资源
-   指令和内存延迟

### 2. 理解线程束执行的本质 Understanding the Nature of Warp Execution

前面已经大概的介绍了CUDA执行模型的大概过程，包括线程网格，线程束，线程间的关系，以及硬件的大概结构，例如SM的大概结构。而对于硬件来说，CUDA执行的实质是线程束的执行，因为硬件根本不知道每个块谁是谁，也不知道先后顺序，硬件(SM)只知道按照机器码跑，而给他什么，先后顺序，这个就是硬件功能设计的直接体现了。

从外表来看，CUDA执行所有的线程，并行的，没有先后次序的，但实际上硬件资源是有限的，不可能同时执行百万个线程，所以从硬件角度来看，物理层面上执行的也只是线程的一部分，而每次执行的这一部分，就是我们前面提到的线程束。

#### 2.1. 线程束和线程块 Warps and Thread Blocks

线程束是SM中基本的执行单元，当一个网格被启动（网格被启动，等价于一个内核被启动，每个内核对应于自己的网格），网格中包含线程块，线程块被分配到某一个SM上以后，将分为多个线程束，每个线程束一般是32个线程（目前的GPU都是32个线程，但不保证未来还是32个）在一个线程束中，所有线程按照单指令多线程SIMT的方式执行，每一步执行相同的指令，但是处理的数据为私有的数据，下图反应的就是逻辑，实际，和硬件的图形化

![logical view and hardware view of a thread block](/images/Professional%20CUDA%20C%20Programming/view%20of%20a%20thread%20block.png)

线程块是个逻辑产物，因为在计算机里，内存总是一维线性存在的，所以执行起来也是一维的访问线程块中的线程，但是在写程序的时候却可以以二维三维的方式进行，原因是方便写程序，比如处理图像或者三维的数据，三维块就会变得很直接，很方便。

- 在块中，每个线程有唯一的编号（可能是个三维的编号），threadIdx
- 网格中，每个线程块也有唯一的编号(可能是个三维的编号)，blockIdx

那么每个线程就有在网格中的唯一编号。

当一个线程块中有128个线程的时候，其分配到SM上执行时，会分成4个块：

```shell
warp0: thread  0, ... , thread 31
warp1: thread 32, ... , thread 63
warp2: thread 64, ... , thread 95
warp3: thread 96, ... , thread 127
```

当编号使用三维编号时，x位于最内层，y位于中层，z位于最外层，想象下c语言的数组，如果把上面这句话写成c语言，假设三维数组t保存了所有的线程，那么(threadIdx.x,threadIdx.y,threadIdx.z)表示为：

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

ceil函数是向上取整的函数，如下图所示的 $ceil(\frac{80}{32}) = 3$。注意，最后半个线程束是不活跃的。即使这些线程未被使用，它们仍然消耗SM的资源，如寄存器，就是前文提到的吃苹果的例子。

![allocate warps of threads](/images/Professional%20CUDA%20C%20Programming/allocate%20warps%20of%20threads.png)

线程束和线程块，一个是硬件层面的线程集合，一个是逻辑层面的线程集合，我们编程时为了程序正确，必须从逻辑层面计算清楚，但是为了得到更快的程序，硬件层面是我们应该注意的。

#### 2.2. 线程束分化 Warp Divergence

线程束被执行的时候会被分配给相同的指令，处理各自私有的数据，还记得前文中的分苹果么？每次分的水果都是一样的，但是你可以选择吃或者不吃，这个吃和不吃就是分支，在CUDA中支持C语言的控制流，比如if…else, for ,while 等，CUDA中同样支持，但是如果一个线程束中的不同线程包含不同的控制条件，那么当我们执行到这个控制条件是就会面临不同的选择。

这里要讲一下CPU了，当我们的程序包含大量的分支判断时，从程序角度来说，程序的逻辑是很复杂的，因为一个分支就会有两条路可以走，如果有10个分支，那么一共有1024条路走，CPU采用流水线化作业，如果每次等到分支执行完再执行下面的指令会造成很大的延迟，所以现在处理器都采用**分支预测技术**，而CPU的这项技术相对于gpu来说高级了不止一点点，而这也是GPU与CPU的不同，设计初衷就是为了解决不同的问题。CPU适合逻辑复杂计算量不大的程序，比如操作系统，控制系统，GPU适合大量计算简单逻辑的任务，所以被用来算数。

如下一段代码：

```C
if (cond) {
    //do something
} else {
    //do something
}
```

假设这段代码是核函数的一部分，那么当一个线程束的32个线程执行这段代码的时候，如果其中16个执行if中的代码段，而另外16个执行else中的代码块，**同一个线程束中的线程，执行不同的指令，这叫做线程束的分化**。在每个指令周期，线程束中的所有线程执行相同的指令，但是线程束又是分化的，所以这似乎是相悖的，但是事实上这两个可以不矛盾。

解决矛盾的办法就是每个线程都执行所有的if和else部分，当一部分cond成立的时候，执行if块内的代码，有一部分线程cond不成立，那么他们怎么办？继续执行else？不可能的，因为分配命令的调度器就一个，所以这些cond不成立的线程等待，就像分水果，你不爱吃，那你就只能看着别人吃，等大家都吃完了，再进行下一轮（也就是下一个指令）。**线程束分化会产生严重的性能下降，条件分支越多，并行性削弱越严重**。

注意线程束分化研究的是一个线程束中的线程，不同线程束中的分支互不影响。

执行过程如下：

![warp divergence](/images/Professional%20CUDA%20C%20Programming/warp%20divergence.png)

因为线程束分化导致的性能下降就应该用线程束的方法解决，根本思路是**避免同一个线程束内的线程分化**，而让我们能控制线程束内线程行为的原因是线程块中线程分配到线程束是有规律的而不是随机的。这就使得我们**根据线程编号来设计分支**是可以的，补充说明下，当一个线程束中所有的线程都执行if，或者都执行else时，不存在性能下降；只有当线程束内有分歧产生分支的时候，性能才会急剧下降。

线程束内的线程是可以被我们控制的，那么我们就把都执行if的线程塞到一个线程束中，或者让一个线程束中的线程都执行if，另外线程都执行else的这种方式可以将效率提高很多。

下面说明线程束分化是如何导致性能下降的（chapter03/simpleDivergence.cu）：

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

下面这个kernel可以产生一个比较低效的分支：

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

这种情况下我们假设只配置一个x=64的一维线程块，那么只有两个线程束，线程束内奇数线程（threadIdx.x为奇数）会执行else，偶数线程执行if，分化很严重。

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

但是如果我们换一种方法，得到相同但是错乱的结果，这个顺序其实是无所谓的，因为我们可以后期调整。那么下面代码就会很高效，即条件 `（tid/warpSize）%2==0` 使分支粒度是线程束大小的倍数；偶数编号的线程执行if子句，奇数编号的线程执行else子句。

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

- 第一个线程束内的线程编号tid从0到31，tid/warpSize都等于0，那么就都执行if语句。
- 第二个线程束内的线程编号tid从32到63，tid/warpSize都等于1，执行else。线程束内没有分支，效率较高。
  
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
./divergence using Device 0: NVIDIA GeForce GTX 1060 6GB
Data size: 64
Execution Configure (block 64 grid 1)
warmup                  <<<1, 64>>>     elapsed 0.000034 sec 
mathKernel1             <<<1, 64>>>     elapsed 0.000009 sec 
mathKernel2             <<<1, 64>>>     elapsed 0.000008 sec 
mathKernel3             <<<1, 64>>>     elapsed 0.000007 sec 
```

代码中warmup部分是提前启动一次GPU，因为第一次启动GPU时会比第二次速度慢一些，具体原因未知，可以去查一下CUDA的相关技术文档了解内容。我们可以通过nvprof分析一下程序执行过程：

```shell
nvprof --metrics branch_efficiency ./simpleDivergence
```

然后得到下面这些参数。编写的CMakeLists禁用了分支预测功能，这样kernel1和kernel3的效率是相近的。即用kernel3的编写方式，会得到没有优化的结果如下：

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

以下是kernel3编译器不会优化的代码：

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

nvcc 在1和3上优化有限，但是也超过了50%以上的利用率

#### 2.3. 资源分配 Resource Partitioning

我们前面提到过，每个SM上执行的基本单位是线程束，也就是说，单指令通过指令调度器广播给某线程束的全部线程，这些线程同一时刻执行同一命令，当然也有分支情况，上一就节我们已经介绍了分支，这是执行的那部分，当然还有很多线程束没执行，那么这些没执行的线程束情况又如何呢？我给他们分成了两类，注意是我分的，不一定官方是不是这么讲。我们离开线程束内的角度（线程束内是观察线程行为，离开线程束我们就能观察线程束的行为了），一类是已经激活的，也就是说这类线程束其实已经在SM上准备就绪了，只是没轮到他执行，这时候他的状态叫做阻塞，还有一类可能分配到SM了，但是还没上到片上，这类我称之为未激活线程束。

而每个SM上有多少个线程束处于激活状态，取决于以下资源：

- 程序计数器
- 寄存器
- 共享内存

线程束一旦被激活来到片上，那么他就不会再离开SM直到执行结束。每个SM都有32位的寄存器组，每个架构寄存器的数量不一样，其存储于寄存器文件中，为每个线程进行分配，同时，固定数量的共享内存，在线程块之间分配。一个SM上被分配多少个线程块和线程束取决于SM中可用的寄存器和共享内存，以及内核需要的寄存器和共享内存大小。

这是一个平衡问题，就像一个固定大小的坑，能放多少萝卜取决于坑的大小和萝卜的大小，相比于一个大坑，小坑内可能放十个小萝卜，或者两个大萝卜，SM上资源也是，当kernel占用的资源较少，那么更多的线程（这是线程越多线程束也就越多）处于活跃状态，相反则线程越少。

关于寄存器资源的分配：

![allocate of register](/images/Professional%20CUDA%20C%20Programming/allocate%20of%20register.png)

![allocate of shared memory](/images/Professional%20CUDA%20C%20Programming/allocate%20of%20shared%20memory.png)

上面讲的主要是线程束，如果从逻辑上来看线程块的话，可用资源的分配也会影响常驻线程块的数量。特别是当SM内的资源没办法处理一个完整块，那么程序将无法启动，这个是我们应该找找自己的毛病，你得把内核写的多大，或者一个块有多少线程，才能出现这种情况。

以下是资源列表：

![Compute Capability3](/images/Professional%20CUDA%20C%20Programming/Compute%20Capability3.png)

当寄存器和共享内存分配给了线程块，这个线程块处于活跃状态，所包含的线程束称为活跃线程束。

活跃的线程束又分为三类：

- 选定的线程束
- 阻塞的线程束
- 符合条件的线程束

当SM要执行某个线程束的时候，执行的这个线程束叫做选定的线程束，准备要执行的叫符合条件的线程束，如果线程束不符合条件还没准备好就是阻塞的线程束。

满足下面的要求，线程束才算是符合条件的：

- 32个CUDA核心可以用于执行
- 执行所需要的资源全部就位

Kepler活跃的线程束数量从开始到结束不得大于64，可以等于。任何周期选定的线程束小于等于4。由于计算资源是在线程束之间分配的，且线程束的整个生命周期都在片上，所以线程束的上下文切换是非常快速的。下面介绍如何通过大量的活跃的线程束切换来隐藏延迟

#### 2.4. 延迟隐藏 Latency Hiding

延迟是什么，就是当你让计算机帮你算一个东西的时候**计算需要用的时间**。举个宏观的例子，比如一个算法验证，你交给计算机，计算机会让某个特定的计算单元完成这个任务，共需要十分钟，而接下来这十分钟，你就要等待，等他算完了你才能计算下一个任务，那么这十分钟计算机的利用率有可能并不是100%，也就是说他的某些功能是空闲的，你就想能不能再跑一个同样的程序不同的数据（做过机器学习的这种情况不会陌生，大家都是同时跑好几个版本）然后你又让计算机跑，这时候你发现还没有完全利用完资源，于是又继续加任务给计算机，结果加到第十分钟了，已经加了十个了，你还没加完，但是第一个任务已经跑完了，如果你这时候停止加任务，等陆陆续续的你后面加的任务都跑完了共用时20分钟，共执行了10个任务，那么平局一个任务用时 $\frac{20}{10}=2$ 分钟/任务 。 但是我们还有一种情况，因为任务还有很多，第十分钟你的第一个任务结束的时候你继续向你的计算机添加任务，那么这个循环将继续进行，那么第二十分钟你停止添加任务，等待第三十分钟所有任务执行完，那么平均每个任务的时间是： $\frac{30}{20}=1.5$ 分钟/任务，如果一直添加下去，$lim_{n\to\infty}\frac{n+10}{n}=1$ 也就是极限速度，一分钟一个，隐藏了9分钟的延迟。

当然上面的另一个重要参数是每十分钟添加了10个任务，如果每十分钟共可以添加100个呢，那么二十分钟就可以执行100个，每个任务耗时： $\frac{20}{100}=0.2$ 分钟/任务 三十分钟就是 $\frac{30}{200}=0.15$ 如果一直添加下去， $lim_{n\to\infty}\frac{n+10}{n\times 10}=0.1$ 分钟/任务。

这是理想情况，有一个必须考虑的就是虽然你十分钟添加了100个任务，可是没准添加50个计算机就满载了，这样的话 极限速度只能是：$lim_{n\to\infty}\frac{n+10}{n\times 5}=0.2$ 分钟/任务 了。

所以**最大化是要最大化硬件，尤其是计算部分的硬件满跑，都不闲着的情况下利用率是最高的**，总有人闲着，利用率就会低很多，即最大化功能单元的利用率。利用率与常驻线程束直接相关。硬件中线程调度器负责调度线程束调度，当每时每刻都有可用的线程束供其调度，这时候可以达到计算资源的完全利用，以此来保证通过其他常驻线程束中发布其他指令的，可以隐藏每个指令的延迟。

与其他类型的编程相比，GPU的延迟隐藏及其重要。对于指令的延迟，通常分为两种：

- 算术指令
- 内存指令

算数指令延迟是一个算术操作从开始，到产生结果之间的时间，这个时间段内只有某些计算单元处于工作状态，而其他逻辑计算单元处于空闲。内存指令延迟很好理解，当产生内存访问的时候，计算单元要等数据从内存拿到寄存器，这个周期是非常长的。

- 算术延迟 10~20 个时钟周期
- 内存延迟 400~800 个时钟周期

下图就是阻塞线程束到可选线程束的过程逻辑图：

![Warp Scheduler](/images/Professional%20CUDA%20C%20Programming/Warp%20Scheduler.png)

其中线程束0在阻塞两段时间后恢复可选模式，但是在这段等待时间中，SM没有闲置。

那么至少需要多少线程，线程束来保证最小化延迟呢？little法则给出了下面的计算公式: 

$$
\text{所需线程束} = \text{延迟} \times \text{吞吐量}
$$

> 注意带宽和吞吐量的区别，带宽一般指的是理论峰值，最大每个时钟周期能执行多少个指令，吞吐量是指实际操作过程中每分钟处理多少个指令。

这个可以想象成一个瀑布，像这样，绿箭头是线程束，只要线程束足够多，吞吐量是不会降低的：

![Throughput](/images/Professional%20CUDA%20C%20Programming/Throughput.png)

下面表格给出了Fermi 和Kepler执行某个简单计算时需要的并行操作数：

![Full Arithmetic Utilization](/images/Professional%20CUDA%20C%20Programming/Full%20Arithmetic%20Utilization.png)

另外有两种方法可以提高并行：

- **指令级并行(ILP):** 一个线程中有很多独立的指令
- **线程级并行(TLP):** 很多并发地符合条件的线程

同样，与指令周期隐藏延迟类似，内存隐藏延迟是靠内存读取的并发操作来完成的，需要注意的是，指令隐藏的关键目的是使用全部的计算资源，而内存读取的延迟隐藏是为了使用全部的内存带宽，内存延迟的时候，计算资源正在被别的线程束使用，所以我们不考虑内存读取延迟的时候计算资源在做了什么，这两种延迟我们看做两个不同的部门但是遵循相同的道理。

我们的根本目的是把计算资源，内存读取的带宽资源全部使用满，这样就能达到理论的最大效率。

同样下表根据Little 法则给出了需要多少线程束来最小化内存读取延迟，不过这里有个单位换算过程，机器的性能指标内存读取速度给出的是GB/s 的单位，而我们需要的是每个时钟周期读取字节数，所以要用这个速度除以频率，例如 2070 的内存带宽是144 GB/s 化成时钟周期： $\frac{144GB/s}{1.566GHz}=92 B/t$ ,这样就能得到单位时间周期的内存带宽了。

![Full Memory Utilization](/images/Professional%20CUDA%20C%20Programming/Full%20Memory%20Utilization.png)

需要说明的是这个速度不是单个SM的而是整个GPU设备的，用的内存带宽是GPU设备的而不是针对一个SM的。Fermi 需要并行的读取74的数据才能让GPU带宽满载，如果每个线程读取4个字节，我们大约需要18500个线程，大约579个线程束才能达到这个峰值。所以，延迟的隐藏取决于活动的线程束的数量，数量越多，隐藏的越好，但是线程束的数量又受到上面的说的资源影响。所以这里就需要寻找最优的执行配置来达到最优的延迟隐藏。

那么我们怎么样确定一个线程束的下界呢，使得当高于这个数字时SM的延迟能充分的隐藏，其实这个公式很简单，也很好理解，就是SM的计算核心数乘以单条指令的延迟。比如32个单精度浮点计算器，每次计算延迟20个时钟周期，那么我需要最少 32x20 =640 个线程使设备处于忙碌状态。

#### 2.5. 占用率 Occupancy

占用率是一个SM种活跃的线程束的数量，占SM最大支持线程束数量的比。前面写的程序chapter02/checkDeviceInfor.cu 中添加几个成员的查询就可以帮我们找到这个值（）。

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

占用率是每个SM中活跃的线程束占最大线程束数量的比值：

$$
\text{占用率} = \frac{\text{活动线程束数量}}{\text{最大线程束数量}}
$$

CUDA工具包中提供一个叫做UCDA占用率计算器的电子表格，填上相关数据可以帮你自动计算网格参数：

![Occupancy Calculator](/images/Professional%20CUDA%20C%20Programming/Occupancy%20Calculator.png)

上面我们已经明确内核使用寄存器的数量会影响SM内线程束的数量，nvcc的编译选项也有手动控制寄存器的使用。

也可以通过调整线程块内线程的多少来提高占用率，当然要合理不能太极端：

- 小的线程块：每个线程块中线程太少，会在所有资源没用完就达到了线程束的最大要求
- 大的线程块：每个线程块中太多线程，会导致每个SM中每个线程可用的硬件资源较少。

使用以下准则可以使程序适用于当前和将来的设备：

- 保持每个块中线程数量是线程束大小（32）的倍数
- 避免块太小：每个块至少要有128或256个线程
- 根据内核资源的需求调整块大小
- 块的数量要远远多于SM的数量，从而在设备中可以显示有足够的并行
- 通过实验得到最佳执行配置和资源使用情况

#### 2.6. 同步 Synchronization

并发程序对同步非常有用，比如pthread中的锁，openmp中的同步机制，这没做的主要目的是避免内存竞争。CUDA同步这里只讲两种：

- 线程块内同步
- 系统级别

可以调用 CUDA API 实现线程同步：

```c
cudaError_t cudaDeviceSynchronize(void);
```

块级别的就是同一个块内的线程会同时停止在某个设定的位置，用

```C
__syncthread();
```

这个函数完成，这个函数只能同步同一个块内的线程，不能同步不同块内的线程，想要同步不同块内的线程，就只能让核函数执行完成，控制程序交换主机，这种方式来同步所有线程。

内存竞争是非常危险的，一定要非常小心，这里经常出错。

#### 2.7. 可扩展性 Scalability

可扩展性其实是相对于不同硬件的，当某个程序在设备1上执行的时候时间消耗是T。当我们使用设备2时，其资源是设备1的两倍，我们希望得到T/2的运行速度，这种性质是CUDA驱动部分提供的特性，目前来说 Nvidia正在致力于这方面的优化，如下图：

![Scalability](/images/Professional%20CUDA%20C%20Programming/Scalability.png)

### 3. 并行性表现 Exposing Parallelism

本节的主要内容就是进一步理解线程束在硬件上执行的本质过程，结合上几节关于执行模型的学习，本文相对简单，通过修改核函数的配置，来观察核函数的执行速度，以及分析硬件利用数据，分析性能。调整核函数配置是CUDA开发人员必须掌握的技能，本篇只研究对核函数的配置是如何影响效率的（也就是通过网格，块的配置来获得不同的执行效率。）

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

没有任何优化的最简单的二维矩阵加法，代码在 chapter03/sumMatrix2D.cu 中。

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

这里用两个 $8192×8192$ 的矩阵相加来测试效率。注意一下这里的GPU内存，一个矩阵是 $2^{14}×2^{14}×2^2=2^{30}$ 字节 也就是 1G，三个矩阵就是 3G。 

#### 3.1. 用 nvprof 检测活跃的线程束 Checking Active Warps with nvprof

对比性能要控制变量，上面的代码只用两个变量，也就是块的x和y的大小，所以，调整x和y的大小来产生不同的效率，结果如下：

```shell
(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 32 32
CPU Execution Time elapsed 0.538640 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.090911 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 32 16
CPU Execution Time elapsed 0.548685 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.086876 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 16 32
CPU Execution Time elapsed 0.544791 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.056706 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 16 16
CPU Execution Time elapsed 0.548078 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.056472 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 16 8
CPU Execution Time elapsed 0.546093 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.086659 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 8 16
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

另外，每个机器执行此代码效果可能定不一样，所以大家要根据自己的硬件分析数据。书上给出的 M2070 就和我们的结果不同，2070 的 (32,16) 效率最高，而我们的 (16, 16) 效率最高，毕竟架构不同，而且CUDA版本不同导致了优化后的机器码差异很大，所以我们还是来看看活跃线程束的情况，使用

```shell
nvprof --metrics achieved_occupancy ./sum_matrix2D 
```

得出结果

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 32 32 
==43939== NVPROF is profiling process 43939, command: ./sum_matrix2D 32 32
CPU Execution Time elapsed 0.550530 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.096127 sec
==43939== Profiling application: ./sum_matrix2D 32 32
==43939== Profiling result:
==43939== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.728469    0.728469    0.728469

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 32 16
==44053== NVPROF is profiling process 44053, command: ./sum_matrix2D 32 16
CPU Execution Time elapsed 0.551584 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.089149 sec
==44053== Profiling application: ./sum_matrix2D 32 16
==44053== Profiling result:
==44053== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.904511    0.904511    0.904511

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 16 32
==44187== NVPROF is profiling process 44187, command: ./sum_matrix2D 16 32
CPU Execution Time elapsed 0.547609 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.070035 sec
==44187== Profiling application: ./sum_matrix2D 16 32
==44187== Profiling result:
==44187== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.817224    0.817224    0.817224

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 16 16
==44285== NVPROF is profiling process 44285, command: ./sum_matrix2D 16 16
CPU Execution Time elapsed 0.550066 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.062846 sec
==44285== Profiling application: ./sum_matrix2D 16 16
==44285== Profiling result:
==44285== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.885973    0.885973    0.885973

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 16 8
==44394== NVPROF is profiling process 44394, command: ./sum_matrix2D 16 8
CPU Execution Time elapsed 0.548652 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.092749 sec
==44394== Profiling application: ./sum_matrix2D 16 8
==44394== Profiling result:
==44394== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.968459    0.968459    0.968459

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 8 16
==44547== NVPROF is profiling process 44547, command: ./sum_matrix2D 8 16
CPU Execution Time elapsed 0.549166 sec
GPU Execution configuration<<<(2048, 1024),(8, 16)>>> Time elapsed 0.062462 sec
==44547== Profiling application: ./sum_matrix2D 8 16
==44547== Profiling result:
==44547== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.870483    0.870483    0.870483
```

|  gridDim   | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy |
|:----------:|:--------:|:------------:|:------------:|:------------------:|
|  512, 512  |  32, 32  |   0.550530   |   0.096127   |      0.728469      |
| 512, 1024  |  32, 16  |   0.551584   |   0.089149   |      0.904511      |
| 1024, 512  |  16, 32  |   0.547609   |   0.070035   |      0.817224      |
| 1024, 1024 |  16, 16  |   0.550066   |   0.062846   |      0.885973      |
| 1024, 2048 |  16, 8   |   0.548652   |   0.092749   |      0.968459      |
| 2048, 1024 |  8, 16   |   0.549166   |   0.062462   |      0.870483      |

可见活跃线程束比例高的未必执行速度快，但实际上从原理出发，应该是利用率越高效率越高，但是还受到其他因素制约。

活跃线程束比例的定义是：每个周期活跃的线程束的平均值与一个SM支持的线程束最大值的比。

#### 3.2. 用 nvprof 检测内存操作 Checking Active Warps with nvprof

下面我们继续用nvprof来看看内存利用率如何

```C
nvprof --metrics gld_throughput ./sum_matrix2D
```

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sum_matrix2D 32 32 
==44801== NVPROF is profiling process 44801, command: ./sum_matrix2D 32 32
CPU Execution Time elapsed 0.544097 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.273369 sec
==44801== Profiling application: ./sum_matrix2D 32 32
==44801== Profiling result:
==44801== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  61.836GB/s  61.836GB/s  61.836GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sum_matrix2D 32 16
==44878== NVPROF is profiling process 44878, command: ./sum_matrix2D 32 16
CPU Execution Time elapsed 0.545615 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.247466 sec
==44878== Profiling application: ./sum_matrix2D 32 16
==44878== Profiling result:
==44878== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  68.650GB/s  68.650GB/s  68.650GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sum_matrix2D 16 32
==44973== NVPROF is profiling process 44973, command: ./sum_matrix2D 16 32
CPU Execution Time elapsed 0.553040 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.244212 sec
==44973== Profiling application: ./sum_matrix2D 16 32
==44973== Profiling result:
==44973== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  34.835GB/s  34.835GB/s  34.835GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sum_matrix2D 16 16
==45123== NVPROF is profiling process 45123, command: ./sum_matrix2D 16 16
CPU Execution Time elapsed 0.545451 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.240271 sec
==45123== Profiling application: ./sum_matrix2D 16 16
==45123== Profiling result:
==45123== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  35.409GB/s  35.409GB/s  35.409GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sum_matrix2D 16 8
==45182== NVPROF is profiling process 45182, command: ./sum_matrix2D 16 8
CPU Execution Time elapsed 0.543101 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.246472 sec
==45182== Profiling application: ./sum_matrix2D 16 8
==45182== Profiling result:
==45182== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  34.444GB/s  34.444GB/s  34.444GB/s

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_throughput ./sum_matrix2D 8 16
==45295== NVPROF is profiling process 45295, command: ./sum_matrix2D 8 16
CPU Execution Time elapsed 0.545891 sec
GPU Execution configuration<<<(2048, 1024),(8, 16)>>> Time elapsed 0.240333 sec
==45295== Profiling application: ./sum_matrix2D 8 16
==45295== Profiling result:
==45295== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_throughput                    Global Load Throughput  17.701GB/s  17.701GB/s  17.701GB/s
```

|  gridDim   | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy | GLD Throughput (GB/s) |
|:----------:|:--------:|:------------:|:------------:|:------------------:|:---------------------:|
|  512, 512  |  32, 32  |   0.544097   |   0.273369   |      0.728469      |        61.836         |
| 512, 1024  |  32, 16  |   0.545615   |   0.247466   |      0.904511      |        68.650         |
| 1024, 512  |  16, 32  |   0.553040   |   0.244212   |      0.817224      |        34.835         |
| 1024, 1024 |  16, 16  |   0.545451   |   0.240271   |      0.885973      |        35.409         |
| 1024, 2048 |  16, 8   |   0.543101   |   0.246472   |      0.968459      |        34.444         |
| 2048, 1024 |  8, 16   |   0.545891   |   0.240333   |      0.870483      |        17.701         |

可以看出综合第二种配置的线程束吞吐量最大。所以可见吞吐量和线程束活跃比例一起都对最终的效率有影响。

接着看看全局加载效率，全局效率的定义是：**被请求的全局加载吞吐量占所需的全局加载吞吐量的比值（全局加载吞吐量）**，也就是说应用程序的加载操作利用了设备内存带宽的程度；注意区别吞吐量和全局加载效率的区别，这个在前面我们已经解释过吞吐量了。

```C
nvprof --metrics gld_efficiency ./sum_matrix2D
```

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sum_matrix2D 32 32
==45602== NVPROF is profiling process 45602, command: ./sum_matrix2D 32 32
CPU Execution Time elapsed 0.544926 sec
==45602== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45602== Profiling application: ./sum_matrix2D 32 32Time elapsed 1.298604 sec
==45602== Profiling result:
==45602== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sum_matrix2D 32 16
==45728== NVPROF is profiling process 45728, command: ./sum_matrix2D 32 16
CPU Execution Time elapsed 0.546795 sec
==45728== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45728== Profiling application: ./sum_matrix2D 32 16 Time elapsed 1.258507 sec
==45728== Profiling result:
==45728== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      12.50%      12.50%      12.50%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sum_matrix2D 16 32
==45829== NVPROF is profiling process 45829, command: ./sum_matrix2D 16 32
CPU Execution Time elapsed 0.549460 sec
==45829== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45829== Profiling application: ./sum_matrix2D 16 32 Time elapsed 1.238372 sec
==45829== Profiling result:
==45829== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sum_matrix2D 16 16
==45926== NVPROF is profiling process 45926, command: ./sum_matrix2D 16 16
CPU Execution Time elapsed 0.548614 sec
==45926== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==45926== Profiling application: ./sum_matrix2D 16 16> Time elapsed 1.219676 sec
==45926== Profiling result:
==45926== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sum_matrix2D 16 8
==46017== NVPROF is profiling process 46017, command: ./sum_matrix2D 16 8
CPU Execution Time elapsed 0.548084 sec
==46017== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==46017== Profiling application: ./sum_matrix2D 16 8> Time elapsed 1.277124 sec
==46017== Profiling result:
==46017== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      25.00%      25.00%      25.00%

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics gld_efficiency ./sum_matrix2D 8 16
==46086== NVPROF is profiling process 46086, command: ./sum_matrix2D 8 16
CPU Execution Time elapsed 0.545527 sec
==46086== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (2 of 2)... 
Replaying kernel "sumMatrix(float*, float*, float*, int, int)" (done)
==46086== Profiling application: ./sum_matrix2D 8 16> Time elapsed 1.219265 sec
==46086== Profiling result:
==46086== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      50.00%      50.00%      50.00%
```

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

线程块中内层的维度（blockDim.x）过小 是否对现在的设备还有影响，我们来看一下下面的试验

```shell
(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 64 2
CPU Execution Time elapsed 0.544023 sec
GPU Execution configuration<<<(256, 8192),(64, 2)>>> Time elapsed 0.356677 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 64 4
CPU Execution Time elapsed 0.544404 sec
GPU Execution configuration<<<(256, 4096),(64, 4)>>> Time elapsed 0.174845 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 64 8
CPU Execution Time elapsed 0.544168 sec
GPU Execution configuration<<<(256, 2048),(64, 8)>>> Time elapsed 0.091977 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 128 2
CPU Execution Time elapsed 0.545258 sec
GPU Execution configuration<<<(128, 8192),(128, 2)>>> Time elapsed 0.355204 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 128 4
CPU Execution Time elapsed 0.547236 sec
GPU Execution configuration<<<(128, 4096),(128, 4)>>> Time elapsed 0.176689 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 128 8
CPU Execution Time elapsed 0.545464 sec
GPU Execution configuration<<<(128, 2048),(128, 8)>>> Time elapsed 0.089984 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 256 2
CPU Execution Time elapsed 0.545916 sec
GPU Execution configuration<<<(64, 8192),(256, 2)>>> Time elapsed 0.363761 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 256 4
CPU Execution Time elapsed 0.548850 sec
GPU Execution configuration<<<(64, 4096),(256, 4)>>> Time elapsed 0.190659 sec

(DeepLearning) linxi@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03$ ./sum_matrix2D 256 8
CPU Execution Time elapsed 0.547406 sec
GPU Execution configuration<<<(64, 2048),(256, 8)>>> Time elapsed 0.000030 sec
```

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

通过这个表我们发现，块最小的反而获得最低的效率，即数据量大可能会影响结果，当数据量大的时候有可能决定时间的因素会发生变化，但是一些结果是可以观察到

- 尽管（64，4） 和 （128，2） 有同样大小的块，但是执行效率不同，说明内层线程块尺寸影响效率
- 最后的块参数无效，所有线程超过了 1024 GPU 最大限制线程数
- 尽管 (64, 2) 线程块最小，但是启动了最多的线程块，速度并不是最快的
- 综合线程块大小和数量，(128, 8) 速度最快

调整块的尺寸，还是为了增加并行性，或者说增加活跃的线程束，看看线程束的活跃比例：

```shell
root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03#  nvprof --metrics achieved_occupancy ./sum_matrix2D 64 2
==47210== NVPROF is profiling process 47210, command: ./sum_matrix2D 64 2
CPU Execution Time elapsed 0.549154 sec
GPU Execution configuration<<<(256, 8192),(64, 2)>>> Time elapsed 0.363687 sec
==47210== Profiling application: ./sum_matrix2D 64 2
==47210== Profiling result:
==47210== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.941718    0.941718    0.941718

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 64 4
==47520== NVPROF is profiling process 47520, command: ./sum_matrix2D 64 4
CPU Execution Time elapsed 0.554265 sec
GPU Execution configuration<<<(256, 4096),(64, 4)>>> Time elapsed 0.182942 sec
==47520== Profiling application: ./sum_matrix2D 64 4
==47520== Profiling result:
==47520== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.939658    0.939658    0.939658

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 64 8
==47609== NVPROF is profiling process 47609, command: ./sum_matrix2D 64 8
CPU Execution Time elapsed 0.552905 sec
GPU Execution configuration<<<(256, 2048),(64, 8)>>> Time elapsed 0.100848 sec
==47609== Profiling application: ./sum_matrix2D 64 8
==47609== Profiling result:
==47609== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.912401    0.912401    0.912401

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 128 2
==47706== NVPROF is profiling process 47706, command: ./sum_matrix2D 128 2
CPU Execution Time elapsed 0.554928 sec
GPU Execution configuration<<<(128, 8192),(128, 2)>>> Time elapsed 0.361216 sec
==47706== Profiling application: ./sum_matrix2D 128 2
==47706== Profiling result:
==47706== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.842183    0.842183    0.842183

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 128 4
==47822== NVPROF is profiling process 47822, command: ./sum_matrix2D 128 4
CPU Execution Time elapsed 0.555749 sec
GPU Execution configuration<<<(128, 4096),(128, 4)>>> Time elapsed 0.182397 sec
==47822== Profiling application: ./sum_matrix2D 128 4
==47822== Profiling result:
==47822== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.833157    0.833157    0.833157

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 128 8
==47928== NVPROF is profiling process 47928, command: ./sum_matrix2D 128 8
CPU Execution Time elapsed 0.550801 sec
GPU Execution configuration<<<(128, 2048),(128, 8)>>> Time elapsed 0.099784 sec
==47928== Profiling application: ./sum_matrix2D 128 8
==47928== Profiling result:
==47928== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.732285    0.732285    0.732285

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 256 2
==48042== NVPROF is profiling process 48042, command: ./sum_matrix2D 256 2
CPU Execution Time elapsed 0.550500 sec
GPU Execution configuration<<<(64, 8192),(256, 2)>>> Time elapsed 0.369576 sec
==48042== Profiling application: ./sum_matrix2D 256 2
==48042== Profiling result:
==48042== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.804247    0.804247    0.804247

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 256 4
==48122== NVPROF is profiling process 48122, command: ./sum_matrix2D 256 4
CPU Execution Time elapsed 0.538097 sec
GPU Execution configuration<<<(64, 4096),(256, 4)>>> Time elapsed 0.197963 sec
==48122== Profiling application: ./sum_matrix2D 256 4
==48122== Profiling result:
==48122== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.791321    0.791321    0.791321

root@linxi1989:~/Devkit/Projects/CUDA/bin/chapter03# nvprof --metrics achieved_occupancy ./sum_matrix2D 256 8
==48214== NVPROF is profiling process 48214, command: ./sum_matrix2D 256 8
CPU Execution Time elapsed 0.549278 sec
GPU Execution configuration<<<(64, 2048),(256, 8)>>> Time elapsed 0.000024 sec
==48214== Profiling application: ./sum_matrix2D 256 8
==48214== Profiling result:
No events/metrics were profiled.
```

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

可见最高的利用率没有最高的效率。没有任何一个因素可以直接左右最后的效率，一定是大家一起作用得到最终的结果，多因一效的典型例子，于是在优化的时候，我们应该首先保证测试时间的准确性，客观性，以及稳定性。

- 大部分情况，单一指标不能优化出最优性能
- 总体性能直接相关的是内核的代码本质（内核才是关键）
- 指标与性能之间选择平衡点
- 从不同的角度寻求指标平衡，最大化效率
- 网格和块的尺寸为调节性能提供了一个不错的起点

总之，用CUDA就是为了高效，而研究这些指标是提高效率最快的途径（当然内核算法提升空间更大）

### 4. 避免分支分化 Avoiding Branch Divergence

#### 4.1. 并行规约问题 The Parallel Reduction Problem

在串行编程中，最最最常见的一个问题就是一组特别多数字通过计算变成一个数字，比如加法，也就是求这一组数据的和，或者乘法，这种计算当有如下特点的时候，可以用并行归约的方法处理他们：

- 结合性
- 交换性

对应的加法或者乘法就是交换律和结合律，所以对于所有有这两个性质的计算，都可以使用归约式计算。归约是一种常见的计算方式（串并行都可以），每次迭代计算方式都是相同的（归），从一组多个数据最后得到一个数（约）。归约的方式基本包括如下几个步骤：

1. 将输入向量划分到更小的数据块中
2. 用一个线程计算一个数据块的部分和
3. 对每个数据块的部分和再求和得到最终的结果。

数据分块保证我们可以用一个线程块来处理一个数据块。一个线程处理更小的块，所以一个线程块可以处理一个较大的块，然后多个块完成整个数据集的处理。最后将所有线程块得到的结果相加，就是结果，这一步一般在cpu上完成。

归约问题最常见的加法计算是把向量的数据分成对，然后用不同线程计算每一对元素，得到的结果作为输入继续分成对，迭代的进行，直到最后一个元素。成对的划分常见的方法有以下两种：

1. **相邻配对：** 元素与他们相邻的元素配对

![Neighbored pair](/images/Professional%20CUDA%20C%20Programming/Neighbored%20pair.png)

2. **交错配对：** 元素与一定距离的元素配对

![Interleaved pair](/images/Professional%20CUDA%20C%20Programming/Interleaved%20pair.png)

图中将两种方式表现的很清楚了，可以用代码实现以下。首先是cpu版本实现交错配对归约计算的代码：

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

和书上的代码有些不同，因为书上的代码没有考虑数组长度非2的整数幂次的结果。所以加了一个处理奇数数组最后一个无人配对的元素的处理。这个加法运算可以改成任何满足结合律和交换律的计算，比如乘法，求最大值等。通过不同的配对方式，不同的数据组织来看CUDA的执行效率。

#### 4.2. 并行规约中的分化 Divergence in Parallel Reduction

**线程束分化**已经明确说明了，有判断条件的地方就会产生分支，比如 if 和 for 这类关键词。如下图所表示的那样，对相邻元素配对进行内核实现的流程描述：

![Parallel Reduction](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction.png)

- **第一步：** 是把这个一个数组分块，每一块只包含部分数据，如上图那样（图中数据较少，但是我们假设一块上只有这么多。），我们假定这是线程块的全部数据
- **第二步：** 就是每个线程要做的事，橙色圆圈就是每个线程做的操作，可见线程threadIdx.x=0 的线程进行了三次计算，奇数线程一致在陪跑，没做过任何计算，但是根据3.2中介绍，这些线程虽然什么都不干，但是不可以执行别的指令，4号线程做了两步计算，2号和6号只做了一次计算。
- **第三步：** 将所有块得到的结果相加，就是最终结果

这个计算划分就是最简单的并行规约算法，完全符合上面我们提到的三步走的套路。值得注意的是，我们每次进行一轮计算（黄色框，这些操作同时并行）的时候，部分全局内存要进行一次修改，但只有部分被替换，而不被替换的，也不会在后面被使用到，如蓝色框里标注的内存，就被读了一次，后面就完全没有人管了。

值得注意的是，我们每次进行一轮计算（黄色框，这些操作同时并行）的时候，部分全局内存要进行一次修改，但只有部分被替换，而不被替换的，也不会在后面被使用到，如蓝色框里标注的内存，就被读了一次，后面就不会处理了。下面是内核代码：

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

原因还是能从图上找到，我们的每一轮操作都是并行的，但是不保证所有线程能同时执行完毕，所以需要等待，执行的快的等待慢的，这样就能避免块内的线程竞争内存了。被操作的两个对象之间的距离叫做跨度，也就是变量stride，完整的执行逻辑如下:

![stride](/images/Professional%20CUDA%20C%20Programming/stride.png)

注意主机端和设备端的分界，注意设备端的数据分块，完整代码在 chapter03/reduceInteger.cu：

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

下面两个是经过优化的写法。warmup 是为了启动gpu防止首次启动计算时gpu的启动过程耽误时间，影响效率测试，warmup 的代码就是 reducneighbored 的代码，可见还是有微弱的差别的。下面两个是经过优化的代码。

#### 4.3. 改善并行规约的分化 Improving Divergence in Parallel Reduction

上面归约显然是最原始的，未经过优化的东西是不能拿出去使用的，或者说一个真理是，不可能一下子就写出来满意的代码。

```C
if ((tid % (2 * stride)) == 0)
```

这个条件判断给内核造成了极大的分支，如图所示：

![Parallel Reduction](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction2.png)

第一轮有 $\frac {1}{2}$ 的线程没用
第二轮有 $\frac {3}{4}$ 的线程没用
第三轮有 $\frac {7}{8}$ 的线程没用

对于上面的低利用率，我们想到了下面这个方案来解决：

![Parallel Reduction3](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction3.png)

注意橙色圆形内的标号是线程符号，这样的计算线程的利用率是高于原始版本的，核函数如下：

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
		int index = 2 * stride *tid;
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
int index = 2 * stride *tid;
```

这一步保证index能够向后移动到有数据要处理的内存位置，而不是简单的用tid对应内存地址，导致大量线程空闲。那么这样做的效率高在哪？

首先我们保证在一个块中前几个执行的线程束是在接近满跑的，而后半部分线程束基本是不需要执行的，当一个线程束内存在分支，而分支都不需要执行的时候，硬件会停止他们调用别人，这样就节省了资源，所以高效体现在这，如果还是所有分支不满足的也要执行，即便整个线程束都不需要执行的时候，那这种方案就无效了，还好现在的硬件比较智能

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

我们前面一直在介绍一个叫做nvprof的工具，那么我们现在就来看看，每个线程束上执行指令的平均数量

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

指标结果是原始内核 886.6 比新内核 386.9 可见原始内核中有很多分支指令被执行，而这些分支指令是没有意义的。

分化程度越高，inst_per_warp这个指标会相对越高。这个大家要记一下，以后测试效率的时候会经常使用。

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

新内核，内存效率要高很多，也接近一倍了，原因还是我们上面分析的，一个线程块，前面的几个线程束都在干活，而后面几个根本不干活，不干活的不会被执行，而干活的内存请求肯定很集中，最大效率的利用带宽，而最naive的内核，不干活的线程也在线程束内跟着跑，又不请求内存，所以内存访问被打碎，理论上是只有一半的内存效率，测试来看非常接近。

#### 4.4. 交错配对的规约 Reducing with Interleaved Pairs

上面的套路是修改线程处理的数据，使部分线程束最大程度利用数据，接下来采用同样的思想，但是方法不同，接下来我们使用的方法是调整跨度，也就是我们每个线程还是处理对应的内存的位置，但内存对不是相邻的了，而是隔了一定距离的：

![Parallel Reduction4](/images/Professional%20CUDA%20C%20Programming/Parallel%20Reduction4.png)

我们依然把上图当做一个完整的线程块，那么前半部分的线程束依然是最大负载在跑，后半部分的线程束也是啥活不干

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

如果单从优化原理的角度，这个内核和前面的内核应该是相同效率的，但是测试结果是，这个新内核比前面的内核速度快了不少，所以还是考察一下指标吧：

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

reduceInterleaved内存效率居然是最低的，但是线程束内分化却是最小的。而书中说reduceInterleaved 的优势在内存读取，而非线程束分化，我们实际操作却得出了完全不同结论，到底是内存的无情，还是编译器的绝望，请看我们下个系列，到时候我们会直接研究机器码，来确定到底是什么影响了看似类似，却有结果悬殊的两个内核

**此处需要查看机器码，确定两个内核的实际不同**。

### 5. 展开循环 Unrolling Loops

像前面讲解执行模型和线程束的时候，明确的指出，GPU没有分支预测能力，所有每一个分支他都是执行的，所以在内核里尽量别写分支，分支包括啥，包括if当然还有for之类的循环语句。

举例：

```C
for (itn i = 0; i < tid; i++) {  
    // to do something
}
```

如果上面这段代码出现在内核中，就会有分支，因为一个线程束第一个线程和最后一个线程 tid 相差 32（如果线程束大小是32的话） 那么每个线程执行的时候，for 终止时完成的计算量都不同，这就需要等待，这也就产生了分支。

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

修改循环体的内容，把本来循环自己搞定的东西，我们自己列出来了，即手动展开循环，循环每次增加 4，并在每次迭代中处理 4 个元素。这样做的好处，从串行较多来看是**减少了条件判断的次数提升性能**。但是如果你把这段代码拿到机器上跑，其实看不出来啥效果，因为现代编译器把上述两个不同的写法，编译成了类似的机器语言，也就是，我们这不循环展开，编译器也会帮我们做。  

不过值得注意的是：**目前CUDA的编译器还不能帮我们做这种优化，人为的展开核函数内的循环，能够非常大的提升内核性能**

在CUDA中展开循环的目的还是那两个：

1.  减少指令消耗
2.  增加更多的独立调度指令  

如果这种指令：

```C
a[i + 0] = b[i + 0] + c[i + 0];
a[i + 1] = b[i + 1] + c[i + 1];
a[i + 2] = b[i + 2] + c[i + 2];
a[i + 3] = b[i + 3] + c[i + 3];
```

被添加到CUDA流水线上，是非常受欢迎的，因为其能最大限度的提高指令和内存带宽。

#### 5.1. 展开的归约 Reducing with Unrolling

在 [4. 避免分支分化 Avoiding Branch Divergence](#4.%20避免分支分化%20Avoiding%20Branch%20Divergence) 中，内核函数 reduceInterleaved 核函数中每个线程块只处理对应那部分的数据，我们现在的一个想法是能不能用一个线程块处理多块数据，其实这是可以实现的，如果在对这块数据进行求和前（因为总是一个线程对应一个数据）使用每个线程进行一次加法，从别的块取数据，相当于先做一个向量加法，然后再归约，这样将会用一句指令，完成之前一般的计算量，这个性价比看起来太诱人了。

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

这就是第二句，第四句指令的意思，我们只处理红色的线程块，而旁边白色线程块我们用

```C
if (idx + blockDim.x < n) {
    g_idata[idx] += g_idata[idx + blockDim.x];
}
```

处理掉了，注意我们这里用的是一维线程，也就是说，我们用原来的一半的块的数量，而每一句只添加一小句指令的情况下，完成了原来全部的计算量，这个效果应该是客观的，所以我们来看一下效果之前先看一下调用核函数的部分：

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

相比于上一节的效率，“高到不知道哪里去了”（总能引用名人名言），比最简单的归约算法快了三倍，warmup的代码，不需要理睬。  

我们上面框里有2，4，8三种尺度的展开，分别是一个块计算2个块，4个块和8个块的数据，对应的调用代码也需要修改（chapter03/reduceUnrolling.cu）

可见直接展开对效率影响非常大，不光是节省了多于的线程块的运行，而且更多的独立内存加载/存储操作会产生更好的性能，更好的隐藏延迟。下面我们看一下他们的吞吐量：

```C
nvprof --metrics dram_read_throughput ./reduceUnrolling
```

```shell

```

可见执行效率是和内存吞吐量是呈正相关的

#### 5.2. 展开线程的归约 Reducing with Unrolled Warps

接着我们的目标是最后那32个线程，因为归约运算是个倒金字塔，最后的结果是一个数，所以每个线程最后64个计算得到一个数字结果的过程，没执行一步，线程的利用率就降低一倍，因为从64到32，然后16。。这样到1的，我们现在想个办法，展开最后的6步迭代（64，32，16，8，4，2，1）使用下面的核函数来展开最后6步分支计算：

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

在 unrolling8 的基础上，我们对于tid在 $[0,32]$ 之间的线程用这个代码展开

```C
volatile int* vmem = idata;
vmem[tid] += vmem[tid + 32];
vmem[tid] += vmem[tid + 16];
vmem[tid] += vmem[tid + 8];
vmem[tid] += vmem[tid + 4];
vmem[tid] += vmem[tid + 2];
vmem[tid] += vmem[tid + 1];
```

第一步定义 volatile int 类型变量我们先不说，我们先把最后这个展开捋顺一下，当只剩下最后下面三角部分，从64个数合并到一个数，首先将前32个数，按照步长为32，进行并行加法，前32个tid得到64个数字的两两和，存在前32个数字中。接着，到了我们的关键技巧了，这32个数加上步长为16的变量，理论上，这样能得到16个数，这16个数的和就是最后这个块的归约结果，但是根据上面 tid<32 的判断条件线程 tid 16到31的线程还在运行，但是结果已经没意义了，这一步很重要（这一步可能产生疑惑的另一个原因是既然是同步执行，会不会比如线程17加上了线程33后写入17号的内存了，这时候1号才来加17号的结果，这样结果就不对了，因为我们的CUDA内核从内存中读数据到寄存器，然后进行加法都是同步进行的，也就是17号线程和1号线程同时读33号和17号的内存，这样17号即便在下一步修改，也不影响1号线程寄存器里面的值了），虽然32以内的tid的线程都在跑，但是没进行一步，后面一半的线程结果将没有用途了，  

这样继续计算，得到最后的一个有效的结果就是 $tid[0]$。  

上面这个过程有点复杂，但是我们自己好好想一想，从硬件取数据，到计算，每一步都分析一下，就能得到实际的结果。

volatile int类型变量是控制变量结果写回到内存，而不是存在共享内存，或者缓存中，因为下一步的计算马上要用到它，如果写入缓存，可能造成下一步的读取会读到错误的数据

你可能不明白

```C
vmem[tid] += vmem[tid + 32];
vmem[tid] += vmem[tid + 16];
```

tid+16 要用到 tid+32 的结果，会不会有其他的线程造成内存竞争，答案是不会的，因为一个线程束，执行的进度是完全相同的，当执行 tid+32的时候，这32个线程都在执行这步，而不会有任何本线程束内的线程会进行到下一句，详情请回忆CUDA执行模型（因为CUDA编译器是激进的，所以我们必须添加volatile，防止编译器优化数据传输而打乱执行顺序）。  

然后我们就得到结果了，看看时间：

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

又往后退了一位，看起来还是很爽的。  
这个展开还有一个节省时间的部分就是减少了5个线程束同步指令 `__syncthreads()`; 这个指令被我们减少了5次，这个也是非常有效果的。我们来看看阻塞减少了多少  

使用命令

```shell
nvprof --metrics stall_sync ./reduceUnrolling
```

```shell

```

哈哈哈，又搞笑了，书上的结果和运行结果又不一样，展开后的stall_sync 指标反而高了，也就是说之前有同步指令的效率更高，哈哈，无解。。可以把锅甩给CUDA编译器

#### 5.3. 完全展开的归约 Reducing with Complete Unrolling

根据上面展开最后64个数据，我们可以直接就展开最后128个，256个，512个，1024个：  

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

内核代码如上，这里用到了tid的大小，和最后32个没用到tid不同的是，这些如果计算完整会有一半是浪费的，而最后32个已经是线程束最小的大小了，所以无论后面的数据有没有意义，那些进程都不会停。  

每一步进行显示的同步，然后我们看结果，哈哈，又又又搞笑了：

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

似乎速度根本没什么影响，所以我觉得是编译器的锅没错了！它已经帮我们优化这一步了。

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

这一步比较应该是多余的，因为 blockDim.x 自内核启动时一旦确定，就不能更改了，所以模板函数帮我解决了这个问题，当编译时编译器会去检查，blockDim.x 是否固定，如果固定，直接可以删除掉内核中不可能的部分也就是上半部分，下半部分是要执行的，比如blockDim.x=512，代码最后生成机器码的就是如下部分：

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

删掉了不可能的部分。  

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

结果是，居然还慢了一些。。书上不是这么说的。。编译器的锅！

加载效率存储效率：

```C
nvprof --metrics gld_efficiency,gst_efficiency ./reduceUnrolling
```

下表概括了本节提到的所有并行归约实现的结果：

| 算法                      | 时间     | 加载效率 | 存储效率 |
| ------------------------- | -------- | -------- | -------- |
| 相邻无分化（上一篇）      | |   | 25.00%   |
| 相邻分化（上一篇）        |  | 25.01%   | 25.00%   |
| 交错（上一篇）            | 0.004956 | 98.04%   | 97.71%   |
| 展开8                     | 0.001294 | 99.60%   | 99.71%   |
| 展开8+最后的展开          | 0.001009 | 99.71%   | 99.68%   |
| 展开8+完全展开+最后的展开 | 0.001001 | 99.71%   | 99.68%   |
| 模板上一个算法            | 0.001008 | 99.71%   | 99.68%   |

虽然和书上结果不太一样，但是指标和效率关系还是很明显的，所以我们今天得出的结论是。。一步一步优化，如果改了代码没效果，那么锅是编译器的！

### 6. 动态并行

本文作为第三章CUDA执行模型的最后一篇介绍动态并行，书中关于动态并行有一部分嵌套归约的例子，但是我认为，这个例子应该对我们用途不大，首先它并不能降低代码复杂度，其次，其运行效率也没有提高，动态并行，相当于串行编程的中的递归调用，递归调用如果能转换成迭代循环，一般为了效率的时候是要转换成循环的，只有当效率不是那么重要，而更注重代码的简洁性的时候，我们才会使用，所以我们本文只介绍简单的一些基础知识，如果需要使用动态并行相关内容的同学，请查询文档或更专业的博客。  

到目前为止，我们所有的内核都是在主机线程中调用的，那么我们肯定会想，是否我们可以在内核中调用内核，这个内核可以是别的内核，也可以是自己，那么我们就需要动态并行了，这个功能在早期的设备上是不支持的。  

动态并行的好处之一就是能让复杂的内核变得有层次，坏处就是写出来的程序更复杂，因为并行行为本来就不好控制。动态并行的另一个好处是等到执行的时候再配置创建多少个网格，多少个块，这样就可以动态的利用GPU硬件调度器和加载平衡器了，通过动态调整，来适应负载。并且在内核中启动内核可以减少一部分数据传输消耗。

#### 6.1. 嵌套执行 Nested Execution
 
前面我们大费周章的其实也就只学了，网格，块，和启动配置，以及一些线程束的知识，现在我们要做的是从内核中启动内核。  

内核中启动内核，和cpu并行中有一个相似的概念，就是父线程和子线程。子线程由父线程启动，但是到了GPU，这类名词相对多了些，比如父网格，父线程块，父线程，对应的子网格，子线程块，子线程。子网格被父线程启动，且必须在对应的父线程，父线程块，父网格结束之前结束。所有的子网格结束后，父线程，父线程块，父网格才会结束。

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250126225006.png)

上图清晰地表明了父网格和子网格的使用情况，一种典型的执行方式：

> 主机启动一个网格（也就是一个内核）-> 此网格（父网格）在执行的过程中启动新的网格（子网格们）->所有子网格们都运行结束后-> 父网格才能结束，否则要等待

如果调用的线程没有显示同步启动子网格，那么运行时保证，父网格和子网格隐式同步。  
图中显式的同步了父网格和子网格，通过设置栅栏的方法。  

父网格中的不同线程启动的不同子网格，这些子网格拥有相同的父线程块，他们之间是可以同步的。线程块中所有的线程创建的所有子网格完成之后，线程块执行才会完成。如果块中的所有线程在子网格完成前退出，那么子网格隐式同步会被触发。隐式同步就是虽然没用同步指令，但是父线程块中虽然所有线程都执行完毕，但是依旧要等待对应的所有子网格执行完毕，然后才能退出。  

前面我们讲过隐式同步，比如cudaMemcpy就能起到隐式同步的作用，但是主机内启动的网格，如果没有显式同步，也没有隐式同步指令，那么cpu线程很有可能就真的退出了，而你的gpu程序可能还在运行，这样就非常尴尬了。父线程块启动子网格需要显示的同步，也就是说不通的线程束需要都执行到子网格调用那一句，这个线程块内的所有子网格才能依据所在线程束的执行，一次执行。  

接着是最头疼的内存，内存竞争对于普通并行就很麻烦了，现在对于动态并行，更麻烦，主要的有下面几点：

1.  父网格和子网格共享相同的全局和常量内存。
2.  父网格子网格有不同的局部内存
3.  有了子网格和父网格间的弱一致性作为保证，父网格和子网格可以对全局内存并发存取。
4.  有两个时刻父网格和子网格所见内存一致：子网格启动的时候，子网格结束的时候
5.  共享内存和局部内存分别对于线程块和线程来说是私有的
6.  局部内存对线程私有，对外不可见。

#### 6.2. 在GPU上嵌套 Hello World    Nested Hello World on the GPU

为了研究初步动态并行，我们先来写个Hello World进行操作，代码如下：

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

这个程序的功能如下：  

- 第一层： 有多个线程块，执行输出，然后在 `tid==0` 的线程，启动子网格，子网格的配置是当前的一半，包括线程数量，和输入参数 iSize。  
- 第二层： 有很多不同的子网格，因为我们上面多个不同的线程块都启动了子网格，我们这里只分析一个子网格，执行输出，然后在 `tid==0` 的子线程，启动子网格，子网格的配置是当前的一半，包括线程数量，和输入参数 iSize。  
- 第三层： 继续递归下去，直到 `iSize==0` 结束。

编译的命令与之前有些不同，工程中使用cmake管理，可在 CMakeLists.txt 中查看：

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

可见，当多层调用子网格的时候，同一家的（就是用相同祖宗线程的子网）是隐式同步的，而不同宗的则是各跑各的。

---

## 参考引用

### 书籍出处


- [CUDA C编程权威指南](asset/CUDA%20&%20GPU%20Programming/CUDA%20C编程权威指南.pdf)
- [Professional CUDA C Programming](asset/CUDA%20&%20GPU%20Programming/Professional%20CUDA%20C%20Programming.pdf)

### 网页链接

- [人工智能编程 | 谭升的博客](https://face2ai.com/program-blog/)