# CUDA编程

## 1 简介 Introduction

### 1.1 并行计算 Parallel Computing

我们的计算机从最早的埃尼阿克到现在的各种超算，都是为了应用而产生的，软件和硬件相互刺激而相互进步，并行计算也是这样产生的。最早的计算机肯定不是并行的，但是可以做成多线程的，因为当时一个CPU只有一个核，所以不可能一个核同时执行两个计算，后来的应用逐步要求计算量越来越高，所以单核的计算速度也在逐步上升，后来大规模并行应用产生了，我们迫切的需要能够同时处理很多数据的机器，比如图像处理，以及处理大规模的同时访问的服务器后台。

并行计算其实设计到两个不同的技术领域：

- 计算机架构（硬件）
- 并行程序设计（软件）
  
这两个很好理解，一个是生产工具，一个用工具产生各种不同应用。硬件主要的目标就是为软件提供更快的计算速度，更低的性能功耗比，硬件结构上支持更快的并行。软件的主要目的是使用当前的硬件压榨出最高的性能，给应用提供更稳定快速的计算结果。

传统的计算机结构一般是哈佛体系结构（后来演变出冯·诺依曼结构）主要分成三部分：

- **内存（指令内存，数据内存）**
- **中央处理单元（控制单元和算数逻辑单元）**
- **输入、输出接口**

![Computer architecture](/images/Model_Deployment/Computer%20architecture.png)

写并行程序主要是分解任务，我们一般把一个程序看成是指令和数据的组合，当然并行也可以分为这两种：

- **指令并行**
- **数据并行**

我们的任务更加关注数据并行，所以我们的主要任务是分析数据的相关性，哪些可以并行，哪些不能不行。

我们研究的是大规模数据计算，计算过程比较单一（不同的数据基本用相同的计算过程）但是数据非常多，所以我们主要是数据并行，分析好数据的相关性，决定了我们的程序设计。CUDA非常适合数据并行

数据并行程序设计，第一步就是把数据依据线程进行划分

1. **块划分**，把一整块数据切成小块，每个小块随机的划分给一个线程，每个块的执行顺序随机

| thread | 1    | 2    | 3    | 4     | 5      |
|--------|------|------|------|-------|--------|
| block  | 1 2 3| 4 5 6| 7 8 9| 10 11 12| 13 14 15|

2. **周期划分**，线程按照顺序处理相邻的数据块，每个线程处理多个数据块，比如我们有五个线程，线程1执行块1，线程2执行块2…..线程5执行块5，线程1执行块6

| thread | 1 | 2 | 3 | 4 | 5 | 1 | 2 | 3 | 4 | 5 | 1 | 2 | 3 | 4 | 5 |
|--------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| block  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10| 11| 12| 13| 14| 15|

下面是示意图，注意颜色相同的块使用的同一个线程，从执行顺序上看如下：

![data partitioning](/images/Model_Deployment/data%20partitioning.png)

下面是数据集上的划分上看：

![data partitioning2](/images/Model_Deployment/data%20partitioning2.png)

### 1.2 计算机架构 Computer Architecture

划分不同计算机结构的方法有很多，广泛使用的一种被称为**佛林分类法（Flynn’s Taxonomy）**，他根据指令和数据进入CPU的方式分类，分为以下四类：

![Flynn’s Taxonomy](/images/Model_Deployment/Flynn’s%20Taxonomy.png)

分别以数据和指令进行分析：

- **单指令单数据SISD**（传统串行计算机，386）
- **单指令多数据SIMD**（并行架构，比如向量机，所有核心指令唯一，但是数据不同，现在CPU基本都有这类的向量指令）
- **多指令单数据MISD**（少见，多个指令围殴一个数据）
- **多指令多数据MIMD**（并行架构，多核心，多指令，异步处理多个数据流，从而实现空间上的并行，MIMD多数情况下包含SIMD，就是MIMD有很多计算核，计算核支持SIMD）

为了提高并行的计算能力，我们要从架构上实现下面这些性能提升：

- **降低延迟：** 延迟是指操作从开始到结束所需要的时间，一般用微秒计算，延迟越低越好
- **提高带宽：** 带宽是单位时间内处理的数据量，一般用 MB/s 或者 GB/s 表示
- **提高吞吐量：** 吞吐量是单位时间内成功处理的运算数量，一般用gflops来表示（十亿次浮点计算

吞吐量和延迟有一定关系，都是反应计算速度的，一个是时间除以运算次数，得到的是单位次数用的时间–延迟，一个是运算次数除以时间，得到的是**单位时间执行次数–吞吐量**。

计算机架构也可以根据内存进行划分：

- **分布式内存的多节点系统**
- **共享内存的多处理器系统**

第一个更大，通常叫做**集群**，就是一个机房好多机箱，每个机箱都有内存处理器电源等一些列硬件，通过网络互动，这样组成的就是分布式。

![clusters](/images/Model_Deployment/clusters.png)

第二个是**单个主板有多个处理器**，他们共享相同的主板上的内存，内存寻址空间相同，通过PCIe和内存互动。

![many-core](/images/Model_Deployment/many-core.png)

多个处理器可以分多片处理器，和单片多核（众核many-core），也就是有些主板上挂了好多片处理器，也有的是一个主板上就一个处理器，但是这个处理器里面有几百个核。现目前发展趋势是众合处理器，集成度更高。GPU就属于众核系统，当然现在CPU也都是多核的了，但是它们还是有很大区别的：

- CPU 适合执行复杂的逻辑，比如多分支，其核心比较重（复杂）
- GPU 适合执行简单的逻辑，大量的数据计算，其吞吐量更高，但是核心比较轻（结构简单）

### 1.3 异构计算 Heterogeneous Computing

GPU本来的任务是做图形图像的，也就是把数据处理成图形图像，图像有个特点就是并行度很高，基本上一定距离意外的像素点之间的计算是独立的，所以属于并行任务。随着时间的推移，GPU 变得越来越强大和通用，使其能够以卓越的性能和高效的能效应用于通用并行计算任务。GPU 与 CPU 之间的配合即为一种异构。
- **同构：** 使用一种或多种相同架构的处理器来执行应用程序
- **异构：** 使用一组不同的处理器架构来执行应用程序，将任务分配给最适合的架构，从而提高性能。

举例，我的服务器用的是一台 AMD 3700X CPU 加上一张 RTX3070Ti GPU 构成的服务器，GPU 插在主板的PCIe卡口上，运行程序的时候，CPU 像是一个控制者，指挥显卡完成工作后进行汇总，和下一步工作安排，所以 CPU 可以把它看做一个指挥者，主机端 host，而完成大量计算的 GPU 是我们的计算设备，设备端 device

![host and device](/images/Model_Deployment/host%20and%20device.png)

上面这张图能大致反应CPU和GPU的架构不同。

- 左图：一个四核CPU一般有四个ALU，ALU是完成逻辑计算的核心，也是我们平时说四核八核的核，控制单元，缓存也在片上，DRAM是内存，一般不在片上，CPU通过总线访问内存。
- 右图：GPU，绿色小方块是ALU，我们注意红色框内的部分SM，这一组ALU公用一个Control单元和Cache，这个部分相当于一个完整的多核CPU，但是不同的是ALU多了，control部分变小，可见计算能力提升了，控制能力减弱了，所以对于控制（逻辑）复杂的程序，一个GPU的SM是没办法和CPU比较的，但是对了逻辑简单，数据量大的任务，GPU更高效。并且，一个GPU有好多个SM，而且越来越多。

一个异构应用包含两种以上架构，所以代码也包括不止一部分：

- **主机代码：** 主机端运行，被编译成主机架构的机器码。主要是控制设备，完成数据传输等控制类工作
- **设备代码：** 在设备上执行，被编译成设备架构的机器码。主要的任务就是计算

主机端的机器码和设备端的机器码是隔离的，自己执行自己的，没办法交换执行。其实当没有 GPU 的时候 CPU 也能完成这些计算，只是速度会慢很多，所以可以把GPU看成CPU的一个**加速设备**。

衡量GPU计算能力的主要靠下面两种**容量特征**：

- **CUDA核心数量（越多越好）**
- **内存大小（越大越好）**

相应的也有计算能力的性能指标:

- **峰值计算能力**
- **内存带宽**

### 1.4 异构范例 Paradigm of Heterogeneous Computing

CPU 和 GPU 相互配合，各有所长，各有所短。低并行逻辑复杂的程序适合用 CPU，高并行逻辑简单的大数据计算适合 GPU

![GPU and CPU](/images/Model_Deployment/GPU%20and%20CPU.png)

一个程序可以进行如下分解，串行部分和并行部分：

![Parallel and Sequence](/images/Model_Deployment/Parallel%20and%20Sequence.png)

CPU和GPU线程的区别：

- CPU线程是**重量级实体**，操作系统交替执行线程，线程上下文切换花销很大
- GPU线程是**轻量级的**，GPU应用一般包含成千上万的线程，多数在排队状态，线程之间切换基本没有开销。
- CPU的核被设计用来尽可能减少一个或两个线程运行时间的延迟，而GPU核则是大量线程，最大幅度提高吞吐量

### 1.5 CUDA Hello World

```C
#include<stdio.h>

__global__ void hello_world(void) {
    printf("GPU: Hello world!\n");
}

int main(int argc,char **argv) {
  printf("CPU: Hello world!\n");
  hello_world<<<1,10>>>();
  cudaDeviceReset();//if no this line ,it can not output hello world from gpu
  return 0;
}
```
简单介绍其中几个关键字

```
__global__
```
是告诉编译器这个是个可以在设备上执行的**核函数**

```
hello_world<<<1,10>>>();
```
这句话C语言中没有’<<<>>>’是对设备进行配置的参数，也是CUDA扩展出来的部分。在调用时需要用<<<grid, block>>>来指定kernel要执行的线程数量

```
cudaDeviceReset();
```
这句话如果没有，则不能正常的运行，因为这句话包含了**隐式同步**，GPU 和 CPU 执行程序是异步的，核函数调用后成立刻会到主机线程继续，而不管GPU端核函数是否执行完毕，所以上面的程序就是GPU刚开始执行，CPU已经退出程序了，所以我们要等GPU执行完了，再退出主机线程。

一般CUDA程序分成下面这些步骤：

- 分配host内存，并进行数据初始化；
- 分配device内存，并从host将数据拷贝到device上；
- 调用CUDA的核函数在device上完成指定的运算；
- 将device上的运算结果拷贝到host上；
- 释放device和host上分配的内存。

## 2 CUDA 编程模型

### 2.1 概念 Concepts

**内核**

CUDA C++ 通过允许程序员定义称为kernel的 C++ 函数来扩展 C++，当调用内核时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

使用 `__global__ ` 声明说明符定义内核，并使用新的 `<<<...>>>` 执行配置（execution configuration）语法指定内核调用时的 CUDA 线程数。每个执行内核的线程都有一个唯一的线程 ID，可以通过内置变量在内核中访问。

**线程层次**

为方便起见，threadIdx 是一个 **3分量(3-component)向量**，因此可以使用一个一维、二维或三维的 **线程索引(thread index)** 来识别线程，形成一个具有一个维度、两个维度或三个维度的、由线程组成的块，我们称之为**线程块(thread block)**。 这提供了一种自然的方法来对某一范围（例如向量、矩阵或空间）内的元素进行访问并调用计算。

- threadldx.[x y z]: 执行当前kernel函数的线程在block中的索引值
- blockldx.[x y z]: 执行当前kernel函数的线程所在block, 在grid中的索引值
- blockDim.[x y z]: 表示一个block中包含多少个线程
- gridDim.[x y z]: 表示一个grid中包含多少个block

### 2.2 CUDA 编程模型概述 CUDA Programming Structure

![CUDA Model](/images/Model_Deployment/CUDA%20Model.png)

其中Communication Abstraction是编程模型和编译器，库函数之间的分界线。编程模型可以理解为，我们要用到的语法，内存结构，线程结构等这些我们写程序时我们自己控制的部分，这些部分控制了异构计算设备的工作模式，都是属于**编程模型**。

从宏观上我们可以从以下几个环节完成CUDA应用开发：

- 领域层
- 逻辑层
- 硬件层

第一步就是在领域层（也就是你所要解决问题的条件）分析数据和函数，以便在并行运行环境中能正确，高效地解决问题。当分析设计完程序就进入了编程阶段，我们关注点应转向如何组织并发进程，这个阶段要从逻辑层面思考。CUDA 模型主要的一个功能就是线程层结构抽象的概念，以允许控制线程行为。这个抽象为并行变成提供了良好的可扩展性（这个扩展性后面有提到，就是一个CUDA程序可以在不同的GPU机器上运行，即使计算能力不同）。在硬件层上，通过理解线程如何映射到机器上，能充分帮助我们提高性能。

CUDA程序的典型处理流程遵循如下：

- 将数据从CPU内存复制到GPU内存。
- 调用核函数在GPU内存中对数据进行操作。
- 将数据从GPU内存复制回CPU内存。

一个完整的CUDA应用可能的执行顺序如下图：

![Process Procedure](/images/Model_Deployment/Process%20Procedure.png)

### 2.3 CUDA 全局索引计算方式 Index Computing

CUDA 线程全局索引的计算，是很容易混淆的概念，因为 CUDA 线程模型的矩阵排布和通常的数学矩阵排布秩序不太一样。基本关系是 Thread 在一起组成了 Block，Block 在一起组成了 Grid，所以是 Grid 包含 Block 再包含 Thread 的关系，如下图所示：

![Grids and Blocks](/images/Model_Deployment/Grids%20and%20Blocks.png)

在上面的图中，一个 Grid 中是包含了 6 个线程块，而每个线程块又包含了 15 个线程，其中 Grid 最大允许网格大小为 $2^{31} -1$ (针对一维网格情况), Block 最大允许线程块大小为 1024。

线程是 CUDA 编程中的最小单位，实际上线程分块是逻辑上的划分，在物理上线程不分块。在调用 GPU 的时候，核函数中是允许开很多线程的，开的线程个数可以远高于 GPU 计算核心的数量。在设计总的线程数时，至少需要等于硬件的计算核心数，这样才可能充分发挥硬件的计算能力。

在实际 CUDA 编程中，代码中是使用 <<<grid_size, block_size>>> 来进行配置线程的，其中 grid_size 是用来配置 block 的大小，而 block_size 是用来配置 thread 的大小。grid_size 通过 gridDim.x 来配置，取值范围 [0 ~ gridDim.x - 1] ; block_size 通过 blockDim.x 来配置，取值范围 [0 ~ blockDim.x - 1]，核函数调用如下：

```C
kernel_fun<<<grid_size, block_size>>>();
```

具体来说，比如定义一个 2 x 3 x 1 的网格、6 x 2 x 2 的线程块，可以这么写:

```C
dim3 grid_size(2, 3);  // or dim3 grid_size(2, 3, 1);
dim3 block_size(6, 2, 2);

kernel_ful<<<grid_size, block_size>>>();
```

#### 2.3.1 一维线程计算

如下图所示，共有32个数据（位于32个方格内）需要处理，如何确定红色方框数据所在线程的位置?

![CUDA threadCounts](/images/Model_Deployment/CUDA%20threadCounts.jpg)

由概念部分，因为每一个线程块共有八个线程，所以 blockDim.x =8；由于红框数据位于第二个线程中（线程从0开始计数），所以 blockIdx.x = 2；又由于红框数据在第二个线程中位于五号位置（同样也是从0开始计数），所以 threadIdx.x = 5；

因此所求的红框数据应位于21号位置，计算如下：

```C
int index = threadIdx.x + blockIdex.x * blockDim.x;
          = 5 + 2 * 8;
          = 21;
```

由此便可以确实当前线程所执行数据的位置

#### 2.3.2 多维线程计算

全部列举出来其实是有下面九种情况:

- 一维网格一维线程块；
- 一维网格两维线程块；
- 一维网格三维线程块；
- 两维网格一维线程块；
- 两维网格两维线程块；
- 两维网格三维线程块；
- 三维网格一维线程块；
- 三维网格两维线程块；
- 三维网格三维线程块；

以下图举例，计算 Thread(2, 2) 的索引值，调用核函数的线程配置代码如下:

![Grids and Blocks](/images/Model_Deployment/Grids%20and%20Blocks.png)

核函数线程配置:

```C
dim3 grid_size(3, 2);
dim3 block_size(3, 5);

kernel_fun<<<grid_size, block_size>>>();
```

```C
int blockId = blockIdx.x + blockId.y * gridDim.x;
int threadId = threadIdx.y * blockDim.x + threadIdx.x;
int id = blockId * (blockDim.x * blockDim.y) + threadId;

// 带入计算
int blockId = 1 + 1 * 3 = 4;
int threadId = 2 * 5 + 2 = 12;
int id = 4 * (3 * 5) + 12 = 72;
```

计算结果 Thread(2, 2) 为 72，符合预期

上面的九种组织情况都可以视为是 三维网格三维线程块 的情况，只是比如一维或者二维的时候，其他维度为 1 而已。若是都把它们都看成三维格式，这样不管哪种线程组织方式，都可以套用 三维网格三维线程块 的计算方式，整理如下，

```C
// 线程块索引
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
// 局部线程索引
int threadId = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
// 全局线程索引
int id = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;
```

### 2.4 核函数 Kernel Fuction

核函数就是在CUDA模型上诸多线程中运行的那段串行代码，这段代码在设备上运行，用NVCC编译，产生的机器码是GPU的机器码，所以我们写CUDA程序就是写核函数，第一步我们要确保核函数能正确的运行产生正切的结果，第二优化CUDA程序的部分，无论是优化算法，还是调整内存结构，线程结构都是要调整核函数内的代码，来完成这些优化的。

声明核函数有一个比较模板化的方法：

```C
__global__ void kernel_name(argument list);
```

在C语言函数前没有的限定符 global，CUDA C 中还有一些其他我们在C中没有的限定符，如下：

| 限定符 | 执行 | 说明 | 备注 |
|--------|------|:------:|:-----:|
| __global__ | 设备端执行 | 可以从主机通用它可以从其他设备分发上的设备 | 必须有一个void的返回类型 |
| __device__ | 设备端执行 | 设备端调用 |  |
| __host__ | 主机端执行 | 主机使用 | 可以检查 |

而且这里有个特殊的情况就是有些函数可以同时定义为 device 和 host ，这种函数可以同时被设备和主机端的代码调用，主机端代码调用函数很正常，设备端调用函数与C语言一致，但是要声明成设备端代码，告诉nvcc编译成设备机器码，同时声明主机端设备端函数，那么就要告诉编译器，生成两份不同设备的机器码。

Kernel核函数编写有以下限制

1. 只能访问设备内存
2. 必须有void返回类型
3. 不支持可变数量的参数
4. 不支持静态变量
5. 显示异步行为

并行程序中经常的一种现象：把串行代码并行化时对串行代码块for的操作，也就是把for并行化

串行：

```C
void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int i = 0; i < N; i++)
    C[i] = A[i] + B[i];
}
```

并行：

```C
__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```

### 2.5 错误处理 Handling Errors

所有编程都需要对错误进行处理，早起的编码错误，编译器会帮我们搞定，内存错误也能观察出来，但是有些逻辑错误很难发现，甚至到了上线运行时才会被发现，而且有些厉害的bug复现会很难，不总出现，但是很致命，而且CUDA基本都是异步执行的，当错误出现的时候，不一定是哪一条指令触发的，这一点非常头疼；这时候我们就需要对错误进行防御性处理了，例如我们代码库头文件里面的这个宏：

```C
#define CHECK(call) {
  const cudaError_t error = call;

  if(error!=cudaSuccess) {
      printf("ERROR: %s:%d,",__FILE__,__LINE__);
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
      exit(1);
  }
}
```

例如，您可以在以下代码中使用这个宏：

```C
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
```

如果内存拷贝或之前的异步操作导致错误，该宏会报告错误代码，打印一条可读的信息，然后停止程序。它也可以在内核调用后用于检查内核错误，例如：

```C
kernel_function<<<grid, block>>>(argument list);
CHECK(cudaDeviceSynchronize());
```

### 给核函数计时 Timing Your Kernel

使用cpu计时的方法是测试时间的一个常用办法，在写C程序的时候最多使用的计时方法是：

```C
clock_t start, finish;
start = clock();
// 要测试的部分
finish = clock();
duration = (double)(finish - start) / CLOCKS_PER_SEC;
```

必须注意的是，并行程序这种计时方式有严重问题！如果想知道具体原因，可以查询clock的源代码（c语言标准函数）。这里我们使用gettimeofday() 函数：

```C
#include <sys/time.h>
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}
```

gettimeofday是linux下的一个库函数，创建一个cpu计时器，从1970年1月1日0点以来到现在的秒数，需要头文件sys/time.h

```C
double iStart = cpuSecond();
kernel_name<<<grid, block>>>(argument list);
cudaDeviceSynchronize();
double iElaps = cpuSecond() - iStart;
```

首先iStart是cpuSecond返回一个秒数，接着执行核函数，核函数开始执行后马上返回主机线程，所以我们必须要加一个同步函数等待核函数执行完毕，如果不加这个同步函数，那么测试的时间是从调用核函数，到核函数返回给主机线程的时间段，而不是核函数的执行时间。加上 `cudaDeviceSynchronize()` 函数后，计时是从调用核函数开始，到核函数执行完并返回给主机的时间段，下面图大致描述了执行过程的不同时间节点：

![Kernal Time Test](/images/Model_Deployment/Kernal%20Time%20Test.png)

可以大概分析下核函数启动到结束的过程：

- 主机线程启动核函数
- 核函数启动成功
- 控制返回主机线程
- 核函数执行完成
- 主机同步函数侦测到核函数执行完

我们要测试的是 2~4 的时间，但是用 CPU 计时方法，只能测试 1~5 的时间，所以测试得到的时间偏长。

### 用nvprof工具计时 Timing with nvprof

nvprof的用法如下：

```shell
nvprof [nvprof_args] <application>[application_args]
# 举例 nvprof ./sum_arrays_timer
```

![nvprof](/images/Model_Deployment/nvprof.png)

工具不仅给出了kernel执行的时间，比例，还有其他cuda函数的执行时间，可以看出核函数执行时间只有6%左右，其他内存分配，内存拷贝占了大部分事件，nvprof给出的核函数执行时间2.8985ms，cpuSecond计时结果是37.62ms。可见，nvprof可能更接近真实值。



## 参考引用 Reference



### 博客 Blogs

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)
- [CUDA学习入门（三） CUDA线程索引 & 如何设置Gridsize和Blocksize](https://blog.csdn.net/weixin_44222088/article/details/135732160)
- [CUDA线程模型与全局索引计算方式](https://zhuanlan.zhihu.com/p/666077650)