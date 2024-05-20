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

代码在文件夹 0_hello_world.cu 中

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
这句话C语言中没有’<<<>>>’是对设备进行配置的参数，也是CUDA扩展出来的部分。在调用时需要用<<<grid, block>>>来指定kernel要执行的线程数量。

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

## 2 CUDA 编程模型 CUDA Programming Model

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

### 2.6 给核函数计时 Timing Your Kernel

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

### 2.7 组织并行线程 Organizing Parallel Threads

由 2.3 节的 CUDA 索引计算，可以使用以下代码打印每个线程的标号信息，代码在 1_thread_index 中。可以得出二维矩阵加法核函数：

```C
__global__ void MatrixAdd(float * MatA, float * MatB, float * MatC, const int num_x, const int num_y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * num_x + row;

    if (row < num_x && col < num_y) {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}
```

下面调整不同的线程组织形式，测试一下不同的效率并保证得到正确的结果，但是什么时候得到最好的效率是后面要考虑的，我们要做的就是用各种不同的相乘组织形式得到正确结果的，代码在 6_MatAdd.cu 中。

首先来看**二维网格二维模块**的代码：

```C
dim3 blockDim_2(dim_x);
dim3 gridDim_2((row + blockDim_2.x - 1) / blockDim_2.x, col);
iStart = cpuSecond();
MatrixAdd<<<gridDim_2, blockDim_2>>>(A_dev, B_dev, C_dev, row, col); // 调用 CUDA 核函数
cudaDeviceSynchronize();
iElaps=cpuSecond() - iStart;
printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
        gridDim_2.x, gridDim_2.y, blockDim_2.x, blockDim_2.y, iElaps);
cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); // 将计算结果从设备端拷贝到主机端
```

运行结果：
```shell
GPU Execution configuration<<<(128, 128),(32, 32)>>> Time elapsed 0.005439 sec
```

接着使用**一维网格一维块**：

```C
dim3 blockDim_1(dim_x);
dim3 gridDim_1((sum + blockDim_1.x - 1) / blockDim_1.x);
iStart = cpuSecond();
MatrixAdd<<<gridDim_1, blockDim_1>>>(A_dev, B_dev, C_dev, sum, 1); // 调用 CUDA 核函数
cudaDeviceSynchronize();
iElaps=cpuSecond() - iStart;
printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
        gridDim_1.x, gridDim_1.y, blockDim_1.x, blockDim_1.y, iElaps);
cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); // 将计算结果从设备端拷贝到主机端
```

运行结果：
```shell
GPU Execution configuration<<<(524288, 1),(32, 1)>>> Time elapsed 0.003211 sec
```

**二维网格一维块**：

```C
dim3 blockDim_2(dim_x);
dim3 gridDim_2((row + blockDim_2.x - 1) / blockDim_2.x, col);
iStart = cpuSecond();
MatrixAdd<<<gridDim_2, blockDim_2>>>(A_dev, B_dev, C_dev, row, col); // 调用 CUDA 核函数
cudaDeviceSynchronize();
iElaps=cpuSecond() - iStart;
printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
        gridDim_2.x, gridDim_2.y, blockDim_2.x, blockDim_2.y, iElaps);
cudaMemcpy(C_host, C_dev, nBytes, cudaMemcpyDeviceToHost); // 将计算结果从设备端拷贝到主机端
```

运行结果：
```shell
GPU Execution configuration<<<(128, 4096),(32, 1)>>> Time elapsed 0.007409 sec
```

用不同的线程组织形式会得到正确结果，但是效率有所区别：

| 线程配置        | 执行时间      |
| -------------- | ------------- |
| (128,128),(32,32) | 0.002152      |
| (524288,1),(32,1) | 0.002965      |
| (128,4096),(32,1) | 0.002965      |

观察结果没有多大差距，而且最主要的是用不同的线程组织模式都得到了正确结果，并且：

- 改变执行配置（线程组织）能得到不同的性能
- 传统的核函数可能不能得到最好的效果
- 一个给定的核函数，通过调整网格和线程块大小可以得到更好的效果

### 2.8 GPU设备信息 Managing Devices

用 CUDA 的时候一般有两种情况，一种自己写完自己用，使用本机或者已经确定的服务器，这时候只要查看说明书或者配置说明就知道用的什么型号的GPU，以及GPU的所有信息，但是如果写的程序是通用的程序或者框架，在使用 CUDA 前要先确定当前的硬件环境，这使得我们的程序不那么容易因为设备不同而崩溃，本文介绍两种方法，第一种适用于通用程序或者框架，第二种适合查询本机或者可登陆的服务器，并且一般不会改变，那么这时候用一条 nvidia 驱动提供的指令查询设备信息就很方便了。

使用代码 10_device_information.cu 可以查询到以下信息：

```shell
./device_information Starting ...
Detected 1 CUDA Capable device(s)
Device 0:"NVIDIA GeForce GTX 1060 6GB"
  CUDA Driver Version / Runtime Version         12.2  /  11.5
  CUDA Capability Major/Minor version number:   6.1
  Total amount of global memory:                5.93 GBytes (6367543296 bytes)
  GPU Clock rate:                               1848 MHz (1.85 GHz)
  Memory Bus width:                             192-bits
  L2 Cache Size:                                1572864 bytes
  Max Texture Dimension Size (x,y,z)            1D=(131072),2D=(131072,65536),3D=(16384,16384,16384)
  Max Layered Texture Size (dim) x layers       1D=(32768) x 2048,2D=(32768,32768) x 2048
  Total amount of constant memory               65536 bytes
  Total amount of shared memory per block:      49152 bytes
  Total number of registers available per block:65536
  Wrap size:                                    32
  Maximun number of thread per multiprocesser:  2048
  Maximun number of thread per block:           1024
  Maximun size of each dimension of a block:    1024 x 1024 x 64
  Maximun size of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximu memory pitch                           2147483647 bytes
----------------------------------------------------------
Number of multiprocessors:                      10
Total amount of constant memory:                64.00 KB
Total amount of shared memory per block:        48.00 KB
Total number of registers available per block:  65536
Warp size                                       32
Maximum number of threads per block:            1024
Maximum number of threads per multiprocessor:  2048
Maximum number of warps per multiprocessor:     64
```

主要用到了下面API，了解API的功能最好不要看博客，因为博客不会与时俱进，要查文档，所以对于API的不了解，解决办法：查文档，查文档，查文档！

```shell
cudaSetDevice
cudaGetDeviceProperties
cudaDriverGetVersion
cudaRuntimeGetVersion
cudaGetDeviceCount
```

这里面很多参数是后面要介绍的，而且每一个都对性能有影响：

1. CUDA驱动版本
2. 设备计算能力编号
3. 全局内存大小（5.93G）
4. GPU主频
5. GPU带宽
6. L2缓存大小
7. 纹理维度最大值，不同维度下的
8. 层叠纹理维度最大值
9. 常量内存大小
10. 块内共享内存大小
11. 块内寄存器大小
12. 线程束大小
13. 每个处理器硬件处理的最大线程数
14. 每个块处理的最大线程数
15. 块的最大尺寸
16. 网格的最大尺寸
17. 最大连续线性内存

上面这些都是后面要用到的关键参数，这些会严重影响效率。后面会一一说到，不同的设备参数要按照不同的参数来使得程序效率最大化，所以必须在程序运行前得到所有关心的参数。

也可以使用 nvidia-smi nvidia驱动程序内带的一个工具返回当前环境的设备信息：

```shell
Fri May 17 22:16:50 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1060 6GB    Off | 00000000:2B:00.0 Off |                  N/A |
|106%   37C    P8               9W / 200W |     15MiB /  6144MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1198      G   /usr/lib/xorg/Xorg                            9MiB |
|    0   N/A  N/A      1479      G   /usr/bin/gnome-shell                          2MiB |
+---------------------------------------------------------------------------------------+
```

也可以使用其他参数：

```shell
nvidia-smi -L
nvidia-smi -q -i 0
nvidia-smi -q -i 0 -d MEMORY | tail -n 5
nvidia-smi -q -i 0 -d UTILIZATION | tail -n 4
```

下面这些nvidia-smi -q -i 0 的参数可以提取我们要的信息

- MEMORY
- UTILIZATION
- ECC
- TEMPERATURE
- POWER
- CLOCK
- COMPUTE
- PIDS
- PERFORMANCE
- SUPPORTED_CLOCKS
- PAGE_RETIREMENT
- ACCOUNTING

至此，CUDA 的编程模型大概就是这些了，核函数，计时，内存，线程，设备参数，这些足够能写出比CPU块很多的程序了。从下一章开始，深入硬件研究背后的秘密

## 3 CUDA 执行模型 CUDA Execution Model

### 3.1 概述 Introducing the CUDA Execution Model

CUDA执行模型揭示了GPU并行架构的抽象视图，再设计硬件的时候，其功能和特性都已经被设计好了，然后去开发硬件，如果这个过程模型特性或功能与硬件设计有冲突，双方就会进行商讨妥协，知道最后产品定型量产，功能和特性算是全部定型，而这些功能和特性就是变成模型的设计基础，而编程模型又直接反应了硬件设计，从而反映了设备的硬件特性。

比如最直观的一个就是内存，线程的层次结构帮助我们控制大规模并行，这个特性就是硬件设计最初设计好，然后集成电路工程师拿去设计，定型后程序员开发驱动，然后在上层可以直接使用这种执行模型来控制硬件。

所以了解CUDA的执行模型，可以帮助我们优化指令吞吐量，和内存使用来获得极限速度。

#### GPU 架构概述 GPU Architecture Overview

GPU架构是围绕一个流式多处理器（SM）的扩展阵列搭建的。通过复制这种结构来实现GPU的硬件并行

![Streaming Multiprocessors](/images/Model_Deployment/Streaming%20Multiprocessors.png)

上图包括关键组件：

- CUDA 核心
- 共享内存/一级缓存
- 寄存器文件
- 加载/存储单元
- 特殊功能单元
- 线程束调度器

##### Streaming Multiprocessors

GPU中每个SM都能支持数百个线程并发执行，每个GPU通常有多个SM，当一个核函数的网格被启动的时候，多个block会被同时分配给可用的SM上执行。当一个blcok被分配给一个SM后，他就只能在这个SM上执行了，不可能重新分配到其他SM上了，多个线程块可以被分配到同一个SM上。在SM上同一个块内的多个线程进行线程级别并行，而同一线程内，指令利用指令级并行将单个线程处理成流水线。

##### 线程束

CUDA 采用单指令多线程SIMT架构管理执行线程，不同设备有不同的线程束大小，但是到目前为止基本所有设备都是维持在32，也就是说每个SM上有多个block，一个block有多个线程（可以是几百个，但不会超过某个最大值），但是从机器的角度，在某时刻T，SM上只执行一个线程束，也就是32个线程在同时同步执行，线程束中的每个线程执行同一条指令，包括有分支的部分

##### SIMD vs SIMT

单指令多数据的执行属于向量机，比如我们有四个数字要加上四个数字，那么我们可以用这种单指令多数据的指令来一次完成本来要做四次的运算。这种机制的问题就是过于死板，不允许每个分支有不同的操作，所有分支必须同时执行相同的指令，必须执行没有例外。

相比之下单指令多线程SIMT就更加灵活了，虽然两者都是将相同指令广播给多个执行单元，但是SIMT的某些线程可以选择不执行，也就是说同一时刻所有线程被分配给相同的指令，SIMD规定所有人必须执行，而SIMT则规定有些人可以根据需要不执行，这样SIMT就保证了线程级别的并行，而SIMD更像是指令级别的并行。

SIMT包括以下SIMD不具有的关键特性：

1. 每个线程都有自己的指令地址计数器
2. 每个线程都有自己的寄存器状态
3. 每个线程可以有一个独立的执行路径
4. 
而上面这三个特性在编程模型可用的方式就是给每个线程一个唯一的标号（blckIdx,threadIdx），并且这三个特性保证了各线程之间的独立

##### 32 

32是个神奇数字，他的产生是硬件系统设计的结果，也就是集成电路工程师搞出来的，所以软件工程师只能接受。

从概念上讲，32是SM以SIMD方式同时处理的工作粒度，这句话这么理解，可能学过后面的会更深刻的明白，一个SM上在某一个时刻，有32个线程在执行同一条指令，这32个线程可以选择性执行，虽然有些可以不执行，但是他也不能执行别的指令，需要另外需要执行这条指令的线程执行完，然后再继续下一条，就像老师给小朋友们分水果：

第一次分苹果，分给所有32个人，你可以不吃，但是不吃也没别的，你就只能在那看别人吃，等别人吃完了，老师会把没吃的苹果回收，防止浪费。
第二次分橘子，你很爱吃，可是有别的小朋友不爱吃，当然这时候他也不能干别的，只能看你吃完。吃完后老师继续回收刚才没吃的橘子。
第三次分桃子，你们都很爱吃，大家一起吃，吃完了老师发现没有剩下的，继续发别的水果，一直发到所有种类的水果都发完了。今天就可以放学了。

简单的类比，但过程就是这样

##### CUDA 编程的组件与逻辑

下图从逻辑角度和硬件角度描述了CUDA编程模型对应的组件

![logical view and hardware view of CUDA](/images/Model_Deployment/logical%20view%20and%20hardware%20view%20of%20CUDA.png)


SM中共享内存，和寄存器是关键的资源，线程块中线程通过共享内存和寄存器相互通信协调。寄存器和共享内存的分配会严重影响性能

因为SM有限，虽然编程模型层面看所有线程都是并行执行的，但是在微观上看，所有线程块也是分批次的在物理层面的机器上执行，线程块里不同的线程可能进度都不一样，但是同一个线程束内的线程拥有相同的进度。并行就会引起竞争，多线程以未定义的顺序访问同一个数据，就导致了不可预测的行为，CUDA只提供了一种块内同步的方式，块之间没办法同步！

同一个SM上可以有不止一个常驻的线程束，有些在执行，有些在等待，他们之间状态的转换是不需要开销的。

#### Fermi 架构 The Fermi Architecture

Fermi架构是第一个完整的GPU架构，所以了解这个架构是非常有必要的

![Fermi Architecture](/images/Model_Deployment/Fermi%20Architecture.png)

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

每个多处理器SM有16个加载/存储单元所以每个时钟周期内有16个线程（半个线程束）计算源地址和目的地址。特殊功能单元SFU执行固有指令，如正弦，余弦，平方根和插值，SFU在每个时钟周期内的每个线程上执行一个固有指令

每个SM有两个线程束调度器，和两个指令调度单元，当一个线程块被指定给一个SM时，线程块内的所有线程被分成线程束，两个线程束选择其中两个线程束，在用指令调度器存储两个线程束要执行的指令（就像上面例子中分水果的水果一样，我们这里有两个班，两个班的老师各自控制的自己的水果，老师就是指令调度器）

像第一张图上的显示一样，每16个CUDA核心为一个组，还有16个加载/存储单元或4个特殊功能单元。当某个线程块被分配到一个SM上的时候，会被分成多个线程束，线程束在SM上交替执行：

![Fermi Execution](/images/Model_Deployment/SM%20Execution.png)

上面曾经说过，每个线程束在同一时间执行同一指令，同一个块内的线程束互相切换是没有时间消耗的。Fermi上支持同时并发执行内核。并发执行内核允许执行一些小的内核程序来充分利用GPU，如图：

![Fermi Execution](/images/Model_Deployment/Fermi%20Execution.png)

#### Kepler 架构 The Kepler Architecture

Kepler架构作为Fermi架构的后代，有以下技术突破：

- 强化的SM
- 动态并行
- Hyper-Q技术

技术参数也提高了不少，比如单个SM上CUDA核的数量，SFU的数量，LD/ST的数量等：

![Kepler Architecture1](/images/Model_Deployment/Kepler%20Architecture1.png)

![Kepler Architecture2](/images/Model_Deployment/Kepler%20Architecture2.png)

kepler架构的最突出的一个特点就是内核可以启动内核了，这使得我们可以使用GPU完成简单的递归操作，流程如下:

![Dynamic Parallelism](/images/Model_Deployment/Dynamic%20Parallelism.png)

Hyper-Q技术主要是CPU和GPU之间的同步硬件连接，以确保CPU在GPU执行的同事做更多的工作。Fermi架构下CPU控制GPU只有一个队列，Kepler架构下可以通过Hyper-Q技术实现多个队列如下图

![Hyper-Q](/images/Model_Deployment/Hyper-Q.png)

计算能力概览：

![Compute Capability1](/images/Model_Deployment/Compute%20Capability1.png)

![Compute Capability2](/images/Model_Deployment/Compute%20Capability2.png)

### 3.2 理解线程束执行的本质 Understanding the Nature of Warp Execution

前面已经大概的介绍了CUDA执行模型的大概过程，包括线程网格，线程束，线程间的关系，以及硬件的大概结构，例如SM的大概结构。而对于硬件来说，CUDA执行的实质是线程束的执行，因为硬件根本不知道每个块谁是谁，也不知道先后顺序，硬件(SM)只知道按照机器码跑，而给他什么，先后顺序，这个就是硬件功能设计的直接体现了。

从外表来看，CUDA执行所有的线程，并行的，没有先后次序的，但实际上硬件资源是有限的，不可能同时执行百万个线程，所以从硬件角度来看，物理层面上执行的也只是线程的一部分，而每次执行的这一部分，就是我们前面提到的线程束。

#### 线程束和线程块 Warps and Thread Blocks

线程束是SM中基本的执行单元，当一个网格被启动（网格被启动，等价于一个内核被启动，每个内核对应于自己的网格），网格中包含线程块，线程块被分配到某一个SM上以后，将分为多个线程束，每个线程束一般是32个线程（目前的GPU都是32个线程，但不保证未来还是32个）在一个线程束中，所有线程按照单指令多线程SIMT的方式执行，每一步执行相同的指令，但是处理的数据为私有的数据，下图反应的就是逻辑，实际，和硬件的图形化

![logical view and hardware view of a thread block](/images/Model_Deployment/view%20of%20a%20thread%20block.png)

线程块是个逻辑产物，因为在计算机里，内存总是一维线性存在的，所以执行起来也是一维的访问线程块中的线程，但是在写程序的时候却可以以二维三维的方式进行，原因是方便写程序，比如处理图像或者三维的数据，三维块就会变得很直接，很方便。

在块中，每个线程有唯一的编号（可能是个三维的编号），threadIdx
网格中，每个线程块也有唯一的编号(可能是个三维的编号)，blockIdx

那么每个线程就有在网格中的唯一编号。
当一个线程块中有128个线程的时候，其分配到SM上执行时，会分成4个块：

```shell
warp0: thread  0,........thread31
warp1: thread 32,........thread63
warp2: thread 64,........thread95
warp3: thread 96,........thread127
```

计算出三维对应的线性地址是：

$$
tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x
$$

一个线程块包含多少个线程束呢？

$$
WarpsPerBlock = ceil(\frac {ThreadsPerBlock}{warpSize})
$$

ceil函数是向上取整的函数，如下图所示的 $ceil(\frac{80}{32}) = 3$

![allocate warps of threads](/images/Model_Deployment/allocate%20warps%20of%20threads.png)

#### 线程束分化

线程束被执行的时候会被分配给相同的指令，处理各自私有的数据，还记得前文中的分苹果么？每次分的水果都是一样的，但是你可以选择吃或者不吃，这个吃和不吃就是分支，在CUDA中支持C语言的控制流，比如if…else, for ,while 等，CUDA中同样支持，但是如果一个线程束中的不同线程包含不同的控制条件，那么当我们执行到这个控制条件是就会面临不同的选择。

这里要讲一下CPU了，当我们的程序包含大量的分支判断时，从程序角度来说，程序的逻辑是很复杂的，因为一个分支就会有两条路可以走，如果有10个分支，那么一共有1024条路走，CPU采用流水线话作业，如果每次等到分支执行完再执行下面的指令会造成很大的延迟，所以现在处理器都采用分支预测技术，而CPU的这项技术相对于gpu来说高级了不止一点点，而这也是GPU与CPU的不同，设计初衷就是为了解决不同的问题。CPU适合逻辑复杂计算量不大的程序，比如操作系统，控制系统，GPU适合大量计算简单逻辑的任务，所以被用来算数。

如下一段代码：

```C
if (con)
{
    //do something
}
else
{
    //do something
}
```

假设这段代码是核函数的一部分，那么当一个线程束的32个线程执行这段代码的时候，如果其中16个执行if中的代码段，而另外16个执行else中的代码块，同一个线程束中的线程，执行不同的指令，这叫做线程束的分化。在每个指令周期，线程束中的所有线程执行相同的指令，但是线程束又是分化的，所以这似乎是相悖的，但是事实上这两个可以不矛盾。

解决矛盾的办法就是每个线程都执行所有的if和else部分，当一部分con成立的时候，执行if块内的代码，有一部分线程con不成立，那么他们怎么办？继续执行else？不可能的，因为分配命令的调度器就一个，所以这些con不成立的线程等待，就像分水果，你不爱吃，那你就只能看着别人吃，等大家都吃完了，再进行下一轮（也就是下一个指令）。线程束分化会产生严重的性能下降，条件分支越多，并行性削弱越严重。

注意线程束分化研究的是一个线程束中的线程，不同线程束中的分支互不影响。

执行过程如下：

![warp divergence](/images/Model_Deployment/warp%20divergence.png)

因为线程束分化导致的性能下降就应该用线程束的方法解决，根本思路是避免同一个线程束内的线程分化，而让我们能控制线程束内线程行为的原因是线程块中线程分配到线程束是有规律的而不是随机的。这就使得我们根据线程编号来设计分支是可以的，补充说明下，当一个线程束中所有的线程都执行if或者，都执行else时，不存在性能下降；只有当线程束内有分歧产生分支的时候，性能才会急剧下降。

线程束内的线程是可以被我们控制的，那么我们就把都执行if的线程塞到一个线程束中，或者让一个线程束中的线程都执行if，另外线程都执行else的这种方式可以将效率提高很多。

下面这个kernel可以产生一个比较低效的分支，代码在 11_divergence：

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

这种情况下我们假设只配置一个x=64的一维线程块，那么只有两个个线程束，线程束内奇数线程（threadIdx.x为奇数）会执行else，偶数线程执行if，分化很严重。

但是如果我们换一种方法，得到相同但是错乱的结果C，这个顺序其实是无所谓的，因为我们可以后期调整。那么下面代码就会很高效

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

第一个线程束内的线程编号tid从0到31，tid/warpSize都等于0，那么就都执行if语句。
第二个线程束内的线程编号tid从32到63，tid/warpSize都等于1，执行else。线程束内没有分支，效率较高。

用另一种方式，编译器就不会优化了：

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
nvprof --metrics branch_efficiency ./divergence
```

然后得到下面这些参数：

```C
==39332== NVPROF is profiling process 39332, command: ./divergence
./divergence using Device 0: NVIDIA GeForce GTX 1060 6GB
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

我们也可以通过编译选项禁用分值预测功能，这样kernel1和kernel3的效率是相近的。而这个值的计算是这样的：

$$
Branch Efficiency = \frac {Branches − DivergentBranches}{Branches}
$$

考察一下事件计数器：

```shell
nvprof --events branch,divergent_branch ./divergence
```

```shell
==39513== NVPROF is profiling process 39513, command: ./divergence
./divergence using Device 0: NVIDIA GeForce GTX 1060 6GB
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

#### 资源分配 Resource Partitioning

我们前面提到过，每个SM上执行的基本单位是线程束，也就是说，单指令通过指令调度器广播给某线程束的全部线程，这些线程同一时刻执行同一命令，当然也有分支情况，上一篇我们已经介绍了分支，这是执行的那部分，当然后有很多线程束没执行，那么这些没执行的线程束情况又如何呢？我给他们分成了两类，注意是我分的，不一定官方是不是这么讲。我们离开线程束内的角度（线程束内是观察线程行为，离开线程束我们就能观察线程束的行为了），一类是已经激活的，也就是说这类线程束其实已经在SM上准备就绪了，只是没轮到他执行，这时候他的状态叫做阻塞，还有一类可能分配到SM了，但是还没上到片上，这类我称之为未激活线程束。
而每个SM上有多少个线程束处于激活状态，取决于以下资源：

- 程序计数器
- 寄存器
- 共享内存

线程束一旦被激活来到片上，那么他就不会再离开SM直到执行结束。每个SM都有32位的寄存器组，每个架构寄存器的数量不一样，其存储于寄存器文件中，为每个线程进行分配，同时，固定数量的共享内存，在线程块之间分配。一个SM上被分配多少个线程块和线程束取决于SM中可用的寄存器和共享内存，以及内核需要的寄存器和共享内存大小。

这是一个平衡问题，就像一个固定大小的坑，能放多少萝卜取决于坑的大小和萝卜的大小，相比于一个大坑，小坑内可能放十个小萝卜，或者两个大萝卜，SM上资源也是，当kernel占用的资源较少，那么更多的线程（这是线程越多线程束也就越多）处于活跃状态，相反则线程越少。

关于寄存器资源的分配：

![allocate of register](/images/Model_Deployment/allocate%20of%20register.png)

![allocate of shared memory](/images/Model_Deployment/allocate%20of%20shared%20memory.png)

上面讲的主要是线程束，如果从逻辑上来看线程块的话，可用资源的分配也会影响常驻线程块的数量。特别是当SM内的资源没办法处理一个完整块，那么程序将无法启动，这个是我们应该找找自己的毛病，你得把内核写的多大，或者一个块有多少线程，才能出现这种情况。

以下是资源列表：

![Compute Capability3](/images/Model_Deployment/Compute%20Capability3.png)

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

#### 延迟隐藏 Latency Hiding

延迟是什么，就是当你让计算机帮你算一个东西的时候**计算需要用的时间**。举个宏观的例子，比如一个算法验证，你交给计算机，计算机会让某个特定的计算单元完成这个任务，共需要十分钟，而接下来这十分钟，你就要等待，等他算完了你才能计算下一个任务，那么这十分钟计算机的利用率有可能并不是100%，也就是说他的某些功能是空闲的，你就想能不能再跑一个同样的程序不同的数据（做过机器学习的这种情况不会陌生，大家都是同时跑好几个版本）然后你又让计算机跑，这时候你发现还没有完全利用完资源，于是又继续加任务给计算机，结果加到第十分钟了，已经加了十个了，你还没加完，但是第一个任务已经跑完了，如果你这时候停止加任务，等陆陆续续的你后面加的任务都跑完了共用时20分钟，共执行了10个任务，那么平局一个任务用时 $\frac{20}{10}=2$ 分钟/任务 。 但是我们还有一种情况，因为任务还有很多，第十分钟你的第一个任务结束的时候你继续向你的计算机添加任务，那么这个循环将继续进行，那么第二十分钟你停止添加任务，等待第三十分钟所有任务执行完，那么平均每个任务的时间是： $\frac{30}{20}=1.5$ 分钟/任务，如果一直添加下去，$lim_{n\to\infty}\frac{n+10}{n}=1$ 也就是极限速度，一分钟一个，隐藏了9分钟的延迟。

当然上面的另一个重要参数是每十分钟添加了10个任务，如果每十分钟共可以添加100个呢，那么二十分钟就可以执行100个，每个任务耗时： $\frac{20}{100}=0.2$ 分钟/任务 三十分钟就是 $\frac{30}{200}=0.15$ 如果一直添加下去， $lim_{n\to\infty}\frac{n+10}{n\times 10}=0.1$ 分钟/任务。

这是理想情况，有一个必须考虑的就是虽然你十分钟添加了100个任务，可是没准添加50个计算机就满载了，这样的话 极限速度只能是：$lim_{n\to\infty}\frac{n+10}{n\times 5}=0.2$ 分钟/任务 了。

所以最大化是要最大化硬件，尤其是计算部分的硬件满跑，都不闲着的情况下利用率是最高的，总有人闲着，利用率就会低很多，即最大化功能单元的利用率。利用率与常驻线程束直接相关。硬件中线程调度器负责调度线程束调度，当每时每刻都有可用的线程束供其调度，这时候可以达到计算资源的完全利用，以此来保证通过其他常驻线程束中发布其他指令的，可以隐藏每个指令的延迟。

与其他类型的编程相比，GPU的延迟隐藏及其重要。对于指令的延迟，通常分为两种：

- 算术指令
- 内存指令

算数指令延迟是一个算术操作从开始，到产生结果之间的时间，这个时间段内只有某些计算单元处于工作状态，而其他逻辑计算单元处于空闲。内存指令延迟很好理解，当产生内存访问的时候，计算单元要等数据从内存拿到寄存器，这个周期是非常长的。

- 算术延迟 10~20 个时钟周期
- 内存延迟 400~800 个时钟周期

下图就是阻塞线程束到可选线程束的过程逻辑图：

![Warp Scheduler](/images/Model_Deployment/Warp%20Scheduler.png)

其中线程束0在阻塞两段时间后恢复可选模式，但是在这段等待时间中，SM没有闲置。

那么至少需要多少线程，线程束来保证最小化延迟呢？little法则给出了下面的计算公式: 

$$
\text{所需线程束} = \text{延迟} \times \text{吞吐量}
$$

> 注意带宽和吞吐量的区别，带宽一般指的是理论峰值，最大每个时钟周期能执行多少个指令，吞吐量是指实际操作过程中每分钟处理多少个指令。

这个可以想象成一个瀑布，像这样，绿箭头是线程束，只要线程束足够多，吞吐量是不会降低的：

![Throughput](/images/Model_Deployment/Throughput.png)

下面表格给出了Fermi 和Kepler执行某个简单计算时需要的并行操作数：

![Full Arithmetic Utilization](/images/Model_Deployment/Full%20Arithmetic%20Utilization.png)

另外有两种方法可以提高并行：

- **指令级并行(ILP):** 一个线程中有很多独立的指令
- **线程级并行(TLP):** 很多并发地符合条件的线程

同样，与指令周期隐藏延迟类似，内存隐藏延迟是靠内存读取的并发操作来完成的，需要注意的是，指令隐藏的关键目的是使用全部的计算资源，而内存读取的延迟隐藏是为了使用全部的内存带宽，内存延迟的时候，计算资源正在被别的线程束使用，所以我们不考虑内存读取延迟的时候计算资源在做了什么，这两种延迟我们看做两个不同的部门但是遵循相同的道理。

我们的根本目的是把计算资源，内存读取的带宽资源全部使用满，这样就能达到理论的最大效率。
同样下表根据Little 法则给出了需要多少线程束来最小化内存读取延迟，不过这里有个单位换算过程，机器的性能指标内存读取速度给出的是GB/s 的单位，而我们需要的是每个时钟周期读取字节数，所以要用这个速度除以频率，例如C 2070 的内存带宽是144 GB/s 化成时钟周期： $\frac{144GB/s}{1.566GHz}=92 B/t$ ,这样就能得到单位时间周期的内存带宽了。

![Full Memory Utilization](/images/Model_Deployment/Full%20Memory%20Utilization.png)

需要说明的是这个速度不是单个SM的而是整个GPU设备的，用的内存带宽是GPU设备的而不是针对一个SM的。Fermi 需要并行的读取74的数据才能让GPU带宽满载，如果每个线程读取4个字节，我们大约需要18500个线程，大约579个线程束才能达到这个峰值。所以，延迟的隐藏取决于活动的线程束的数量，数量越多，隐藏的越好，但是线程束的数量又受到上面的说的资源影响。所以这里就需要寻找最优的执行配置来达到最优的延迟隐藏。

那么我们怎么样确定一个线程束的下界呢，使得当高于这个数字时SM的延迟能充分的隐藏，其实这个公式很简单，也很好理解，就是SM的计算核心数乘以单条指令的延迟。比如32个单精度浮点计算器，每次计算延迟20个时钟周期，那么我需要最少 32x20 =640 个线程使设备处于忙碌状态。

#### 占用率 Occupancy

占用率是一个SM种活跃的线程束的数量，占SM最大支持线程束数量的比。前面写的程序10_device_Information 中添加几个成员的查询就可以帮我们找到这个值。

CUDA工具包中提供一个叫做UCDA占用率计算器的电子表格，填上相关数据可以帮你自动计算网格参数：

![Occupancy Calculator](/images/Model_Deployment/Occupancy%20Calculator.png)

上面我们已经明确内核使用寄存器的数量会影响SM内线程束的数量，nvcc的编译选项也有手动控制寄存器的使用。
也可以通过调整线程块内线程的多少来提高占用率，当然要合理不能太极端：

- 小的线程块：每个线程块中线程太少，会在所有资源没用完就达到了线程束的最大要求
- 大的线程块：每个线程块中太多线程，会导致每个SM中每个线程可用的硬件资源较少。

#### 同步 Synchronization

并发程序对同步非常有用，比如pthread中的锁，openmp中的同步机制，这没做的主要目的是避免内存竞争。CUDA同步这里只讲两种：

- 线程块内同步
- 系统级别

块级别的就是同一个块内的线程会同时停止在某个设定的位置，用

```C
__syncthread();
```

这个函数完成，这个函数只能同步同一个块内的线程，不能同步不同块内的线程，想要同步不同块内的线程，就只能让核函数执行完成，控制程序交换主机，这种方式来同步所有线程。

内存竞争是非常危险的，一定要非常小心，这里经常出错。

#### 可扩展性 Scalability

可扩展性其实是相对于不同硬件的，当某个程序在设备1上执行的时候时间消耗是T。当我们使用设备2时，其资源是设备1的两倍，我们希望得到T/2的运行速度，这种性质是CUDA驱动部分提供的特性，目前来说 Nvidia正在致力于这方面的优化，如下图：

![Scalability](/images/Model_Deployment/Scalability.png)

### 3.3 并行性表现 Exposing Parallelism

本节的主要内容就是进一步理解线程束在硬件上执行的本质过程，结合上几篇关于执行模型的学习，本文相对简单，通过修改核函数的配置，来观察核函数的执行速度，以及分析硬件利用数据，分析性能。调整核函数配置是CUDA开发人员必须掌握的技能，本篇只研究对核函数的配置是如何影响效率的（也就是通过网格，块的配置来获得不同的执行效率。）

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

没有任何优化的最简单的二维矩阵加法，代码在 5_sum_matrix2D 中。这里用两个 $8192×8192
$ 的矩阵相加来测试效率。注意一下这里的GPU内存，一个矩阵是 $2^{14}×2^{14}×2^2=2^{30}$ 字节 也就是 1G，三个矩阵就是 3G。 

#### 用 nvprof 检测活跃的线程束 Checking Active Warps with nvprof

对比性能要控制变量，上面的代码只用两个变量，也就是块的x和y的大小，所以，调整x和y的大小来产生不同的效率，结果如下：

```shell
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 32 32
CPU Execution Time elapsed 0.538640 sec
GPU Execution configuration<<<(512, 512),(32, 32)>>> Time elapsed 0.090911 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 32 16
CPU Execution Time elapsed 0.548685 sec
GPU Execution configuration<<<(512, 1024),(32, 16)>>> Time elapsed 0.086876 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 16 32
CPU Execution Time elapsed 0.544791 sec
GPU Execution configuration<<<(1024, 512),(16, 32)>>> Time elapsed 0.056706 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 16 16
CPU Execution Time elapsed 0.548078 sec
GPU Execution configuration<<<(1024, 1024),(16, 16)>>> Time elapsed 0.056472 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 16 8
CPU Execution Time elapsed 0.546093 sec
GPU Execution configuration<<<(1024, 2048),(16, 8)>>> Time elapsed 0.086659 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 8 16
CPU Execution Time elapsed 0.545576 sec
GPU Execution configuration<<<(2048, 1024),(8, 16)>>> Time elapsed 0.056402 sec
```

汇总成表格:

| gridDim  | blockDim | CPU Time (s) | GPU Time (s)   |
|----------|----------|--------------|----------------|
| 512, 512 | 32, 32   | 0.538640     | 0.090911       |
| 512, 1024| 32, 16   | 0.548685     | 0.086876       |
| 1024, 512| 16, 32   | 0.544791     | 0.056706       |
| 1024, 1024| 16, 16  | 0.548078     | 0.056472       |
| 1024, 2048| 16, 8   | 0.546093     | 0.086659       |
| 2048, 1024| 8, 16   | 0.545576     | 0.056402       |

当块大小超过硬件的极限，并没有报错，而是返回了错误值，这个值得注意。另外，每个机器执行此代码效果可能定不一样，所以大家要根据自己的硬件分析数据。书上给出的 M2070 就和我们的结果不同，2070的 (32,16) 效率最高，而我们的 (16, 16) 效率最高，毕竟架构不同，而且CUDA版本不同导致了优化后的机器码差异很大，所以我们还是来看看活跃线程束的情况，使用

```shell
nvprof --metrics achieved_occupancy ./sum_matrix2D 
```

得出结果

```shell
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 32 32 
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 32 16
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 16 32
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 16 16
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 16 8
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 8 16
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

| gridDim  | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy |
|----------|----------|--------------|--------------|---------------------|
| 512, 512 | 32, 32   | 0.550530     | 0.096127     | 0.728469            |
| 512, 1024| 32, 16   | 0.551584     | 0.089149     | 0.904511            |
| 1024, 512| 16, 32   | 0.547609     | 0.070035     | 0.817224            |
| 1024, 1024| 16, 16  | 0.550066     | 0.062846     | 0.885973            |
| 1024, 2048| 16, 8   | 0.548652     | 0.092749     | 0.968459            |
| 2048, 1024| 8, 16   | 0.549166     | 0.062462     | 0.870483            |

可见活跃线程束比例高的未必执行速度快，但实际上从原理出发，应该是利用率越高效率越高，但是还受到其他因素制约。活跃线程束比例的定义是：每个周期活跃的线程束的平均值与一个sm支持的线程束最大值的比。

#### 用 nvprof 检测内存操作 Checking Active Warps with nvprof

下面我们继续用nvprof来看看内存利用率如何

```C
nvprof --metrics gld_throughput ./sum_matrix2D
```

```shell
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_throughput ./sum_matrix2D 32 32 
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_throughput ./sum_matrix2D 32 16
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_throughput ./sum_matrix2D 16 32
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_throughput ./sum_matrix2D 16 16
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_throughput ./sum_matrix2D 16 8
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_throughput ./sum_matrix2D 8 16
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

| gridDim  | blockDim | CPU Time (s) | GPU Time (s) | Achieved Occupancy | GLD Throughput (GB/s) |
|----------|----------|--------------|--------------|---------------------|-----------------------|
| 512, 512 | 32, 32   | 0.544097     | 0.273369     | 0.728469            | 61.836                |
| 512, 1024| 32, 16   | 0.545615     | 0.247466     | 0.904511            | 68.650                |
| 1024, 512| 16, 32   | 0.553040     | 0.244212     | 0.817224            | 34.835                |
| 1024, 1024| 16, 16  | 0.545451     | 0.240271     | 0.885973            | 35.409                |
| 1024, 2048| 16, 8   | 0.543101     | 0.246472     | 0.968459            | 34.444                |
| 2048, 1024| 8, 16   | 0.545891     | 0.240333     | 0.870483            | 17.701                |

可以看出综合第二种配置的线程束吞吐量最大。所以可见吞吐量和线程束活跃比例一起都对最终的效率有影响。


接着看看全局加载效率，全局效率的定义是：**被请求的全局加载吞吐量占所需的全局加载吞吐量的比值（全局加载吞吐量）**，也就是说应用程序的加载操作利用了设备内存带宽的程度；注意区别吞吐量和全局加载效率的区别，这个在前面我们已经解释过吞吐量了。

```C
nvprof --metrics gld_efficiency ./sum_matrix2D
```

```shell
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_efficiency ./sum_matrix2D 32 32
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_efficiency ./sum_matrix2D 32 16
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_efficiency ./sum_matrix2D 16 32
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_efficiency ./sum_matrix2D 16 16
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_efficiency ./sum_matrix2D 16 8
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics gld_efficiency ./sum_matrix2D 8 16
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

| gridDim     | blockDim  | CPU Time (s) | GPU Time (s) | Achieved Occupancy | GLD Throughput (GB/s) | GLD Efficiency |
|-------------|-----------|--------------|--------------|---------------------|-----------------------|----------------|
| (512, 512)  | (32, 32)  | 0.544097     | 0.273369     | 0.728469            | 61.836                | 12.50%         |
| (512, 1024) | (32, 16)  | 0.545615     | 0.247466     | 0.904511            | 68.650                | 12.50%         |
| (1024, 512) | (16, 32)  | 0.553040     | 0.244212     | 0.817224            | 34.835                | 25.00%         |
| (1024, 1024)| (16, 16)  | 0.545451     | 0.240271     | 0.885973            | 35.409                | 25.00%         |
| (1024, 2048)| (16, 8)   | 0.543101     | 0.246472     | 0.968459            | 34.444                | 25.00%         |
| (2048, 1024)| (8, 16)   | 0.545891     | 0.240333     | 0.870483            | 17.701                | 50.00%         |


可见，当线程束越小，内存效率越高。有效加载效率是指在全部的内存请求中（当前在总线上传递的数据）有多少是我们要用于计算的。

#### 增大并行性 Exposing More Parallelism

线程块中内层的维度（blockDim.x）过小 是否对现在的设备还有影响，我们来看一下下面的试验

```shell
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 64 2
CPU Execution Time elapsed 0.544023 sec
GPU Execution configuration<<<(256, 8192),(64, 2)>>> Time elapsed 0.356677 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 64 4
CPU Execution Time elapsed 0.544404 sec
GPU Execution configuration<<<(256, 4096),(64, 4)>>> Time elapsed 0.174845 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 64 8
CPU Execution Time elapsed 0.544168 sec
GPU Execution configuration<<<(256, 2048),(64, 8)>>> Time elapsed 0.091977 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 128 2
CPU Execution Time elapsed 0.545258 sec
GPU Execution configuration<<<(128, 8192),(128, 2)>>> Time elapsed 0.355204 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 128 4
CPU Execution Time elapsed 0.547236 sec
GPU Execution configuration<<<(128, 4096),(128, 4)>>> Time elapsed 0.176689 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 128 8
CPU Execution Time elapsed 0.545464 sec
GPU Execution configuration<<<(128, 2048),(128, 8)>>> Time elapsed 0.089984 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 256 2
CPU Execution Time elapsed 0.545916 sec
GPU Execution configuration<<<(64, 8192),(256, 2)>>> Time elapsed 0.363761 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 256 4
CPU Execution Time elapsed 0.548850 sec
GPU Execution configuration<<<(64, 4096),(256, 4)>>> Time elapsed 0.190659 sec
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D$ ./sum_matrix2D 256 8
CPU Execution Time elapsed 0.547406 sec
GPU Execution configuration<<<(64, 2048),(256, 8)>>> Time elapsed 0.000030 sec
```

| gridDim     | blockDim  | CPU Time (s) | GPU Time (s) |
|-------------|-----------|--------------|--------------|
| (256, 8192) | (64, 2)   | 0.544023     | 0.356677     |
| (256, 4096) | (64, 4)   | 0.544404     | 0.174845     |
| (256, 2048) | (64, 8)   | 0.544168     | 0.091977     |
| (128, 8192) | (128, 2)  | 0.545258     | 0.355204     |
| (128, 4096) | (128, 4)  | 0.547236     | 0.176689     |
| (128, 2048) | (128, 8)  | 0.545464     | 0.089984     |
| (64, 8192)  | (256, 2)  | 0.545916     | 0.363761     |
| (64, 4096)  | (256, 4)  | 0.548850     | 0.190659     |
| (64, 2048)  | (256, 8)  | 0.547406     | 0.000030     |


通过这个表我们发现，块最小的反而获得最低的效率，即数据量大可能会影响结果，当数据量大的时候有可能决定时间的因素会发生变化，但是一些结果是可以观察到

- 尽管（64，4） 和 （128，2） 有同样大小的块，但是执行效率不同，说明内层线程块尺寸影响效率
- 最后的块参数无效，所有线程超过了 1024 GPU 最大限制线程数
- 尽管 (64, 2) 线程块最小，但是启动了最多的线程快，速度并不是最快的
- 综合线程块大小和数量，(128, 8) 速度最快

调整块的尺寸，还是为了增加并行性，或者说增加活跃的线程束，看看线程束的活跃比例：

```shell
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D#  nvprof --metrics achieved_occupancy ./sum_matrix2D 64 2
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 64 4
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 64 8
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 128 2
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 128 4
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 128 8
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 256 2
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 256 4
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
root@linxi1989:/home/linxi/DevKit/Projects/2024-03-18_EdgeComputingCV/docs/Model_Deployment/CUDA/CUDA_Learning/build/5_sum_matrix2D# nvprof --metrics achieved_occupancy ./sum_matrix2D 256 8
==48214== NVPROF is profiling process 48214, command: ./sum_matrix2D 256 8
CPU Execution Time elapsed 0.549278 sec
GPU Execution configuration<<<(64, 2048),(256, 8)>>> Time elapsed 0.000024 sec
==48214== Profiling application: ./sum_matrix2D 256 8
==48214== Profiling result:
No events/metrics were profiled.
```

| gridDim     | blockDim  | CPU Time (s) | GPU Time (s) | Achieved Occupancy |
|-------------|-----------|--------------|--------------|---------------------|
| (256, 8192) | (64, 2)   | 0.549154     | 0.363687     | 0.941718            |
| (256, 4096) | (64, 4)   | 0.554265     | 0.182942     | 0.939658            |
| (256, 2048) | (64, 8)   | 0.552905     | 0.100848     | 0.912401            |
| (128, 8192) | (128, 2)  | 0.554928     | 0.361216     | 0.842183            |
| (128, 4096) | (128, 4)  | 0.555749     | 0.182397     | 0.833157            |
| (128, 2048) | (128, 8)  | 0.550801     | 0.099784     | 0.732285            |
| (64, 8192)  | (256, 2)  | 0.550500     | 0.369576     | 0.804247            |
| (64, 4096)  | (256, 4)  | 0.538097     | 0.197963     | 0.791321            |
| (64, 2048)  | (256, 8)  | 0.549278     | 0.000024     | -                   |

可见最高的利用率没有最高的效率。没有任何一个因素可以直接左右最后的效率，一定是大家一起作用得到最终的结果，多因一效的典型例子，于是在优化的时候，我们应该首先保证测试时间的准确性，客观性，以及稳定性。

- 大部分情况，单一指标不能优化出最优性能
- 总体性能直接相关的是内核的代码本质（内核才是关键）
- 指标与性能之间选择平衡点
- 从不同的角度寻求指标平衡，最大化效率
- 网格和块的尺寸为调节性能提供了一个不错的起点

### 3.4 避免分支分化 Avoiding Branch Divergence

#### 并行规约问题 The Parallel Reduction Problem

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
![Neighbored pair](/images/Model_Deployment/Neighbored%20pair.png)

2. **交错配对：** 元素与一定距离的元素配对
![Interleaved pair](/images/Model_Deployment/Interleaved%20pair.png)

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

这个加法运算可以改成任何满足结合律和交换律的计算，比如乘法，求最大值等。通过不同的配对方式，不同的数据组织来看CUDA的执行效率。

#### 并行规约中的分化 Divergence in Parallel Reduction

**线程束分化**已经明确说明了，有判断条件的地方就会产生分支，比如 if 和 for 这类关键词。如下图所表示的那样，对相邻元素配对进行内核实现的流程描述：

![Parallel Reduction](/images/Model_Deployment/Parallel%20Reduction.png)

**第一步：** 是把这个一个数组分块，每一块只包含部分数据，如上图那样（图中数据较少，但是我们假设一块上只有这么多。），我们假定这是线程块的全部数据
**第二步：** 就是每个线程要做的事，橙色圆圈就是每个线程做的操作，可见线程threadIdx.x=0 的线程进行了三次计算，奇数线程一致在陪跑，没做过任何计算，但是根据3.2中介绍，这些线程虽然什么都不干，但是不可以执行别的指令，4号线程做了两步计算，2号和6号只做了一次计算。
**第三步：** 将所有块得到的结果相加，就是最终结果

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

![stride](/images/Model_Deployment/stride.png)

注意主机端和设备端的分界，注意设备端的数据分块，完整代码在 6_reduceInteger.cu，结果如下：

```shell
        with array size 16777216  grid 16384 block 1024 
cpu sum:1 
cpu reduce                 elapsed 0.003834 ms cpu_sum: 1
gpu warmup                 elapsed 0.064948 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.062404 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.031218 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.013156 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
Test success!
```

warmup 是为了启动gpu防止首次启动计算时gpu的启动过程耽误时间，影响效率测试，warmup 的代码就是 reducneighbored 的代码，可见还是有微弱的差别的。下面两个是经过优化的代码。

#### 改善并行规约的分化 Improving Divergence in Parallel Reduction

上面归约显然是最原始的，未经过优化的东西是不能拿出去使用的，或者说一个真理是，不可能一下子就写出来满意的代码。

```C
if ((tid % (2 * stride)) == 0)
```
这个条件判断给内核造成了极大的分支，如图所示：

![Parallel Reduction](/images/Model_Deployment/Parallel%20Reduction2.png)

第一轮有 $\frac {1}{2}$ 的线程没用
第二轮有 $\frac {3}{4}$ 的线程没用
第三轮有 $\frac {7}{8}$ 的线程没用

对于上面的低利用率，我们想到了下面这个方案来解决：

![Parallel Reduction3](/images/Model_Deployment/Parallel%20Reduction3.png)

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
cpu sum:1 
cpu reduce                 elapsed 0.003834 ms cpu_sum: 1
gpu warmup                 elapsed 0.064948 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.062404 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.031218 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.013156 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
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
cpu sum:1 
cpu reduce                 elapsed 0.003717 ms cpu_sum: 1
gpu warmup                 elapsed 0.074615 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.069011 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.035108 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.017052 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
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
cpu sum:1 
cpu reduce                 elapsed 0.003933 ms cpu_sum: 1
gpu warmup                 elapsed 0.182649 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.162657 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.069491 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.060214 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
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

#### 交错配对的规约 Reducing with Interleaved Pairs

上面的套路是修改线程处理的数据，使部分线程束最大程度利用数据，接下来采用同样的思想，但是方法不同，接下来我们使用的方法是调整跨度，也就是我们每个线程还是处理对应的内存的位置，但内存对不是相邻的了，而是隔了一定距离的：

![Parallel Reduction4](/images/Model_Deployment/Parallel%20Reduction4.png)

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
cpu sum:1 
cpu reduce                 elapsed 0.003834 ms cpu_sum: 1
gpu warmup                 elapsed 0.064948 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.062404 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.031218 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.013156 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
Test success!
```

如果单从优化原理的角度，这个内核和前面的内核应该是相同效率的，但是测试结果是，这个新内核比前面的内核速度快了不少，所以还是考察一下指标吧：

```C
nvprof --metrics inst_per_warp ./reduceInteger
```

```shell
        with array size 16777216  grid 16384 block 1024 
==58133== NVPROF is profiling process 58133, command: ./reduceInteger
cpu sum:1 
cpu reduce                 elapsed 0.004141 ms cpu_sum: 1
gpu warmup                 elapsed 0.074924 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.066874 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.035046 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.016936 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
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
cpu sum:1 
cpu reduce                 elapsed 0.003949 ms cpu_sum: 1
gpu warmup                 elapsed 0.181888 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighbored       elapsed 0.164936 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceNeighboredLess   elapsed 0.072196 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
gpu reduceInterleaved      elapsed 0.060335 ms gpu_sum: 1   <<<grid 16384 block 1024>>>
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









































## 参考引用 Reference



### 博客 Blogs

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)
- [CUDA学习入门（三） CUDA线程索引 & 如何设置Gridsize和Blocksize](https://blog.csdn.net/weixin_44222088/article/details/135732160)
- [CUDA线程模型与全局索引计算方式](https://zhuanlan.zhihu.com/p/666077650)