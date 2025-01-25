## 2 - CUDA 编程模型 CUDA Programming Model

---

### 1. 概念 Concepts

**内核**

CUDA C++ 通过允许程序员定义称为kernel的 C++ 函数来扩展 C++，当调用内核时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次。

使用 `__global__ ` 声明说明符定义内核，并使用新的 `<<<...>>>` 执行配置（execution configuration）语法指定内核调用时的 CUDA 线程数。每个执行内核的线程都有一个唯一的线程 ID，可以通过内置变量在内核中访问。

**线程层次**

为方便起见，threadIdx 是一个 **3分量(3-component)向量**，因此可以使用一个一维、二维或三维的 **线程索引(thread index)** 来识别线程，形成一个具有一个维度、两个维度或三个维度的、由线程组成的块，我们称之为**线程块(thread block)**。 这提供了一种自然的方法来对某一范围（例如向量、矩阵或空间）内的元素进行访问并调用计算。

- threadldx.[x y z]: 执行当前kernel函数的线程在block中的索引值
- blockldx.[x y z]: 执行当前kernel函数的线程所在block, 在grid中的索引值
- blockDim.[x y z]: 表示一个block中包含多少个线程
- gridDim.[x y z]: 表示一个grid中包含多少个block

### 2. CUDA 编程模型概述 Introducing the CUDA Programming Model 

CUDA编程模型为应用和硬件设备之间的桥梁，所以CUDA C是编译型语言，不是解释型语言，OpenCL就有点类似于解释型语言，通过编译器和链接，给操作系统执行（操作系统包括GPU在内的系统），下面的结构图片能形象的表现他们之间的关系：

![CUDA Model](/images/Professional%20CUDA%20C%20Programming/CUDA%20Model.png)

其中Communication Abstraction是编程模型和编译器，库函数之间的分界线。编程模型可以理解为，我们要用到的语法，内存结构，线程结构等这些我们写程序时我们自己控制的部分，这些部分控制了异构计算设备的工作模式，都是属于**编程模型**。

GPU中大致可以分为：

-   核函数
-   内存管理
-   线程管理
-   流

以上这些理论同时也适用于其他非CPU+GPU异构的组合。  

下面我们会说两个我们GPU架构下特有几个功能：

-   通过组织层次结构在GPU上**组织线程**的方法
-   通过组织层次结构在GPU上**组织内存**的方法

从宏观上我们可以从以下几个环节完成CUDA应用开发：

- 领域层
- 逻辑层
- 硬件层

第一步就是在领域层（也就是你所要解决问题的条件）分析数据和函数，以便在并行运行环境中能正确，高效地解决问题。

当分析设计完程序就进入了编程阶段，我们关注点应转向如何组织并发进程，这个阶段要从逻辑层面思考。

CUDA 模型主要的一个功能就是线程层结构抽象的概念，以允许控制线程行为。这个抽象为并行变成提供了良好的可扩展性（这个扩展性后面有提到，就是一个CUDA程序可以在不同的GPU机器上运行，即使计算能力不同）。

在硬件层上，通过理解线程如何映射到机器上，能充分帮助我们提高性能。

#### 2.1. CUDA 编程结构 CUDA Programming Structure

一个异构环境，通常有多个CPU多个GPU，他们都通过PCIe总线相互通信，也是通过PCIe总线分隔开的。所以我们要区分一下两种设备的内存：

-   主机：CPU及其内存
-   设备：GPU及其内存

注意这两个内存从硬件到软件都是隔离的（CUDA6.0 以后支持统一寻址），我们目前先不研究统一寻址，我们现在还是用内存来回拷贝的方法来编写调试程序，以巩固大家对两个内存隔离这个事实的理解。

CUDA程序的典型处理流程遵循如下：

- 将数据从CPU内存复制到GPU内存。
- 调用核函数在GPU内存中对数据进行操作。
- 将数据从GPU内存复制回CPU内存。

一个完整的CUDA应用可能的执行顺序如下图。串行代码（及任务并行代码）在主机
CPU上执行，而并行代码在GPU上执行：

![Process Procedure](/images/Professional%20CUDA%20C%20Programming/Process%20Procedure.png)

从host的串行到调用核函数（核函数被调用后控制马上归还主机线程，也就是在第一个并行代码执行时，很有可能第二段host代码已经开始同步执行了）。

#### 2.2. 内存管理 Managing Memory

内存管理在传统串行程序是非常常见的，寄存器空间，栈空间内的内存由机器自己管理，堆空间由用户控制分配和释放，CUDA程序同样，只是CUDA提供的API可以分配管理设备上的内存，当然也可以用CDUA管理主机上的内存，主机上的传统标准库也能完成主机内存管理。  

下面表格有一些主机API和CUDA C的API的对比：

| 标准C函数 | CUDA C 函数 |   说明   |
|:---------:|:-----------:|:--------:|
|  malloc   | cudaMalloc  | 内存分配 |
|  memcpy   | cudaMemcpy  | 内存复制 |
|  memset   | cudaMemset  | 内存设置 |
|   free    |  cudaFree   | 释放内存 |

我们先研究最关键的一步，这一步要走总线的

```C
cudaError_t cudaMemcpy(void * dst,const void * src,size_t count,  
  cudaMemcpyKind kind)
```

这个函数是内存拷贝过程，可以完成以下几种过程（cudaMemcpyKind kind）

-   cudaMemcpyHostToHost
-   cudaMemcpyHostToDevice
-   cudaMemcpyDeviceToHost
-   cudaMemcpyDeviceToDevice

这四个过程的方向可以清楚的从字面上看出来，如果函数执行成功，则会返回 cudaSuccess 否则返回 cudaErrorMemoryAllocation

使用下面这个指令可以吧上面的错误代码翻译成详细信息：

```C
char* cudaGetErrorString(cudaError_t error)
```

内存是分层次的，下图可以简单地描述，但是不够准确，后面我们会详细介绍每一个具体的环节：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124154320.png)

共享内存（shared Memory）和全局内存（global Memory）后面我们会特别详细深入的研究，这里我们来个例子，两个向量的加法，数组a的第一个元素与数组b的第一个元素相加，得到的结果作为数组c的第一个元素，重复这个过程直到数组中的所有元素都进行了一次运算。代码在 chapter02/sumArrays 中：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124154851.png)

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void sumArrays(float* a, float* b, float* res, const int size) {
    for (int i = 0; i < size; i += 4) {
        res[i]     = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}
__global__ void sumArraysGPU(float* a, float* b, float* res) {
    int i  = threadIdx.x;
    res[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 32;
    printf("Vector size:%d\n", nElem);
    int    nByte          = sizeof(float) * nElem;
    float* a_h            = (float*)malloc(nByte);
    float* b_h            = (float*)malloc(nByte);
    float* res_h          = (float*)malloc(nByte);
    float* res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((float**)&a_d, nByte));
    CHECK(cudaMalloc((float**)&b_d, nByte));
    CHECK(cudaMalloc((float**)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(nElem);
    dim3 grid(nElem / block.x);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configuration %d,%d\n", block.x, grid.x);
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}
```

解释下内存管理部分的代码：

```C
cudaMalloc((float**)&a_d,nByte);
```

分配设备端的内存空间，为了区分设备和主机端内存，我们可以给变量加后缀或者前缀h_表示host，d_表示device。**一个经常会发生的错误就是混用设备和主机的内存地址**！！

#### 2.3. 线程管理 Organizing Threads

当内核函数开始执行，如何组织GPU的线程就变成了最主要的问题了，我们必须明确，一个核函数只能有一个grid，一个grid可以有很多个块，每个块可以有很多的线程，这种分层的组织结构使得我们的并行过程更加自如灵活：

![Grids and Blocks](/images/Professional%20CUDA%20C%20Programming/Grids%20and%20Blocks.png)

一个线程块block中的线程可以完成下述协作：

-   同步
-   共享内存

**不同块内线程不能相互影响！他们是物理隔离的！**

接下来就是给每个线程一个编号了，我们知道每个线程都执行同样的一段串行代码，那么怎么让这段相同的代码对应不同的数据呢？首先第一步就是让这些线程彼此区分开，才能对应到相应从线程，使得这些线程也能区分自己的数据。如果线程本身没有任何标记，那么没办法确认其行为。  

依靠下面两个内置结构体确定线程标号：

-   blockIdx（线程块在线程网格内的位置索引）
-   threadIdx（线程在线程块内的位置索引）

注意这里的Idx是index的缩写，这两个内置结构体基于 uint3 定义，包含三个无符号整数的结构，通过三个字段来指定：

-   blockIdx.x
-   blockIdx.y
-   blockIdx.z
-   threadIdx.x
-   threadIdx.y
-   threadIdx.z

上面这两个是坐标，当然我们要有同样对应的两个结构体来保存其范围，也就是blockIdx中三个字段的范围threadIdx中三个字段的范围：

-   blockDim
-   gridDim

他们是dim3类型(基于uint3定义的数据结构)的变量，也包含三个字段x,y,z.

-   blockDim.x
-   blockDim.y
-   blockDim.z

网格和块的维度一般是二维和三维的，也就是说一个网格通常被分成二维的块，而每个块常被分成三维的线程。  

注意：dim3是手工定义的，主机端可见。uint3是设备端在执行的时候可见的，不可以在核函数运行时修改，初始化完成后uint3值就不变了。他们是有区别的！这一点必须要注意。

下面有一段代码（chapter02/checkDimension），块的索引和维度：

```C
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */

__global__ void checkIndex(void) {
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d) gridDim(%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char** argv) {
    // define total data element
    int nElem = 6;

    // define grid and block structure
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return (0);
}
```

可以运行得到不同线程分解方式：

```shell
grid.x 2 grid.y 1 grid.z 1
block.x 3 block.y 1 block.z 1
threadIdx:(0,0,0) blockIdx:(1,0,0) blockDim:(3,1,1) gridDim(2,1,1)
threadIdx:(1,0,0) blockIdx:(1,0,0) blockDim:(3,1,1) gridDim(2,1,1)
threadIdx:(2,0,0) blockIdx:(1,0,0) blockDim:(3,1,1) gridDim(2,1,1)
threadIdx:(0,0,0) blockIdx:(0,0,0) blockDim:(3,1,1) gridDim(2,1,1)
threadIdx:(1,0,0) blockIdx:(0,0,0) blockDim:(3,1,1) gridDim(2,1,1)
threadIdx:(2,0,0) blockIdx:(0,0,0) blockDim:(3,1,1) gridDim(2,1,1)
```

接下来这段代码是检查网格和块的大小的（chapter02/defineGridBlock）：

```C
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Demonstrate defining the dimensions of a block of threads and a grid of
 * blocks from the host.
 */

int main(int argc, char** argv) {
    // define total data element
    int nElem = 1024;

    // define grid and block structure
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 512;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x  = (nElem + block.x - 1) / block.x;
    printf("grid.x %d block.x %d \n", grid.x, block.x);

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return (0);
}
```

```shell
grid.x 1 block.x 1024 
grid.x 2 block.x 512 
grid.x 4 block.x 256 
grid.x 8 block.x 128
```

网格和块的维度存在几个限制因素，块大小主要与可利用的计算资源有关，如寄存器共享内存。分成网格和块的方式可以使得我们的CUDA程序可以在任意的设备上执行。

CUDA 线程全局索引的计算，是很容易混淆的概念，因为 CUDA 线程模型的矩阵排布和通常的数学矩阵排布秩序不太一样。基本关系是 Thread 在一起组成了 Block，Block 在一起组成了 Grid，所以是 Grid 包含 Block 再包含 Thread 的关系，如下图所示：

![Grids and Blocks](/images/Professional%20CUDA%20C%20Programming/Grids%20and%20Blocks.png)

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

##### 2.3.1. 一维线程计算

如下图所示，共有32个数据（位于32个方格内）需要处理，如何确定红色方框数据所在线程的位置?

![CUDA threadCounts](/images/Professional%20CUDA%20C%20Programming/CUDA%20threadCounts.jpg)

由概念部分，因为每一个线程块共有八个线程，所以 blockDim.x =8；由于红框数据位于第二个线程中（线程从0开始计数），所以 blockIdx.x = 2；又由于红框数据在第二个线程中位于五号位置（同样也是从0开始计数），所以 threadIdx.x = 5；

因此所求的红框数据应位于21号位置，计算如下：

```C
int index = threadIdx.x + blockIdex.x * blockDim.x;
          = 5 + 2 * 8;
          = 21;
```

由此便可以确实当前线程所执行数据的位置

##### 2.3.2. 多维线程计算

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

![Grids and Blocks](/images/Professional%20CUDA%20C%20Programming/Grids%20and%20Blocks.png)

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

#### 2.4. 启动一个CUDA核函数 Launching a CUDA Kernel

核函数就是在CUDA模型上诸多线程中运行的那段串行代码，这段代码在设备上运行，用NVCC编译，产生的机器码是GPU的机器码，所以我们写CUDA程序就是写核函数，第一步我们要确保核函数能正确的运行产生正切的结果，第二优化CUDA程序的部分，无论是优化算法，还是调整内存结构，线程结构都是要调整核函数内的代码，来完成这些优化的。

启动核函数，通过的以下的ANSI C 扩展出的CUDA C指令：

```C
kernel_name<<<grid,block>>>(argument list);
```

其标准C的原型就是C语言函数调用：

```C
function_name(argument list);
```

这个三个尖括号 `<<<grid,block>>>` 内是对设备代码执行的线程结构的配置（或者简称为对内核进行配置），也就是上文中提到的线程结构中的网格，块。回忆一下上文，我们通过CUDA C内置的数据类型dim3类型的变量来配置grid和block（上文提到过：在设备端访问grid和block属性的数据类型是uint3不能修改的常类型结构，这里反复强调一下）。  

通过指定grid和block的维度，我们可以配置：

-   内核中线程的数目
-   内核中使用的线程布局

我们可以使用dim3类型的grid维度和block维度配置内核，也可以使用int类型的变量，或者常量直接初始化：

```C
kernel_name<<<4,8>>>(argument list);
```

上面这条指令的线程布局是：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124175654.png)

我们的核函数是同时复制到多个线程执行的，上文我们说过一个对应问题，多个计算执行在一个数据，肯定是浪费时间，所以为了让多线程按照我们的意愿对应到不同的数据，就要给线程一个唯一的标识，由于设备内存是线性的（基本市面上的内存硬件都是线性形式存储数据的）我们观察上图，可以用threadIdx.x 和blockIdx.x 来组合获得对应的线程的唯一标识（后面我们会看到，threadIdx和blockIdx能组合出很多不一样的效果）

接下来我们就是修改代码的时间了，改变核函数的配置，产生运行出结果一样，但效率不同的代码：

1.  一个块：
   
```C
kernel_name<<<1,32>>>(argument list);
```

2.  32个块

```C
kernel_name<<<32,1>>>(argument list);
```

上述代码如果没有特殊结构在核函数中，执行结果应该一致，但是有些效率会一直比较低。

上面这些是启动部分，当主机启动了核函数，控制权马上回到主机，而不是主机等待设备完成核函数的运行，这一点上文也有提到过（就是等待hello world输出的那段代码后面要加一句）

想要主机等待设备端执行可以用下面这个指令：

```C
cudaError_t cudaDeviceSynchronize(void);
```

这是一个显示的方法，对应的也有隐式方法，隐式方法就是不明确说明主机要等待设备端，而是设备端不执行完，主机没办法进行，比如内存拷贝函数：

```C
cudaError_t cudaMemcpy(void* dst,const void * src,  
					  size_t count,cudaMemcpyKind kind);
```

这个函数上文已经介绍过了，当核函数启动后的下一条指令就是从设备复制数据回主机端，那么主机端必须要等待设备端计算完成。

**所有CUDA核函数的启动都是异步的，这点与C语言是完全不同的**

#### 2.5. 编写核函数 Writing Your Kernel

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

并行程序中经常的一种现象：**把串行代码并行化时对串行代码块for的操作**，也就是把for并行化

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

#### 2.6. 验证核函数 Verifying Your Kernel

验证核函数就是验证其正确性，chapter02/defineGridBlock 同样包含验证核函数的方法。

在开发阶段，每一步都进行验证是绝对高效的，比把所有功能都写好，然后进行测试这种过程效率高很多，同样写CUDA也是这样的每个代码小块都进行测试，看起来慢，实际会提高很多效率。  

CUDA小技巧，当我们进行调试的时候可以把核函数配置成单线程的：

```C
kernel_name<<<1,1>>>(argument list)
```

#### 2.7. 错误处理 Handling Errors

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

当然在 release 版本中可以去除这部分，**但是开发的时候一定要有的**。

#### 2.8. 编译和执行 Compiling and Executing

将上文总结的所有细节编码（chapter02/sumArraysOnGPU-small-case.cu），为另一种向量加法：

```C
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 */

void sumArraysOnHost(float* A, float* B, float* C, const int N) {
    for (int idx = 0; idx < N; idx++)
        C[idx] = A[idx] + B[idx];
}

__global__ void sumArraysOnGPU(float* A, float* B, float* C, const int N) {
    int i = threadIdx.x;

    if (i < N)
        C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 5;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float*)malloc(nBytes);
    h_B     = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes);
    gpuRef  = (float*)malloc(nBytes);

    // initialize data at host side
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    dim3 block(nElem);
    dim3 grid(1);

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    printf("Execution configure %d, %d\n", grid.x, block.x);

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // add vector at host side for result checks
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return (0);
}
```

编译结果如下：

```shell
./sumArraysOnGPU-small-case Starting...
Vector size 32
Execution configure <<<1, 32>>>
Check result success!
```

### 3. 给核函数计时 Timing Your Kernel

#### 3.1. 用CPU计时 Timing with CPU Timer

使用cpu计时的方法是测试时间的一个常用办法，在写C程序的时候最多使用的计时方法是：

```C
clock_t start, finish;
start = clock();
// 要测试的部分
finish = clock();
duration = (double)(finish - start) / CLOCKS_PER_SEC;
```

其中 clock() 是个关键的函数，“clock函数测出来的时间为进程运行时间，单位为滴答数(ticks)”；字面上理解 CLOCKS_PER_SEC 这个宏，就是没秒中多少clocks，在不同的系统中值可能不同。**必须注意的是，并行程序这种计时方式有严重问题！如果想知道具体原因，可以查询clock的源代码（c语言标准函数）**。

这里我们使用gettimeofday() 函数：

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

那么我们使用这个函数测试核函数运行时间（chapter02/sumArraysTimer）：

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void sumArrays(float* a, float* b, float* res, const int size) {
    for (int i = 0; i < size; i += 4) {
        res[i]     = a[i] + b[i];
        res[i + 1] = a[i + 1] + b[i + 1];
        res[i + 2] = a[i + 2] + b[i + 2];
        res[i + 3] = a[i + 3] + b[i + 3];
    }
}

__global__ void sumArraysGPU(float* a, float* b, float* res, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        res[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 1 << 24;
    printf("Vector size:%d\n", nElem);
    int    nByte          = sizeof(float) * nElem;
    float* a_h            = (float*)malloc(nByte);
    float* b_h            = (float*)malloc(nByte);
    float* res_h          = (float*)malloc(nByte);
    float* res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((float**)&a_d, nByte));
    CHECK(cudaMalloc((float**)&b_d, nByte));
    CHECK(cudaMalloc((float**)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(512);
    dim3 grid((nElem - 1) / block.x + 1);

    // timer
    double iStart, iElaps;
    iStart = cpuSecond();
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, nElem);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    iElaps = cpuSecond() - iStart;
    printf("Execution configuration %d,%d Time elapsed %f sec\n", grid.x, block.x, iElaps);
    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}
```

测试运行时间：

```shell
Vector size:16777216
Execution configuration<<<32768,512>>> Time elapsed 0.108600 sec
Check result success!
```

当维度调整至 256 时，速度有所提升：

```shell
Vector size:16777216
Execution configuration<<<65536,256>>> Time elapsed 0.073352 sec
Check result success!
```

主要分析计时这段，首先iStart是cpuSecond返回一个秒数，接着执行核函数，核函数开始执行后马上返回主机线程，所以我们必须要加一个同步函数等待核函数执行完毕，如果不加这个同步函数，那么测试的时间是从调用核函数，到核函数返回给主机线程的时间段，而不是核函数的执行时间。加上 `cudaDeviceSynchronize()` 函数后，计时是从调用核函数开始，到核函数执行完并返回给主机的时间段，下面图大致描述了执行过程的不同时间节点：

![Kernal Time Test](/images/Professional%20CUDA%20C%20Programming/Kernal%20Time%20Test.png)

可以大概分析下核函数启动到结束的过程：

- 主机线程启动核函数
- 核函数启动成功
- 控制返回主机线程
- 核函数执行完成
- 主机同步函数侦测到核函数执行完

我们要测试的是 2~4 的时间，但是用 CPU 计时方法，只能测试 1~5 的时间，所以测试得到的时间偏长。

#### 3.2. 用nvprof工具计时 Timing with nvprof

CUDA 5.0后有一个工具叫做nvprof的命令行分析工具，nvprof的用法如下：

```shell
nvprof [nvprof_args] <application>[application_args]
# 举例 nvprof ./sum_arrays_timer
```

![nvprof](/images/Professional%20CUDA%20C%20Programming/nvprof.png)

工具不仅给出了kernel执行的时间，比例，还有其他cuda函数的执行时间，可以看出核函数执行时间只有6%左右，其他内存分配，内存拷贝占了大部分事件，nvprof给出的核函数执行时间2.8985ms，cpuSecond计时结果是37.62ms。可见，nvprof可能更接近真实值。

nvprof这个强大的工具给了我们优化的目标，分析数据可以得出我们重点工作要集中在哪部分。

得到了实际操作值，我们需要知道的是我们能优化的极限值是多少，也就是机器的理论计算极限，这个极限我们永远也达不到，但是我们必须明确的知道，比如理论极限是2秒，我们已经从10秒优化到2.01秒了，基本就没有必要再继续花大量时间优化速度了，而应该考虑买更多的机器或者更新的设备。  

各个设备的理论极限可以通过其芯片说明计算得到，比如说：

-   Tesla K10 单精度峰值浮点数计算次数：745MHz核心频率 x 2GPU/芯片 x（8个多处理器 x 192个浮点计算单元 x 32 核心/多处理器） x 2 OPS/周期 =4.58 TFLOPS
-   Tesla K10 内存带宽峰值： 2GPU/芯片 x 256 位 x 2500 MHz内存时钟 x 2 DDR/8位/字节 = 320 GB/s
-   指令比：字节 4.58 TFLOPS/320 GB/s =13.6 个指令： 1个字节

### 4. 组织并行线程 Organizing Parallel Threads

[2.1 CUDA 编程结构 CUDA Programming Structure](#2.1%20CUDA%20编程结构%20CUDA%20Programming%20Structure) 中大概的介绍了CUDA编程的几个关键点，包括内存，kernel，以及将要讲的线程组织形式。2.1 中还介绍了每个线程的编号是依靠，块的坐标（blockIdx.x等），网格的大小（gridDim.x 等），线程编号（threadIdx.x等），线程的大小（tblockDim.x等）  

这一节就详细介绍每一个线程是怎么确定唯一的索引，然后建立并行计算，并且不同的线程组织形式是怎样影响性能的：

-   二维网格二维线程块
-   一维网格一维线程块
-   二维网格一维线程块

#### 4.1. 使用块和线程建立矩阵索引 Indexing Matrices with Blocks and Threads

多线程的优点就是每个线程处理不同的数据计算，那么怎么分配好每个线程处理不同的数据，而不至于多个不同的线程处理同一个数据，或者避免不同的线程没有组织的乱访问内存。如果多线程不能按照组织合理的干活，那么就相当于一群没训练过的哈士奇拉雪橇，往不同的方向跑，那么是没办法前进的，必须有组织，有规则的计算才有意义。  

线程模型2.1 中已经有个大概的介绍，但是下图可以非常形象的反应线程模型，不过注意硬件实际的执行和存储不是按照图中的模型来的，大家注意区分：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124224629.png)

这里 (ix,iy) 就是整个线程模型中任意一个线程的索引，或者叫做全局地址，局部地址当然就是 (threadIdx.x,threadIdx.y) 了，当然这个局部地址目前还没有什么用处，他只能索引线程块内的线程，不同线程块中有相同的局部索引值，比如同一个小区，A栋有16楼，B栋也有16楼，A栋和B栋就是blockIdx，而16就是threadIdx啦

图中的横坐标就是：

$$
ix=threadldx.x+blockIdx.x \times blockDim.x 
$$

纵坐标是：

$$
iy=threadldx.y+blockIdx.y \times blockDim.y 
$$

这样就得到了每个线程的唯一标号，并且在运行时kernel是可以访问这个标号的。前面讲过CUDA每一个线程执行相同的代码，也就是异构计算中说的多线程单指令，如果每个不同的线程执行同样的代码，又处理同一组数据，将会得到多个相同的结果，显然这是没意义的，为了让不同线程处理不同的数据，CUDA常用的做法是让不同的线程对应不同的数据，也就是用线程的全局标号对应不同组的数据。

设备内存或者主机内存都是线性存在的，比如一个二维矩阵 $(8×6)$，存储在内存中是这样的：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124225132.png)

我们要做管理的就是：

-   线程和块索引（来计算线程的全局索引）
-   矩阵中给定点的坐标（ix,iy）
-   (ix,iy)对应的线性内存的位置

线性位置的计算方法是：

$$
idx=ix+iy*nx
$$

我们上面已经计算出了线程的全局坐标，用线程的全局坐标对应矩阵的坐标，也就是说，线程的坐标(ix,iy)对应矩阵中(ix,iy)的元素，这样就形成了一一对应，不同的线程处理矩阵中不同的数据，举个具体的例子，ix=10,iy=10的线程去处理矩阵中(10,10)的数据，当然你也可以设计别的对应模式，但是这种方法是最简单出错可能最低的。  

接下来的代码来输出每个线程的标号信息（chapter02/checkThreadIndex.cu）：

```C
#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example helps to visualize the relationship between thread/block IDs and
 * offsets into data. For each CUDA thread, this example displays the
 * intra-block thread ID, the inter-block block ID, the global coordinate of a
 * thread, the calculated offset into input data, and the input data at that
 * offset.
 */

void printMatrix(int* C, const int nx, const int ny) {
    int* ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);

    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%3d", ic[ix]);
        }

        ic += nx;
        printf("\n");
    }

    printf("\n");
    return;
}

__global__ void printThreadIndex(int* A, const int nx, const int ny) {
    int          ix  = threadIdx.x + blockIdx.x * blockDim.x;
    int          iy  = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index"
           " %2d ival %2d\n",
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y,
           ix, iy, idx, A[idx]);
}

int main(int argc, char** argv) {
    printf("%s Starting...\n", argv[0]);

    // get device information
    int            dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx     = 8;
    int ny     = 6;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);

    // malloc host memory
    int* h_A;
    h_A = (int*)malloc(nBytes);

    // iniitialize host matrix with integer
    for (int i = 0; i < nxy; i++) {
        h_A[i] = i;
    }
    printMatrix(h_A, nx, ny);

    // malloc device memory
    int* d_MatA;
    CHECK(cudaMalloc((void**)&d_MatA, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

    // set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // invoke the kernel
    printThreadIndex<<<grid, block>>>(d_MatA, nx, ny);
    CHECK(cudaGetLastError());

    // free host and devide memory
    CHECK(cudaFree(d_MatA));
    free(h_A);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
```

这段代码输出了一组我们随机生成的矩阵，并且核函数打印自己的线程标号，注意，核函数能调用printf这个特性是CUDA后来加的，最早的版本里面不能printf，输出结果：

```shell
./checkThreadIndex Starting...
Using Device 0: NVIDIA GeForce RTX 3090

Matrix: (8.6)
  0  1  2  3  4  5  6  7
  8  9 10 11 12 13 14 15
 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31
 32 33 34 35 36 37 38 39
 40 41 42 43 44 45 46 47

thread_id (0,0) block_id (1,0) coordinate (4,0) global index  4 ival  4
thread_id (1,0) block_id (1,0) coordinate (5,0) global index  5 ival  5
thread_id (2,0) block_id (1,0) coordinate (6,0) global index  6 ival  6
thread_id (3,0) block_id (1,0) coordinate (7,0) global index  7 ival  7
thread_id (0,1) block_id (1,0) coordinate (4,1) global index 12 ival 12
thread_id (1,1) block_id (1,0) coordinate (5,1) global index 13 ival 13
thread_id (2,1) block_id (1,0) coordinate (6,1) global index 14 ival 14
thread_id (3,1) block_id (1,0) coordinate (7,1) global index 15 ival 15
thread_id (0,0) block_id (0,1) coordinate (0,2) global index 16 ival 16
thread_id (1,0) block_id (0,1) coordinate (1,2) global index 17 ival 17
thread_id (2,0) block_id (0,1) coordinate (2,2) global index 18 ival 18
thread_id (3,0) block_id (0,1) coordinate (3,2) global index 19 ival 19
thread_id (0,1) block_id (0,1) coordinate (0,3) global index 24 ival 24
thread_id (1,1) block_id (0,1) coordinate (1,3) global index 25 ival 25
thread_id (2,1) block_id (0,1) coordinate (2,3) global index 26 ival 26
thread_id (3,1) block_id (0,1) coordinate (3,3) global index 27 ival 27
thread_id (0,0) block_id (1,1) coordinate (4,2) global index 20 ival 20
thread_id (1,0) block_id (1,1) coordinate (5,2) global index 21 ival 21
thread_id (2,0) block_id (1,1) coordinate (6,2) global index 22 ival 22
thread_id (3,0) block_id (1,1) coordinate (7,2) global index 23 ival 23
thread_id (0,1) block_id (1,1) coordinate (4,3) global index 28 ival 28
thread_id (1,1) block_id (1,1) coordinate (5,3) global index 29 ival 29
thread_id (2,1) block_id (1,1) coordinate (6,3) global index 30 ival 30
thread_id (3,1) block_id (1,1) coordinate (7,3) global index 31 ival 31
thread_id (0,0) block_id (0,0) coordinate (0,0) global index  0 ival  0
thread_id (1,0) block_id (0,0) coordinate (1,0) global index  1 ival  1
thread_id (2,0) block_id (0,0) coordinate (2,0) global index  2 ival  2
thread_id (3,0) block_id (0,0) coordinate (3,0) global index  3 ival  3
thread_id (0,1) block_id (0,0) coordinate (0,1) global index  8 ival  8
thread_id (1,1) block_id (0,0) coordinate (1,1) global index  9 ival  9
thread_id (2,1) block_id (0,0) coordinate (2,1) global index 10 ival 10
thread_id (3,1) block_id (0,0) coordinate (3,1) global index 11 ival 11
thread_id (0,0) block_id (0,2) coordinate (0,4) global index 32 ival 32
thread_id (1,0) block_id (0,2) coordinate (1,4) global index 33 ival 33
thread_id (2,0) block_id (0,2) coordinate (2,4) global index 34 ival 34
thread_id (3,0) block_id (0,2) coordinate (3,4) global index 35 ival 35
thread_id (0,1) block_id (0,2) coordinate (0,5) global index 40 ival 40
thread_id (1,1) block_id (0,2) coordinate (1,5) global index 41 ival 41
thread_id (2,1) block_id (0,2) coordinate (2,5) global index 42 ival 42
thread_id (3,1) block_id (0,2) coordinate (3,5) global index 43 ival 43
thread_id (0,0) block_id (1,2) coordinate (4,4) global index 36 ival 36
thread_id (1,0) block_id (1,2) coordinate (5,4) global index 37 ival 37
thread_id (2,0) block_id (1,2) coordinate (6,4) global index 38 ival 38
thread_id (3,0) block_id (1,2) coordinate (7,4) global index 39 ival 39
thread_id (0,1) block_id (1,2) coordinate (4,5) global index 44 ival 44
thread_id (1,1) block_id (1,2) coordinate (5,5) global index 45 ival 45
thread_id (2,1) block_id (1,2) coordinate (6,5) global index 46 ival 46
thread_id (3,1) block_id (1,2) coordinate (7,5) global index 47 ival 47
```

也可以使用浮点数进行矩阵加法（chapter02/checkThreadIndexFloat.cu）:

```C
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void thread_index_kernel(float* A, const int num_x, const int num_y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = col * num_x + row;

    // Ensure we are within the bounds of the array
    if (row < num_x && col < num_y) {
        printf("thread_id: (%d, %d) block_id: (%d, %d) coordinate: (%d, %d) global index: %2d val: %f\n",
               threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, row, col, idx, A[idx]);
    }
}

void printMatrix(float* C, const int nx, const int ny) {
    float* ic = C;
    printf("Matrix %d,%d:\n", ny, nx);

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6f ", ic[j]);
        }

        ic += nx;
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int nx = 8, ny = 6;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);

    // Allocate memory on the host
    float* A_host = (float*)malloc(nBytes);

    // Initialize host array with some values
    for (int i = 0; i < nxy; ++i) {
        A_host[i] = (float)rand() / RAND_MAX;    // Assign random float values between 0 and 1
    }
    printMatrix(A_host, nx, ny);

    // Allocate memory on the device
    float* A_dev = NULL;
    cudaMalloc((void**)&A_dev, nBytes);

    // Copy data from host to device
    cudaMemcpy(A_dev, A_host, nBytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // Launch the kernel
    thread_index_kernel<<<grid, block>>>(A_dev, nx, ny);
    cudaDeviceSynchronize();    // Ensure kernel execution is complete

    // Free device memory
    cudaFree(A_dev);

    // Free host memory
    free(A_host);

    cudaDeviceReset();
    return 0;
}
```

输出如下：

```C
Matrix<6,8>:
0.840188 0.394383 0.783099 0.798440 0.911647 0.197551 0.335223 0.768230 
0.277775 0.553970 0.477397 0.628871 0.364784 0.513401 0.952230 0.916195 
0.635712 0.717297 0.141603 0.606969 0.016301 0.242887 0.137232 0.804177 
0.156679 0.400944 0.129790 0.108809 0.998924 0.218257 0.512932 0.839112 
0.612640 0.296032 0.637552 0.524287 0.493583 0.972775 0.292517 0.771358 
0.526745 0.769914 0.400229 0.891529 0.283315 0.352458 0.807725 0.919026 
thread_id: (0, 0) block_id: (1, 0) coordinate: (4, 0) global index:  4 val: 0.911647
thread_id: (1, 0) block_id: (1, 0) coordinate: (5, 0) global index:  5 val: 0.197551
thread_id: (2, 0) block_id: (1, 0) coordinate: (6, 0) global index:  6 val: 0.335223
thread_id: (3, 0) block_id: (1, 0) coordinate: (7, 0) global index:  7 val: 0.768230
thread_id: (0, 1) block_id: (1, 0) coordinate: (4, 1) global index: 12 val: 0.364784
thread_id: (1, 1) block_id: (1, 0) coordinate: (5, 1) global index: 13 val: 0.513401
thread_id: (2, 1) block_id: (1, 0) coordinate: (6, 1) global index: 14 val: 0.952230
thread_id: (3, 1) block_id: (1, 0) coordinate: (7, 1) global index: 15 val: 0.916195
thread_id: (0, 0) block_id: (1, 1) coordinate: (4, 2) global index: 20 val: 0.016301
thread_id: (1, 0) block_id: (1, 1) coordinate: (5, 2) global index: 21 val: 0.242887
thread_id: (2, 0) block_id: (1, 1) coordinate: (6, 2) global index: 22 val: 0.137232
thread_id: (3, 0) block_id: (1, 1) coordinate: (7, 2) global index: 23 val: 0.804177
thread_id: (0, 1) block_id: (1, 1) coordinate: (4, 3) global index: 28 val: 0.998924
thread_id: (1, 1) block_id: (1, 1) coordinate: (5, 3) global index: 29 val: 0.218257
thread_id: (2, 1) block_id: (1, 1) coordinate: (6, 3) global index: 30 val: 0.512932
thread_id: (3, 1) block_id: (1, 1) coordinate: (7, 3) global index: 31 val: 0.839112
thread_id: (0, 0) block_id: (0, 1) coordinate: (0, 2) global index: 16 val: 0.635712
thread_id: (1, 0) block_id: (0, 1) coordinate: (1, 2) global index: 17 val: 0.717297
thread_id: (2, 0) block_id: (0, 1) coordinate: (2, 2) global index: 18 val: 0.141603
thread_id: (3, 0) block_id: (0, 1) coordinate: (3, 2) global index: 19 val: 0.606969
thread_id: (0, 1) block_id: (0, 1) coordinate: (0, 3) global index: 24 val: 0.156679
thread_id: (1, 1) block_id: (0, 1) coordinate: (1, 3) global index: 25 val: 0.400944
thread_id: (2, 1) block_id: (0, 1) coordinate: (2, 3) global index: 26 val: 0.129790
thread_id: (3, 1) block_id: (0, 1) coordinate: (3, 3) global index: 27 val: 0.108809
thread_id: (0, 0) block_id: (0, 0) coordinate: (0, 0) global index:  0 val: 0.840188
thread_id: (1, 0) block_id: (0, 0) coordinate: (1, 0) global index:  1 val: 0.394383
thread_id: (2, 0) block_id: (0, 0) coordinate: (2, 0) global index:  2 val: 0.783099
thread_id: (3, 0) block_id: (0, 0) coordinate: (3, 0) global index:  3 val: 0.798440
thread_id: (0, 1) block_id: (0, 0) coordinate: (0, 1) global index:  8 val: 0.277775
thread_id: (1, 1) block_id: (0, 0) coordinate: (1, 1) global index:  9 val: 0.553970
thread_id: (2, 1) block_id: (0, 0) coordinate: (2, 1) global index: 10 val: 0.477397
thread_id: (3, 1) block_id: (0, 0) coordinate: (3, 1) global index: 11 val: 0.628871
thread_id: (0, 0) block_id: (0, 2) coordinate: (0, 4) global index: 32 val: 0.612640
thread_id: (1, 0) block_id: (0, 2) coordinate: (1, 4) global index: 33 val: 0.296032
thread_id: (2, 0) block_id: (0, 2) coordinate: (2, 4) global index: 34 val: 0.637552
thread_id: (3, 0) block_id: (0, 2) coordinate: (3, 4) global index: 35 val: 0.524287
thread_id: (0, 1) block_id: (0, 2) coordinate: (0, 5) global index: 40 val: 0.526745
thread_id: (1, 1) block_id: (0, 2) coordinate: (1, 5) global index: 41 val: 0.769914
thread_id: (2, 1) block_id: (0, 2) coordinate: (2, 5) global index: 42 val: 0.400229
thread_id: (3, 1) block_id: (0, 2) coordinate: (3, 5) global index: 43 val: 0.891529
thread_id: (0, 0) block_id: (1, 2) coordinate: (4, 4) global index: 36 val: 0.493583
thread_id: (1, 0) block_id: (1, 2) coordinate: (5, 4) global index: 37 val: 0.972775
thread_id: (2, 0) block_id: (1, 2) coordinate: (6, 4) global index: 38 val: 0.292517
thread_id: (3, 0) block_id: (1, 2) coordinate: (7, 4) global index: 39 val: 0.771358
thread_id: (0, 1) block_id: (1, 2) coordinate: (4, 5) global index: 44 val: 0.283315
thread_id: (1, 1) block_id: (1, 2) coordinate: (5, 5) global index: 45 val: 0.352458
thread_id: (2, 1) block_id: (1, 2) coordinate: (6, 5) global index: 46 val: 0.807725
thread_id: (3, 1) block_id: (1, 2) coordinate: (7, 5) global index: 47 val: 0.919026
```

#### 4.2. 二维网格和二维块 Summing Matrices with a 2D Grid and 2D Blocks

由 2.3 节的 CUDA 索引计算，可以使用以下代码打印每个线程的标号信息。可以得出二维矩阵加法核函数：

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

下面调整不同的线程组织形式，测试一下不同的效率并保证得到正确的结果，但是什么时候得到最好的效率是后面要考虑的，我们要做的就是用各种不同的相乘组织形式得到正确结果的，代码在 chapter02/sumMatrix2D.cu 中。

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
    int row    = 1 << 14;    // 2^12, 16384
    int col    = 1 << 14;    // 2^12, 16384
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

    // 1d block and 1d grid
    dim3 blockDim_1(dim_x);
    dim3 gridDim_1((sum - 1) / blockDim_1.x + 1);
    iStart = cpuSecond();
    sumMatrix<<<gridDim_1, blockDim_1>>>(A_dev, B_dev, C_dev, sum, 1);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
           gridDim_1.x, gridDim_1.y, blockDim_1.x, blockDim_1.y, iElaps);
    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, sum);

    // 2d block and 1d grid
    dim3 blockDim_2(dim_x);
    dim3 gridDim_2((row - 1) / blockDim_2.x + 1, col);
    iStart = cpuSecond();
    sumMatrix<<<gridDim_2, blockDim_2>>>(A_dev, B_dev, C_dev, row, col);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
           gridDim_2.x, gridDim_2.y, blockDim_2.x, blockDim_2.y, iElaps);
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

首先来看**二维网格二维模块**的代码：

```C
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
```

运行结果：

```shell
CPU Execution Time elapsed 0.069774 sec
GPU Execution configuration<<<(128, 128),(32, 32)>>> Time elapsed 0.004390 sec
Check result success!
```

可以看到，在 GPU 上的运行时间确实比在 CPU 上快近15倍的速度！

#### 4.3. 一维网格和一维块 Summing Matrices with a 1D Grid and 1D Blocks

接着使用**一维网格一维块**：

```C
	dim3 blockDim_1(dim_x);
    dim3 gridDim_1((sum + blockDim_1.x - 1) / blockDim_1.x);
    iStart = cpuSecond();
    sumMatrix<<<gridDim_1, blockDim_1>>>(A_dev, B_dev, C_dev, sum, 1);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
           gridDim_1.x, gridDim_1.y, blockDim_1.x, blockDim_1.y, iElaps);
    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, sum);
```

运行结果：

```shell
GPU Execution configuration<<<(524288, 1),(32, 1)>>> Time elapsed 0.000558 sec
Check result success!
```

#### 4.4. 二维网格和一维块 Summing Matrices with a 2D Grid and 1D Blocks

**二维网格一维块**：

```C
    dim3 blockDim_2(dim_x);
    dim3 gridDim_2((row - 1) / blockDim_2.x + 1, col);
    iStart = cpuSecond();
    sumMatrix<<<gridDim_2, blockDim_2>>>(A_dev, B_dev, C_dev, row, col);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d, %d),(%d, %d)>>> Time elapsed %f sec\n",
           gridDim_2.x, gridDim_2.y, blockDim_2.x, blockDim_2.y, iElaps);
    CHECK(cudaMemcpy(C_from_gpu, C_dev, nBytes, cudaMemcpyDeviceToHost));
    checkResult(C_host, C_from_gpu, sum);
```

运行结果：
```shell
GPU Execution configuration<<<(128, 4096),(32, 1)>>> Time elapsed 0.002046 sec
Check result success!
```

用不同的线程组织形式会得到正确结果，但是效率有所区别：

|     线程配置      | 执行时间 |
|:-----------------:|:--------:|
|     CPU单线程     | 0.069774 |
| (128,128),(32,32) | 0.004390 |
| (524288,1),(32,1) | 0.000558 |
| (128,4096),(32,1) | 0.002046 |

观察结果没有多大差距，但是明显比CPU快了很多，而且最主要的是用不同的线程组织模式都得到了正确结果，并且：

- 改变执行配置（线程组织）能得到不同的性能
- 传统的核函数可能不能得到最好的效果
- 一个给定的核函数，通过调整网格和线程块大小可以得到更好的效果

### 5. 设备管理 Managing Devices

用 CUDA 的时候一般有两种情况，一种自己写完自己用，使用本机或者已经确定的服务器，这时候只要查看说明书或者配置说明就知道用的什么型号的GPU，以及GPU的所有信息，但是如果写的程序是通用的程序或者框架，在使用 CUDA 前要先确定当前的硬件环境，这使得我们的程序不那么容易因为设备不同而崩溃，本文介绍两种方法，第一种适用于通用程序或者框架，第二种适合查询本机或者可登陆的服务器，并且一般不会改变，那么这时候用一条 nvidia 驱动提供的指令查询设备信息就很方便了。

#### 5.1. API查询GPU信息 Using the Runtime API to Query GPU Information

使用代码 chapter02/checkDeviceInfor.cu 可以在软件内查询信息：

```C
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    printf("%s Starting ...\n", argv[0]);

    int         deviceCount = 0;
    cudaError_t error_id    = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n ->%s\n",
               (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d:\"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version         %d.%d  /  %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:   %d.%d\n",
           deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                %.2f GBytes (%zu bytes)\n",
           (float)deviceProp.totalGlobalMem / pow(1024.0, 3), deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                               %.0f MHz (%0.2f GHz)\n",
           deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf("  Memory Bus width:                             %d-bits\n",
           deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize) {
        printf("  L2 Cache Size:                            	%d bytes\n",
               deviceProp.l2CacheSize);
    }
    printf("  Max Texture Dimension Size (x,y,z)            1D=(%d),2D=(%d,%d),3D=(%d,%d,%d)\n",
           deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers       1D=(%d) x %d,2D=(%d,%d) x %d\n",
           deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
           deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
           deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory               %lu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:      %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block:%d\n",
           deviceProp.regsPerBlock);
    printf("  Wrap size:                                    %d\n", deviceProp.warpSize);
    printf("  Maximun number of thread per multiprocesser:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximun number of thread per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Maximun size of each dimension of a block:    %d x %d x %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Maximun size of each dimension of a grid:     %d x %d x %d\n",
           deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximu memory pitch                           %lu bytes\n", deviceProp.memPitch);
    printf("----------------------------------------------------------\n");
    printf("Number of multiprocessors:                      %d\n", deviceProp.multiProcessorCount);
    printf("Total amount of constant memory:                %4.2f KB\n",
           deviceProp.totalConstMem / 1024.0);
    printf("Total amount of shared memory per block:        %4.2f KB\n",
           deviceProp.sharedMemPerBlock / 1024.0);
    printf("Total number of registers available per block:  %d\n",
           deviceProp.regsPerBlock);
    printf("Warp size                                       %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor:     %d\n",
           deviceProp.maxThreadsPerMultiProcessor / 32);
    return EXIT_SUCCESS;
}
```

运行的效果如下：

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

#### 5.2. NVIDIA-SMI

也可以使用 nvidia-smi nvidia 驱动程序内带的一个工具返回当前环境的设备信息：

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
nvidia-smi -q -i 0                             # 多设备把0改成对应的设备号
nvidia-smi -q -i 0 -d MEMORY | tail -n 5       # 得到内存信息
nvidia-smi -q -i 0 -d UTILIZATION | tail -n 4  # 利用率
```

这些指令可以在脚本中帮我们得到设备信息，比如我们可以写通用程序时在编译前执行脚本来获取设备信息，然后在编译时固化最优参数，这样程序运行时就不会被查询设备信息的过程浪费资源。  

也就是我们可以用以下两种方式编写通用程序：

-   运行时获取设备信息：
    -   编译程序
    -   启动程序
    -   查询信息，将信息保存到全局变量
    -   功能函数通过全局变量判断当前设备信息，优化参数
    -   程序运行完毕
-   编译时获取设备信息
    -   脚本获取设备信息
    -   编译程序，根据设备信息调整固化参数到二进制机器码
    -   运行程序
    -   程序运行完毕

指令 nvidia-smi -q -i 0 可以提取以下我们要的信息：

-   MEMORY
-   UTILIZATION
-   ECC
-   TEMPERATURE
-   POWER
-   CLOCK
-   COMPUTE
-   PIDS
-   PERFORMANCE
-   SUPPORTED_CLOCKS
-   PAGE_RETIREMENT
-   ACCOUNTING

至此，CUDA 的编程模型大概就是这些了，核函数，计时，内存，线程，设备参数，这些足够能写出比CPU快很多的程序了。从下一章开始，深入硬件研究背后的秘密

---

## 参考引用

### 书籍出处

- [CUDA C编程权威指南](asset/CUDA%20&%20GPU%20Programming/CUDA%20C编程权威指南.pdf)
- [Professional CUDA C Programming](asset/CUDA%20&%20GPU%20Programming/Professional%20CUDA%20C%20Programming.pdf)

### 网页链接

- [人工智能编程 | 谭升的博客](https://face2ai.com/program-blog/)