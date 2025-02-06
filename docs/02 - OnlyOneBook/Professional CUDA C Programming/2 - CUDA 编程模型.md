## CUDA C 编程入门 2 - CUDA 编程模型 CUDA Programming Model

---

### 1. 概念 Concepts

**内核（Kernel）**

CUDA C 通过扩展标准 C 语法，允许开发者定义特殊的 **核函数（Kernel Function）**。与传统 C 函数不同，核函数在被调用时会被 N 个 CUDA 线程并行执行 N 次，这种 **SIMT（单指令多线程）** 执行模式是 GPU 并行计算的核心机制。

核函数需使用 `__global__` 修饰符声明，并通过 `<<<...>>>` 语法指定执行配置（Execution Configuration）。该语法定义了线程网格（Grid）和线程块（Block）的维度结构。每个执行核函数的线程可通过内置变量（如 `threadIdx`、`blockIdx`）访问其唯一的逻辑坐标。

**线程层次（Thread Hierarchy）**

CUDA采用三级线程层次模型：**线程（Thread）→ 线程块（Block）→ 网格（Grid）**。这种分层结构不仅映射GPU的物理硬件架构（流多处理器→ CUDA核心），也为程序员提供了灵活的并行任务组织方式。

在CUDA编程模型中，线程通过多维索引进行组织。每个线程块（Thread Block）最多可包含1024个线程（具体上限取决于GPU架构），并通过以下三维内置变量进行坐标定位：

-   `threadIdx.[x/y/z]`：当前线程在其所属线程块内的三维坐标
-   `blockIdx.[x/y/z]`：当前线程块在网格中的三维坐标
-   `blockDim.[x/y/z]`：定义线程块各维度的线程数量，即一个线程块中包含多少个线程
-   `gridDim.[x/y/z]`：定义网格各维度的线程块数量，即一个网格中包含多少个block

索引计算公式：  
全局索引 = `blockIdx.[dim] * blockDim.[dim] + threadIdx.[dim]

### 2. CUDA 编程模型概述 Introducing the CUDA Programming Model 

CUDA 编程模型作为连接应用程序与异构计算硬件的桥梁，采用编译型语言范式实现高效执行。与 OpenCL 采用的运行时**即时编译（JIT）机制**不同，CUDA C 需要通过预编译和链接生成可直接在包含 GPU 的异构系统中执行的二进制代码。下图展示了该模型的分层架构：

![CUDA Model](/images/Professional%20CUDA%20C%20Programming/CUDA%20Model.png)

模型中通信抽象层（Communication Abstraction）划分了编程模型与底层编译器/库函数的界限。编程模型的核心要素包括开发者可直接控制的语法规范、内存架构和线程组织机制，这些要素共同决定了异构计算设备的工作模式。

GPU编程的核心组件可归纳为：

-   **核函数（Kernel）**：并行计算单元
-   **内存体系**：层次化存储管理
-   **线程架构**：多维网格组织
-   **流处理器**：异步任务管道

值得注意的是，这些设计理念不仅适用于CPU+GPU异构系统，也可扩展至其他异构计算架构。 

CUDA 架构的独特性体现在两大核心机制：

1.  **线程层次化组织架构**：通过网格（Grid）->块（Block）->线程（Thread）三级结构实现大规模并行
2.  **内存层次化存储体系**：包含寄存器、共享内存、全局内存等多级存储结构

从软件开发视角，CUDA 应用开发遵循三层次方法论：

- **领域层（Domain Layer）**：聚焦问题域的数据特征与算法特性，进行并行化可行性分析。重点在于数据依赖性解耦和计算任务分解，为并行执行奠定基础。
- **逻辑层（Logical Layer）**：通过CUDA线程模型实现并行逻辑设计。此阶段需构建多维线程拓扑结构，设计内存访问模式，并建立主机-设备协作机制。CUDA的线程抽象层提供了独特的可扩展性保障，确保程序能自适应不同计算能力的GPU设备。
- **硬件层（Hardware Layer）**：关注物理硬件的执行特征，包括 SM（流式多处理器）的线程调度机制、内存带宽的优化和利用、流水线指令的特性匹配等

#### 2.1. CUDA 编程结构 CUDA Programming Structure

在异构计算架构中，系统通常由多个 CPU 和 GPU 通过 PCIe 总线构建而成，形成物理隔离的计算单元。这种架构特性要求开发者必须明确区分两类关键存储区域：

-   **主机（Host）**：指 CPU 及其关联的内存空间（系统内存）
-   **设备（Device）**：指 GPU 及其专用显存空间（全局内存）

需要特别指出的是，在CUDA 6.0引入统一内存（Unified Memory）机制之前，这两类存储空间在物理硬件层和软件管理层均保持严格隔离。为加深对CUDA内存模型本质的理解，本教程将暂时采用传统显式内存拷贝机制进行编写代码。

CUDA 程序的典型处理流程遵循如下：

1.  **数据传输阶段**：将待处理数据从主机内存拷贝至设备显存
2.  **核心计算阶段**：调用CUDA核函数（Kernel）在GPU上进行并行计算
3.  **结果回传阶段**：将处理结果从设备显存拷贝回主机内存

下图展示了完整的 CUDA 应用执行顺序：

![Process Procedure](/images/Professional%20CUDA%20C%20Programming/Process%20Procedure.png)

-   **串行执行域**：常规串行代码在主机 CPU 上执行
-   **并行执行域**：计算密集型任务通过核函数在 GPU 上并行处理
-   **异步执行机制**：核函数调用采用非阻塞模式，主机线程在发起核函数后立即获得控制权，实现 CPU-GPU 的异步流水线执行（图中核函数被调用后控制马上归还主机线程，也就是在第一个并行代码执行时，很有可能第二段host代码已经开始同步执行了）

#### 2.2. 内存管理 Managing Memory

在异构计算环境中，内存管理呈现多维层次特征。与传统C/C++程序类似，CUDA编程涉及三类存储空间的管理：

-   **寄存器空间**：由编译器自动管理
-   **栈空间**：函数局部变量存储区，系统自动分配回收
-   **堆空间**：开发者显式控制生命周期

下面表格有一些主机 API 和 CUDA C 的 API 的对比：

| 标准C函数 | CUDA C 函数 |   说明   |
|:---------:|:-----------:|:--------:|
|  malloc   | cudaMalloc  | 设备堆内存动态分配 |
|  memcpy   | cudaMemcpy  | 跨存储空间数据拷贝 |
|  memset   | cudaMemset  | 设备内存初始化 |
|   free    |  cudaFree   | 设备堆内存释放 |

> 注意：CUDA API也提供 cudaMallocHost 等主机端内存管理函数，但常规主机内存建议使用标准库管理

下面介绍核心数据传输函数原型：

```C
cudaError_t cudaMemcpy(void* dst, 
                      const void* src,
                      size_t count,
                      cudaMemcpyKind kind);
```

这个函数是内存拷贝过程，传输方向通过 `kind` 参数指定：

-   `cudaMemcpyHostToHost`：主机内存间拷贝
-   `cudaMemcpyHostToDevice`：主机→设备传输
-   `cudaMemcpyDeviceToHost`：设备→主机回传
-   `cudaMemcpyDeviceToDevice`：设备内存间拷贝

这四个过程的方向可以清楚的从字面上看出来，如果函数执行成功，则会返回 `cudaSuccess`，否则返回 `cudaErrorMemoryAllocation`

使用下面这个指令可以把上面的错误代码翻译成详细信息：

```C
char* cudaGetErrorString(cudaError_t error)
```

CUDA设备内存具有多级层次结构（示意图如下），不同层级的内存具有显著的性能差异：

-   **全局内存（Global Memory）**：设备主存，容量大但延迟高
-   **共享内存（Shared Memory）**：块内线程共享的片上缓存，低延迟访问

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124154320.png)


后面我们会特别详细深入分析各内存层级特性及优化策略。

这里我们来个例子，实现两个向量的逐元素相加（c[i] = a[i] + b[i]），流程如下：

1.  主机端分配初始化输入数组 `a_h`、`b_h`
2.  设备端分配存储空间 `a_d`、`b_d`、`res_d`   
3.  数据拷贝：主机→设备（`a_h`→`a_d`, b_h→`b_d`）
4.  启动核函数执行并行加法
5.  结果回传：设备→主机（`res_d`→`res_from_gpu_h`）  
6.  释放所有设备内存

代码在 `chapter02/sumArrays.cu` 中：

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

输出如下：

```shell
Vector size:32
Execution configuration<<<32,1>>>
Check result success!
```

解释下内存管理部分的代码：

```C
cudaMalloc((float**)&a_d, nByte);
```

分配设备端的内存空间，为了区分设备和主机端内存，我们可以给变量加后缀或者前缀 `h_`表示 host，`d_` 表示device。**一个经常会发生的错误就是混用设备和主机的内存地址**！！

#### 2.3. 线程管理 Organizing Threads

在 CUDA 计算模型中，当核函数（Kernel Function）开始执行时，如何高效地组织 GPU 线程成为核心问题。CUDA 采用层次化的线程组织结构，其中：

-   **一个核函数（Kernel）仅对应一个线程网格（Grid）。**
-   **一个线程网格由多个线程块（Block）组成。**
-   **每个线程块内部包含多个线程（Thread）。**

这种分层结构使得并行计算具有更强的灵活性和可扩展性。

在一个线程块（Block）内部，线程可以进行如下协作：

-   **同步（Synchronization）**：线程可以使用 `__syncthreads()` 进行同步，以确保某些计算步骤在所有线程执行完毕后再继续。
-   **共享内存（Shared Memory）**：同一线程块的所有线程可以访问共享内存，提高数据复用效率，减少全局内存访问开销。

然而，不同线程块之间的线程是**物理隔离的**，它们不能直接通信或共享数据。

CUDA 采用线程索引（Thread Indexing）来唯一标识每个线程，以便将同一段核函数代码映射到不同的数据元素。为了实现这一目标，每个线程需要明确自身的标识，而 CUDA 提供了以下两个关键的内置结构体：

1. **线程块索引（blockIdx）**：表示当前线程块在整个线程网格（Grid）中的位置。
2. **线程索引（threadIdx）**：表示当前线程在所属线程块（Block）中的位置。

这两个结构体均为 **`uint3` 类型**，即包含 `x, y, z` 三个无符号整数字段，以支持一维、二维或三维的线程布局：

-   `blockIdx.x`, `blockIdx.y`, `blockIdx.z` —— 线程块的索引
-   `threadIdx.x`, `threadIdx.y`, `threadIdx.z` —— 线程在块内的索引

此外，为了确定线程的组织规模，CUDA 还提供了两个对应的结构体来存储网格和线程块的维度信息：

-   **`gridDim`（网格维度）**：指定线程网格中块的数量，包含 `x, y, z` 三个字段。
-   **`blockDim`（块维度）**：指定每个线程块中线程的数量，包含 `x, y, z` 三个字段。

这些结构体同样支持一维、二维和三维布局：

-   `gridDim.x`, `gridDim.y`, `gridDim.z` —— 线程网格的维度（即网格中包含多少个线程块）。
-   `blockDim.x`, `blockDim.y`, `blockDim.z` —— 线程块的维度（即块中包含多少个线程）。

在实际应用中，线程网格通常被划分为二维的线程块，而线程块内部往往采用三维组织方式，以适应不同计算任务的需求。

**`dim3` 与 `uint3` 的区别**

需要注意的是，CUDA 提供的 **`dim3` 类型** 和 **`uint3` 类型** 存在重要区别：

-   **`dim3`** 是主机端（Host）可见的数据类型，用户可以在主机端代码中定义 `dim3` 变量，并用于内核函数的线程配置。
-   **`uint3`** 是设备端（Device）可见的数据结构，仅在核函数执行时有效。一旦核函数启动，其值在整个执行过程中保持不变，且不能被修改。

理解这一差异对于正确配置 CUDA 线程层次结构至关重要。

下面有一段代码 `chapter02/checkDimension.cu`，是块的索引和维度：

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

解释这段代码的输出结果，根据上文网格及线程块的定义，一共有六个元素，要求每个线程块为三个，则 $6/3=2$ 个 网格。只有 6 个元素，那么 y 和 z 坐标都为 1，所以网格及线程块的索引固定为 `gridDim(2,1,1)` 和 `blockDim:(3,1,1)`。因为 y 和 z 坐标都为 1，所以 `threadIdx` 和 `blockIdx` 的 y、z 固定为 0。

接下来这段代码是检查网格和块的大小的 `chapter02/defineGridBlock.cu`：

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

输出结果如下：

```shell
grid.x 1 block.x 1024 
grid.x 2 block.x 512 
grid.x 4 block.x 256 
grid.x 8 block.x 128
```

CUDA 网格（Grid）和线程块（Block）的划分受多种限制因素影响，其中最主要的因素包括可用的计算资源，例如寄存器、共享内存等。合理地划分网格和线程块，使 CUDA 代码能够适应不同计算设备的架构，最大程度地提高计算效率。

CUDA 线程全局索引的计算，是很容易混淆的概念。在 CUDA 线程模型中，线程（Thread）是最小的执行单元，多个线程组成一个线程块（Block），多个线程块共同构成网格（Grid）。即 Grid 由多个 Block 组成，而 Block 由多个 Thread 组成，其层次关系如下所示：

![Grids and Blocks](/images/Professional%20CUDA%20C%20Programming/Grids%20and%20Blocks.png)

在该示例中，一个 Grid 由 6 个线程块（Blocks）组成，每个线程块包含 15 个线程（Threads）。需要注意的是，CUDA 允许的网格和线程块规模受到硬件约束，例如：

-   **Grid 规模限制**：在一维网格（1D Grid）情况下，最大允许网格大小为 $2^{31}−1$；
-   **Block 规模限制**：单个线程块的最大线程数通常为 1024。

值得注意的是，线程块的划分仅是逻辑上的，物理上线程之间并无明显的边界。CUDA 允许在 GPU 上启动大量线程，其总数可以远超 GPU 核心数量，以充分利用硬件资源。通常，合理的线程数量应至少等于计算核心数量，这样才可能充分发挥硬件的计算能力。

在 CUDA 编程中，核函数的执行配置采用 `<<<grid_size, block_size>>>` 语法，其中：

-   `grid_size`： 指定网格（Grid）中线程块（Block）的数量，对应 `gridDim.x`，取值范围为 $[0,gridDim.x−1]$；
-   `block_size`： 指定线程块（Block）中线程（Thread）的数量，对应 `blockDim.x`，取值范围为 $[0,blockDim.x−1]$。

核函数调用示例如下：

```C
kernel_fun<<<grid_size, block_size>>>();
```

例如，若定义一个 $2 \times 3 \times 1$ 的网格、$6 \times 2 \times 2$ 的线程块，可以这么写:

```C
dim3 grid_size(2, 3);  // or dim3 grid_size(2, 3, 1);
dim3 block_size(6, 2, 2);

kernel_ful<<<grid_size, block_size>>>();
```

解释：
-   **网格（Grid）** 由 $2 \times 3 = 6$ 个线程块组成；
-   **每个线程块（Block）** 由 $6 \times 2 \times 2 = 24$ 个线程组成。

##### 2.3.1. 一维线程计算 One-dimensional thread computing

如下图所示，共有32个数据（位于32个方格内）需要处理，如何确定红色方框数据所在线程的位置?

![CUDA threadCounts](/images/Professional%20CUDA%20C%20Programming/CUDA%20threadCounts.jpg)

由概念部分，因为每一个线程块共有八个线程，所以 $blockDim.x =8$；由于红框数据位于第二个线程中（线程从0开始计数），所以 $blockIdx.x = 2$；又由于红框数据在第二个线程中位于五号位置（同样也是从0开始计数），所以 $threadIdx.x = 5$；

因此所求的红框数据应位于21号位置，计算如下：

```C
int index = threadIdx.x + blockIdex.x * blockDim.x;
          = 5 + 2 * 8;
          = 21;
```

由此便可以确实当前线程所执行数据的位置

##### 2.3.2. 多维线程计算 Multi-dimensional threads computing

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

以下图举例，计算 $Thread(2, 2)$ 的索引值，调用核函数的线程配置代码如下:

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

计算结果 $Thread(2, 2)$ 为 72，符合预期

上面的九种组织情况都可以视为是**三维网格三维线程块**的情况，只是比一维或者二维的时候，其他维度为 1 而已。若是都把它们都看成三维格式，这样不管哪种线程组织方式，都可以套用**三维网格三维线程块**的计算方式，整理如下，

```C
// 线程块索引
int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
// 局部线程索引
int threadId = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
// 全局线程索引
int id = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;
```

#### 2.4. 启动一个CUDA核函数 Launching a CUDA Kernel

CUDA 核函数（Kernel）是在 GPU 设备上并行运行的代码片段，由 `nvcc` 编译后生成 GPU 机器码。编写 CUDA 程序的核心工作就是编写核函数，并确保其正确执行以产生预期结果。优化 CUDA 程序时，无论是改进算法，还是调整内存结构和线程布局，最终都涉及到对核函数内部代码的优化。

CUDA 提供了基于 ANSI C 的扩展语法，用于在 GPU 上启动核函数：

```C
kernel_name<<<grid, block>>>(argument list);
```

其标准 C 语言的等效形式是普通 C 函数调用：

```C
function_name(argument list);
```

其中，`<<<grid, block>>>` 指定了 GPU 端的线程组织结构，即网格（Grid）和线程块（Block）。CUDA 提供 `dim3` 数据类型来配置 `grid` 和 `block` 的维度，而在设备端，这些信息以 `uint3` 结构体的形式提供，只读且不可修改。（上文提到过：在设备端访问 grid 和 block 属性的数据类型是 `uint3` **不能修改的常类型结构**，这里反复强调一下）。  

通过调整 `grid` 和 `block` 的配置，可以灵活控制：

-   **线程总数**：决定计算任务的并行度。
-   **线程布局**：影响数据访问模式及计算效率。

可以使用 dim3 类型的 grid 维度和 block 维度配置内核，也可以使用 int 类型的变量，或者常量直接初始化：

```C
kernel_name<<<4, 8>>>(argument list);
```

上面这条指令的线程布局是：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124175654.png)

在上述线程布局示意图中，每个核函数实例都会在多个线程上并行执行。然而，如果多个线程执行相同的数据计算，则会造成计算资源浪费。因此，为了确保每个线程处理不同的数据，我们需要为其分配唯一的索引。

由于 GPU 设备端的全局内存是线性存储的（大多数现代存储硬件均采用线性存储方式），我们可以通过 `threadIdx.x` 和 `blockIdx.x` 的组合来唯一标识线程。例如：

```C
int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
```

这种索引计算方式确保不同的线程处理不同的数据，从而实现高效的并行计算。此外，`threadIdx` 和 `blockIdx` 的不同组合方式能够适应多种计算需求，例如一维、二维或三维数据映射，后续章节将进一步探讨其应用。

例如，以下两种核函数启动方式均可产生 32 个线程但效率不同的代码：
   
```C
kernel_name<<<1, 32>>>(argument list);
kernel_name<<<32, 1>>>(argument list);
```

尽管二者执行结果相同，但性能可能存在显著差异，具体取决于线程的调度及数据访问方式。

默认情况下，CUDA 核函数启动后，主机（Host）不会等待设备（Device）执行完成，而是立即返回控制权。因此，若主机端代码依赖于设备端计算结果，需要显式同步：

```C
cudaError_t cudaDeviceSynchronize(void);
```

此外，某些操作（如 `cudaMemcpy`）隐式实现同步，即主机必须等待设备完成计算后，才能执行数据传输：

```C
cudaError_t cudaMemcpy(void* dst,
					   const void * src,  
					   size_t count,
					   cudaMemcpyKind kind);
```

这个函数上文已经介绍过了，当核函数启动后的下一条指令就是从设备复制数据回主机端，那么主机端必须要等待设备端计算完成。

**所有CUDA核函数的启动都是异步的，这点与C语言是完全不同的**

#### 2.5. 编写核函数 Writing Your Kernel

CUDA 核函数的声明遵循特定的格式，通常采用以下模板化方式：

```C
__global__ void kernel_name(argument list);
```

其中，`__global__` 是 CUDA C 的限定符，标识该函数为设备端执行的核函数。此外，CUDA C 还提供了其他限定符，与标准 C 语言不同：

| 限定符        | 执行位置  | 说明           | 备注         |
|------------|-------|--------------|------------|
| __global__ | 设备端执行 | 由主机调用，运行于设备端 | 必须返回 void  |
| __device__ | 设备端执行 | 仅设备端调用       |            |
| __host__   | 主机端执行 | 仅主机端调用       | 可用于标识主机端函数 |

此外，某些函数可同时声明为 `__device__` 和 `__host__`，使其能够在主机和设备端均可调用。例如：

```C
__host__ __device__ float add(float a, float b) {
    return a + b;
}
```

这种声明方式会让 `nvcc` 生成两份不同的机器码，分别用于主机和设备端。

在编写 CUDA 核函数时，需要遵循以下限制：

1. 只能访问设备内存（不能直接访问主机内存）。
2. 必须具有 `void` 返回类型（不支持返回值）。
3. 不支持可变参数（`...` 语法）。
4. 不支持静态变量（`static`）。
5. 具有显式的异步执行行为。

在并行程序设计中，通常的优化思路是**将串行代码并行化**。典型示例如下：

**串行实现**：

```C
void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int i = 0; i < N; i++)
	C[i] = A[i] + B[i];
}
```

**并行实现**：

```C
__global__ void sumArraysOnGPU(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```

在 CUDA 计算模型下，可以通过线程并行化 `for` 循环的计算任务，从而显著提升执行效率。后续内容将介绍如何使用核函数实现这一并行化过程。

#### 2.6. 验证核函数 Verifying Your Kernel

验证核函数的正确性是 CUDA 开发过程中至关重要的一步。`chapter02/defineGridBlock` 中也提供了核函数验证的方法，检验计算结果是否符合预期。

在开发阶段，每个代码模块都应尽早进行独立测试，而不是等到所有功能实现后再进行整体测试。这种逐步验证的方式通常比一次性测试完整功能更高效，能够及早发现并修正潜在问题，从而加快整体开发进度。

在 CUDA 代码调试时，一个常用的小技巧是将核函数配置为**单线程执行**，便于逐步检查计算逻辑。例如：

```C
kernel_name<<<1, 1>>>(argument list)
```

这种方式能够使核函数在单个线程中运行，从而简化调试过程，便于定位问题。在核函数通过验证后，再逐步调整 `grid` 和 `block` 的配置，以适应实际的并行计算需求。

#### 2.7. 错误处理 Handling Errors

在任何编程环境中，错误处理都是必不可少的。早期的编码错误通常可以由编译器检测，而内存访问错误也往往也可以在运行时出现。但某些**逻辑错误**往往难以察觉，甚至可能在代码上线运行后才暴露出来。更麻烦的是，一些错误的复现并不稳定，它们可能只在特定条件下触发，而一旦发生，后果可能十分严重。

CUDA 编程的一个特殊地方在于其**异步执行特性**，即当某条指令出现错误时，错误可能不会立刻出现，而是在执行的某个时刻才被触发。这给错误定位带来了很大困难，因此在 CUDA 编程中，采用**防御性编程**手段至关重要。

为简化错误检查，可以在代码库的头文件中定义如下错误检查宏：

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

该宏可以用于检查 CUDA API 调用，例如：

```C
CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));
```

如果 `cudaMemcpy` 或其之前的异步操作失败，该宏会输出**错误代码**、**错误原因**以及**发生错误的文件和行号**，然后终止程序，以便及时发现问题。

此外，它也可以用于检查**核函数调用**是否存在异常：

```C
kernel_function<<<grid, block>>>(argument list);
CHECK(cudaDeviceSynchronize());
```

`cudaDeviceSynchronize()` 可确保核函数执行完成，并在此时检查是否有错误发生。

在**开发阶段**，最好启用错误检查，以便尽早发现问题。然而，在**发布（release）版本**中，出于性能考虑，可以去除这些检查代码，以避免额外的同步开销。不过，在调试和优化过程中，错误检查应始终保持开启，以确保程序的正确性和稳定性。

#### 2.8. 编译和执行 Compiling and Executing

将上文总结的所有内容重新编码 `chapter02/sumArraysOnGPU-small-case.cu`，为另一种向量加法：

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

使用 CPU 计时是一种常见的时间测量方法。在 C 语言中，最常用的计时方式是使用 `clock()` 函数：

```C
clock_t start, finish;
start = clock();
// 要测试的部分
finish = clock();
duration = (double)(finish - start) / CLOCKS_PER_SEC;
```

其中，`clock()` 返回进程运行的**时钟滴答数（ticks）**，单位通常为 CPU 时钟周期，而 `CLOCKS_PER_SEC` 代表每秒的时钟滴答数，这个值可能因系统而异。

>⚠ **重要提示**  
>这种计时方法在**并行程序**中存在严重问题！`clock()` 计算的是**进程级时间**，并不能准确反映核函数的执行时间，尤其是当 GPU 执行任务时，它可能与 CPU 并行运行，使得 `clock()` 计时结果失真。如果想了解更深入的原因，可以查看 C 语言标准库 `clock()` 的实现源码。

相比 `clock()`，`gettimeofday()` 提供了更高精度的时间测量方法：

```C
#include <sys/time.h>
double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec * 1e-6);
}
```

`gettimeofday()` 是 Linux 下的系统调用，它返回从 **1970 年 1 月 1 日 00:00:00**（Unix 纪元）到当前的时间，单位为秒（`tv_sec`）和微秒（`tv_usec`）。通过将 `tv_usec` 转换为秒，我们可以得到一个精确到微秒的计时器。需要头文件 `sys/time.h`

在 CUDA 代码中，可以使用如下方式来计时核函数的执行时间：

```C
double iStart = cpuSecond();
kernel_name<<<grid, block>>>(argument list);
cudaDeviceSynchronize();
double iElaps = cpuSecond() - iStart;
printf("Kernel execution time: %f sec\n", iElaps);
```

那么我们使用这个函数测试核函数运行时间 `chapter02/sumArraysTimer.cu`：

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

主要分析计时这段，`iStart` 通过 `cpuSecond()` 获取当前时间戳（秒级）。由于 CUDA 内核函数的调用是异步的，调用后主机线程立即返回，因此需要 `cudaDeviceSynchronize()` 以确保计时精确。

若缺少同步，测得的时间范围为从内核函数调用到主机线程获取控制权的时间，而非内核实际执行时间。加入 `cudaDeviceSynchronize()` 后，计时范围调整为从内核启动到执行完毕并返回主机，确保测量的是实际计算时间。

![Kernal Time Test](/images/Professional%20CUDA%20C%20Programming/Kernal%20Time%20Test.png)

可以大概分析下核函数启动到结束的过程：

1. **主机线程启动核函数**
2. **核函数启动成功**
3. **控制返回主机线程**
4. **核函数执行完成**
5. **主机同步函数检测到核函数执行完成**

理想情况下，我们希望测量 2~4 的时间，但 CPU 计时方法只能测得 1~5，使得测量结果偏长。因此，合理的同步机制对于精确评估 CUDA 计算性能至关重要。

#### 3.2. 用nvprof工具计时 Timing with nvprof

从 CUDA 5.0 开始，NVIDIA 提供了 `nvprof` 这一命令行分析工具，用于精确测量 CUDA 应用的执行时间。其基本用法如下：

```shell
nvprof [nvprof_args] <application>[application_args]
# 举例 
nvprof ./sum_arrays_timer
```

![nvprof](/images/Professional%20CUDA%20C%20Programming/nvprof.png)

该工具不仅能够报告核函数（kernel）的执行时间及占比，还能分析 CUDA 相关 API（如内存分配、数据拷贝）的耗时情况。例如，在 `nvprof` 的分析结果中，核函数执行时间仅占 6%，而内存操作占据了主要部分。测得的核函数执行时间为 **2.8985ms**，而 `cpuSecond()` 计时结果却为 **37.62ms**，可见 `nvprof` 的测量更接近实际值。

`nvprof` 提供了优化的方向，帮助我们明确应关注的性能瓶颈。然而，优化的最终目标不仅是提升当前性能，还需明确 **理论计算极限**。理论极限是硬件决定的，即便再优化也无法突破，因此在接近极限时，应考虑升级设备而非投入更多时间优化。例如，若某计算的理论极限是 2s，而当前已优化至 **2.01s**，则继续优化的收益极低，反而更适合增加算力资源。

不同设备的计算极限可以根据芯片参数估算，例如 **Tesla K10** 的性能计算如下：

- **单精度峰值计算能力**：

$$745MHz×2GPUs×(8SMs×192ALUs×32cores/SM)×2OPS/cycle=4.58TFLOPS$$

-   **内存带宽峰值**：
$$2GPUs×256bits×2500MHz×2DDR/8bits/byte=320GB/s$$
-   **算术强度（FLOP/Byte）**：
$$\frac {4.58TFLOPS}{320GB/s}​=13.6FLOP/Byte$$

### 4. 组织并行线程 Organizing Parallel Threads

[2.1 CUDA 编程结构 CUDA Programming Structure](#2.1%20CUDA%20编程结构%20CUDA%20Programming%20Structure) 中，我们概述了 CUDA 编程的核心概念，包括内存管理、核函数（kernel）执行以及线程组织方式。2.1 中还介绍了每个线程的编号是依靠，块的坐标（blockIdx.x等），网格的大小（gridDim.x 等），线程编号（threadIdx.x等），线程的大小（tblockDim.x等）  

本节将深入探讨线程索引的计算方式，并分析不同的线程组织形式如何影响计算性能：

-   二维网格二维线程块
-   一维网格一维线程块
-   二维网格一维线程块

#### 4.1. 使用块和线程建立矩阵索引 Indexing Matrices with Blocks and Threads

多线程计算的核心优势在于并行处理不同的数据，但如何合理分配计算任务至关重要。如果多个线程访问相同的数据，可能会导致竞争条件 (race condition) 或冗余计算；如果线程访问数据无序，则可能引发内存访问冲突，降低计算效率。打个比方，相当于一群未经训练的雪橇犬拉动雪橇，如果方向不一致，最终难以前进。要确保高效计算，必须遵循合理的线程组织方式。  

在 [2.1. CUDA 编程结构 CUDA Programming Structure](#2.1.%20CUDA%20编程结构%20CUDA%20Programming%20Structure) 中，我们已经介绍了线程索引的基本概念，下面的图示则更加直观地展现了线程模型的结构。不过需要注意，图示仅用于理解线程的逻辑组织，而硬件的实际执行和存储方式可能有所不同：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124224629.png)

这里 $(ix,iy)$ 就是整个线程模型中任意一个线程的索引，或者叫做全局地址，局部地址当然就是 $(threadIdx.x, threadIdx.y)$ 了，当然这个局部地址目前还没有什么用处，他只能索引线程块内的线程，不同线程块中有相同的局部索引值，比如同一个小区，A栋有16楼，B栋也有16楼，A栋和B栋就是 $blockIdx$ ，而16就是 $threadIdx$。

在 CUDA 中，每个线程的唯一索引（或称全局地址）可以通过 **块索引 (blockIdx)** 和 **线程索引 (threadIdx)** 计算得出。假设线程以二维方式组织，则横纵坐标计算公式如下：

$$
\begin{align*}
ix&=threadldx.x+blockIdx.x \times blockDim.x \\
iy&=threadldx.y+blockIdx.y \times blockDim.y 
\end{align*}
$$

这样，每个线程都拥有唯一的 **(ix, iy)** 坐标，并可在核函数中访问该索引。由于 CUDA 采用 **单指令多线程 (SIMT, Single Instruction Multiple Thread)** 方式，即所有线程执行相同的代码，因此需要确保每个线程处理的数据互不重叠，否则会导致重复计算，影响结果正确性。为了让不同线程处理不同的数据，CUDA 常用的做法是让不同的线程对应不同的数据，也就是用线程的全局标号对应不同组的数据。

CUDA 设备内存（或主机内存）通常是线性存储的，即一个二维矩阵在内存中是按行存储的，例如 $(8 \times 6)$ 矩阵的存储方式如下：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250124225132.png)

我们要做管理的就是：

-   线程和块索引（来计算线程的全局索引）
-   矩阵中给定点的坐标 $(ix,iy)$
-   $(ix,iy)$ 对应的线性内存的位置

为了正确映射线程索引到矩阵中的位置，需要计算 **线性内存地址 (idx)**。对于 **(ix, iy)** 处的矩阵元素，其对应的一维索引可通过以下公式计算：

$$
idx=ix+iy*nx
$$

其中 `nx` 是矩阵的列数（即每行包含的元素个数）。

通过这种方式，每个线程的全局坐标 **(ix, iy)** 可唯一对应矩阵中的一个数据点，实现数据的均匀分配。例如，索引 **(10,10)** 的线程处理矩阵中 **(10,10)** 位置的数据。虽然可以采用其他映射方式，但这种直接对应的策略最为直观，且最不容易出错。 

接下来的代码来输出每个线程的标号信息 `chapter02/checkThreadIndex.cu` ：

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

这段代码输出了一组我们随机生成的矩阵，并且核函数打印自己的线程标号，注意，核函数能调用 printf，这个特性是CUDA后来加的，最早的版本里面不能 printf。又由于 CUDA 采用 **SIMT（Single Instruction Multiple Threads）** 模型，每个线程块的执行顺序由 GPU 的调度器决定，并**不保证所有线程块按照某种固定顺序启动**。所以**输出的顺序是不按先后顺序执行的**。输出结果：

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

也可以使用浮点数进行矩阵加法 `chapter02/checkThreadIndexFloat.cu` :

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

由之前介绍的 CUDA 索引计算，可以使用以下代码打印每个线程的标号信息。可以得出二维矩阵加法核函数：

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

下面调整不同的线程组织形式，测试一下不同的效率并保证得到正确的结果，但是什么时候得到最好的效率是后面才要考虑的。我们要做的就是用各种不同的相乘组织形式得到正确结果，代码在 `chapter02/sumMatrix2D.cu` 中。

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
    int row    = 1 << 14;    // 2^12, 4096
    int col    = 1 << 14;    // 2^12, 4096
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

观察结果没有多大差距，但是明显比 CPU 快了很多，而且最主要的是用不同的线程组织模式都得到了正确结果，并且：

- 改变执行配置（线程组织）能得到不同的性能
- 传统的核函数可能不能得到最好的效果
- 一个给定的核函数，通过调整网格和线程块大小可以得到更好的效果

### 5. 设备管理 Managing Devices

在 CUDA 开发中，设备管理至关重要，尤其是在通用软件或框架开发中，确保程序能够适配不同 GPU 硬件环境，以避免因设备差异导致的崩溃。一般情况下，设备管理可分为两类：

1.  **通用程序或框架的动态检测**  
    对于需要在不同硬件环境下运行的程序，应在 CUDA 初始化前查询 GPU 设备信息，以确保兼容性。例如，可使用 `cudaGetDeviceCount()` 获取可用设备数量，并遍历查询计算能力、全局内存、共享内存等关键参数，从而动态调整计算策略。
2.  **固定环境下的静态查询**  
    对于运行于特定服务器或工作站的应用，通常不需要程序级别的设备检测，而是直接使用 NVIDIA 驱动提供的命令行工具，例如 `nvidia-smi`，快速查看 GPU 规格、显存占用和运行状态，以便进行优化和监控。
    

合理选择设备管理方式不仅能提高程序的稳定性，还能最大化利用硬件资源，提升 CUDA 应用的可移植性和执行效率。

#### 5.1. API查询GPU信息 Using the Runtime API to Query GPU Information

使用代码 `chapter02/checkDeviceInfor.cu` 可以在软件内查询信息：

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

主要用到了下面 API。了解 API 的功能最好不要看博客，因为博客不会与时俱进，要查文档，所以对于API的不了解，

**建议查阅官方文档，而非博客，以确保信息的准确性和时效性！**

**建议查阅官方文档，而非博客，以确保信息的准确性和时效性！**

**建议查阅官方文档，而非博客，以确保信息的准确性和时效性！**

常用的 API 包括：

-   `cudaSetDevice`：设置当前使用的 GPU 设备。
-   `cudaGetDeviceProperties`：获取 GPU 设备的详细信息，如计算能力、内存大小等。
-   `cudaDriverGetVersion`：查询当前驱动程序版本。
-   `cudaRuntimeGetVersion`：获取 CUDA 运行时版本。
-   `cudaGetDeviceCount`：获取可用 GPU 设备的数量。

这些 API 提供的设备信息直接影响 CUDA 程序的优化策略，例如：

1.  **CUDA 驱动与运行时版本**：决定了可用的 CUDA 语言特性和 API 兼容性。
2.  **计算能力（Compute Capability）**：影响硬件支持的指令集、并行计算能力和线程调度机制。
3.  **全局内存大小**：决定可加载的数据规模，影响数据预取与缓存策略。
4.  **GPU 主频 & 带宽**：决定计算吞吐量和数据传输效率。
5.  **L2 缓存大小**：影响数据复用效率，优化访存性能。
6.  **纹理维度与层叠纹理**：影响图像处理与计算的存储布局。
7.  **常量内存与共享内存**：决定数据共享策略，优化线程间通信。
8.  **线程束（warp）大小**：影响 SIMT（Single Instruction Multiple Thread）执行效率。
9.  **处理单元的最大线程数**：决定任务分配和并行度优化。
10.  **块、网格尺寸及最大线性内存**：影响线程组织方式和计算资源利用率。

上面这些都是后面要用到的关键参数，这些会严重影响效率。程序运行前，应动态查询设备信息，确保配置合理，以实现最优的计算性能。后续章节将详细探讨不同参数对程序性能的影响，并介绍优化策略。

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

1. **运行时获取设备信息（动态优化）**：适用于通用软件或需要在不同 GPU 上适配的程序，具体流程如下
	- 编译程序
	- 启动程序
	- 运行时查询 GPU 设备信息，并存入全局变量
	- 各功能模块根据查询到的设备信息动态调整计算参数
	-  程序执行完毕
	这种方法适用于需要在**多种设备**上运行的程序，但查询设备信息会带来一定的运行时开销


2. **编译时获取设备信息（静态优化）**：适用于部署在**特定硬件环境**下的高性能计算程序，流程如下
	-   运行脚本预先查询 GPU 设备信息
	-   编译阶段根据查询结果调整优化参数，并固化到二进制代码中
	-   运行时直接使用预优化参数，无需额外查询
	-   程序运行完毕
	这种方法能够**最大化程序性能**，避免运行时查询的额外开销，但不适用于硬件环境经常变动的情况

指令 nvidia-smi -q -i 0 可以提取以下我们要的信息：

-   **MEMORY**（显存大小、占用情况）
-   **UTILIZATION**（GPU 使用率）
-   **ECC**（错误校正状态）
-   **TEMPERATURE**（核心温度）
-   **POWER**（功耗状态）
-   **CLOCK**（当前频率）
-   **COMPUTE**（计算能力）
-   **PIDS**（进程信息）
-   **PERFORMANCE**（性能模式）
-   **SUPPORTED_CLOCKS**（支持的频率范围）
-   **PAGE_RETIREMENT**（内存页故障情况）
-   **ACCOUNTING**（计算任务日志）

### 6. 总结

至此，CUDA 编程的核心概念已涵盖：核函数、计时、内存管理、线程组织、设备参数等。掌握这些内容，已经能够编写出比 CPU 快得多的 CUDA 程序。下一篇将深入研究 GPU 硬件架构背后的性能优化策略。

---

## 参考引用

### 书籍出处

- [CUDA C编程权威指南](asset/CUDA%20&%20GPU%20Programming/CUDA%20C编程权威指南.pdf)
- [Professional CUDA C Programming](asset/CUDA%20&%20GPU%20Programming/Professional%20CUDA%20C%20Programming.pdf)

### 网页链接

- [人工智能编程 | 谭升的博客](https://face2ai.com/program-blog/)
- [CUDA学习入门（三） CUDA线程索引 & 如何设置Gridsize和Blocksize](https://blog.csdn.net/weixin_44222088/article/details/135732160)
- [CUDA线程模型与全局索引计算方式](https://zhuanlan.zhihu.com/p/666077650)