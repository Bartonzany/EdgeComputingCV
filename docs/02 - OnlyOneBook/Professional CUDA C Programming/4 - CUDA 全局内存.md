## 4 - CUDA 全局内存 CUDA Global Memory

---

### 0. 前言

在之前的文章中，我们已经介绍了线程如何在GPU上执行，以及如何通过操作线程束来优化核函数性能。然而，核函数性能并不仅仅取决于线程束的执行。在执行模型中，核函数的配置也对程序执行效率起着决定性作用。除了线程束和线程块等执行结构外，内存的性能也对程序效率有着重要影响。

举个例子，考虑一个工厂的生产流程，我们可以通过优化生产线、分配工人和提高工人素质来提高生产速度。但是如果把这个工厂建在珠穆朗玛峰的山顶，且一年只能有一辆原料运输车到达（对应显存带宽受限），因为工人和生产线都在等待原料运输，故而导致整个工厂的生产效率就会大大降低，这就是 GPU 内存系统影响系统效率的典型例子。因此，**内存带宽和访问速度也是影响程序吞吐量的重要因素之一**。

本篇我们将分析核函数与全局内存的联系和性能影响。理解内存子系统的工作原理，掌握全局内存的空间局部性优化技巧，是解锁 GPU 真实算力的重要方法。

### 1. CUDA内存模型概述 INTRODUCING THE CUDA MEMORY MODEL

#### 1.1. 内存层次结构的优点 Benefi ts of a Memory Hierarchy

程序具有局部性特点，包括**时间局部性**和**空间局部性**：

- **时间局部性**：如果一个内存位置的数据在某个时刻被访问，那么在接下来的一段时间内，该数据很可能会被再次访问。随着时间流逝，该数据被再次访问的可能性逐渐降低。
- **空间局部性**：如果某个内存位置的数据被访问，附近的数据也有很高的可能性会被访问到。

现代计算机的内存结构通常包括以下部分：

![CUDA Memory Architecture](/images/Professional%20CUDA%20C%20Programming/CUDA%20Memory%20Architecture.png)

这个内存模型包括**寄存器**、**缓存**、**主存**和**大容量磁盘**，也是大部分冯诺依曼计算机结构的存储结构。已经学习过串行编程的人应该对内存模型有所了解，最快的存储器是寄存器，它可以与 CPU 同步协作；其次是缓存，位于 CPU 芯片上。然后是主存储器，通常是内存条，显卡上也有内存芯片（即显存），最慢的是硬盘。这些内存设备的速度和容量呈反比关系，**速度越快，容量越小；速度越慢，容量越大**。

对于最后一层大容量设备（如硬盘、磁带等），其特点包括：

-   **每比特位的价格更低**
-   **更高的容量**
-   **更低的延迟**
-   **更少的处理器访问频率**

CPU 和 GPU 的主存通常使用的是 **DRAM（动态随机存取存储器**），而低延迟内存（如 CPU 一级缓存）则使用的是 **SRAM（静态随机存取存储器）**。尽管低层次的大容量存储器延迟高、容量大，但当其中某些数据频繁使用时，这些数据会向更高层次的存储器传输。例如，在程序处理数据时，首先**将数据从硬盘传输到主存中**。

GPU 和 CPU 在内存层次结构设计中遵循相似的准则和模型。其中一个主要区别是，CUDA编程模型更好地展现了内存层次结构，并提供了显式的控制能力，使我们能够更好地管理其行为。

#### 1.2. CUDA 内存模型 CUDA Memory Model

对于程序员而言，关于内存分类的方法有很多种，但最常见的分法可以分为以下两种：

1.  **可编程内存**：可以通过编写代码来控制其行为，需要手动控制数据存放在可编程内存中
2.  **不可编程内存**：用户无法控制其行为，不能决定数据的存放位置，其行为在出厂后就已经固化。用户只能了解其原理，尽可能利用其规则来提高程序性能

在 CPU 内存结构中，一级和二级缓存通常被视为不可编程（无法直接控制）的存储设备。与 CPU 相比，在 CUDA 内存模型中，GPU 上的内存设备更加丰富，包括：

-   寄存器
-   共享内存
-   本地内存
-   常量内存
-   纹理内存
-   全局内存

在CUDA中，每种内存都有其特定的作用域、生命周期和缓存行为。具体而言：

-   **每个线程都拥有私有的本地内存**
-   **线程块拥有共享内存，对线程块内所有线程可见**
-   所有线程都可以访问**只读**的**常量内存**和**纹理内存**进行读取操作
-   全局内存、常量内存和纹理内存有各自不同的用途

在一个应用程序中，全局内存、常量内存和纹理内存具有相同的生命周期。下图为这些内存空间的层次结构，后面的内容将逐一介绍这些内存的特性和用法。

![CUDA Memory Architecture Model](/images/Professional%20CUDA%20C%20Programming/CUDA%20Memory%20Architecture%20Model.png)

##### 1.2.1. 寄存器 Registers

**寄存器是 CPU 和 GPU 中速度的内存空间**，不同之处在于 **GPU 的寄存器数量通常更多一些**。在核函数内，**如果声明一个变量而不加修饰符**，该变量将存储在寄存器中。而在 CPU 程序中，只有**当前需要计算的变量存储在寄存器中**，其余变量存储在主存中，在需要时传输至寄存器。同样，**在核函数中定义长度固定的数组也会在寄存器中分配地址**。

**每个线程拥有私有的寄存器**，寄存器通常保存频繁使用的私有变量。需要注意的是，这些变量必须是线程私有的，否则变量之间将不可见，可能导致多个线程同时修改一个变量而互相不知情。**寄存器变量的生命周期与核函数一致，从开始运行到结束，执行完毕后，寄存器便无法再访问**。

**寄存器是 SM 中一种稀缺资源**，例如 Fermi 架构中每个线程最多只能拥有 63 个寄存器。Kepler 架构将这一数量扩展到 255 个寄存器。如果一个线程使用较少的寄存器，那么将允许更多的线程块驻留，从而提高并发性能，效率也会更高。

我们可以使用以下代码查看当前服务器下显卡中**寄存器的数量**、**共享内存的字节数**以及**每个线程所使用的常量内存的字节数**（`chapter04/checkRegisters.cu`）

```c
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

int main(int argc, char* argv[]) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Constant memory per thread: %zu bytes\n", prop.totalConstMem / prop.multiProcessorCount);
        printf("\n");
    }

    return 0;
}
```

输出如下：

```shell
Device 0: NVIDIA GeForce GTX 1060 6GB
Registers per block: 65536
Shared memory per block: 49152 bytes
Constant memory per thread: 6553 bytes
```

当核函数使用超过硬件限制数量的寄存器时，多余的寄存器将被本地内存取代。这种寄存器溢出将对性能产生不利影响，因此应尽量避免发生。为避免寄存器溢出，可以在核函数代码中使用额外的信息以协助编译器进行优化：

```c
__global__ void  
__lauch_bounds__(maxThreadaPerBlock,minBlocksPerMultiprocessor)  
kernel(...) {  
    /* kernel code */  
}
```

在核函数定义前添加了一个关键字 `launch_bounds`，后面跟着两个变量：

-   `maxThreadsPerBlock`：线程块内包含的最大线程数，线程块由核函数启动。
-   `minBlocksPerMultiprocessor`：可选参数，每个 SM 中预期的最小常驻线程块数。

> **注意**：针对特定的核函数，启动边界的优化可能因不同的架构而异

此外，可以在**编译选项**中添加 `-maxrregcount=32` 来控制一个编译单元中所有核函数使用的最大寄存器数量。

##### 1.2.2 本地内存 Local Memory

在核函数中，**符合存储在寄存器中但由于种种原因无法被分配到寄存器空间的变量**，会被存储在本地内存中。编译器可能会将以下类型的变量放置在本地内存中：

-   使用未知索引引用的本地数组
-   **较大的本地数组或结构体**，可能会占用大量寄存器空间
-   任何不符合核函数寄存器限制条件的变量

**本地内存实际上与全局内存位于同一块存储区域**，其访问特点是**高延迟、低带宽**。

对于2.0及更高版本的设备，本地内存存储在每个SM的一级缓存或设备的二级缓存中。

##### 1.2.3 共享内存 Shared Memory

在核函数中，使用 `__shared__` 修饰符声明的内存被称为共享内存。每个 SM 拥有一定数量由线程块分配的共享内存，共享内存位于片上内存，速度要比主存快得多，具有**低延迟**和**高带宽**。类似于一级缓存，但可以被编程控制。

在使用共享内存时，**不要因为过度使用导致降低 SM 上活跃的线程束数量**，即一个线程块使用过多共享内存会导致无法启动更多线程块，从而影响活跃线程束的数量。

**共享内存在核函数内进行声明，其生命周期与线程块一致**。当线程块开始运行时，共享内存被分配；当线程块结束时，共享内存被释放。

**共享内存是线程间相互通信的基本方式**。同一块内的线程可以通过共享内存中的数据相互合作。由于共享内存是块内线程可见的，会引发竞争问题，这时可以通过共享内存进行通信。为了避免内存竞争，可以使用同步语句：

```c
void __syncthreads();
```

该语句在线程块执行时充当一个同步点，**所有线程必须到达该点才能进行下一步计算**，在[3 - CUDA 执行模型](docs/02%20-%20OnlyOneBook/Professional%20CUDA%20C%20Programming/3%20-%20CUDA%20执行模型.md)中有相应的代码案例。通过合理设计可以避免共享内存中的内存竞争。需要注意的是，过度频繁地使用 `__syncthreads();` 会影响内核的执行效率。

在 SM 中，**一级缓存和共享内存共享同一片 64 KB的片上内存**（具体容量可能会因设备不同而异），它们之间通过静态划分占用各自的空间。在运行时，可以通过以下语句进行设置：

```c
cudaError_t cudaFuncSetCacheConfig(const void * func,enum cudaFuncCache);
```

这个函数用于设置内核中共享内存和一级缓存的比例。`cudaFuncCache` 参数可以选择以下配置：

-   `cudaFuncCachePreferNone`：无特定偏好，默认设置
-   `cudaFuncCachePreferShared`：48KB 共享内存，16KB 一级缓存
-   `cudaFuncCachePreferL1`：48KB 一级缓存，16KB 共享内存
-   `cudaFuncCachePreferEqual`：32KB 一级缓存，32KB 共享内存

Fermi 架构支持前三种配置，而后续架构的设备都支持这四种配置。

##### 1.2.4. 常量内存 Constant Memory

常量内存位于设备内存中，每个 SM 都有专用的常量内存缓存。常量变量需要使用 `__constant__` 修饰符进行声明。

**常量变量必须在全局空间内和所有核函数之外声明**。在每个设备上，只能分配 64KB 的常量内存。常量内存的分配是静态的，并且对同一编译单元中的所有核函数可见。

**常量内存在设备端核函数代码中不能被修改**，主机端代码可以初始化常量内存。初始化函数如下：

```C
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void *src, size_t count);
```

这个函数与 `cudaMemcpy` 参数列表类似，从源地址 `src` 复制 `count` 个字节的内存到常量内存 `symbol` 中。大多数情况下，此函数是同步执行的，会立即生效。

在线程束内所有线程都从相同的地址取数据时，常量内存表现较好。例如，在执行多项式计算时，如果系数存储在常量内存中，效率会非常高。但如果不同的线程需要访问不同的地址，常量内存的效率就不那么好了。这是因为常量内存的读取机制是**一次读取会被广播给线程束内的所有线程**。

##### 1.2.5. 纹理内存 Texture Memory

纹理内存位于设备内存中，并且缓存在每个 SM 的**只读缓存**中。纹理内存是通过指定的缓存进行全局内存访问的，只读缓存支持硬件滤波，可以在读取过程中执行浮点插值。纹理内存是针对二维空间局部性进行优化的。

总的来说，纹理内存的设计目的是为了**优化 GPU 的本职工作**，但对于某些特定类型的程序可能效果更好，例如需要滤波的程序，可以直接利用硬件完成滤波操作。

##### 1.2.6. 全局内存 Global Memory

**全局内存在 GPU 上拥有最大内存空间**，**延迟最高并且最常使用的内存**。全局内存在作用域和生命周期上定义，**通常在主机端代码中声明**，但也可以在设备端定义，需要添加修饰符。只要不销毁，全局内存与应用程序具有相同的生命周期。全局内存在逻辑上对应于设备内存，是一种逻辑表示和硬件表示的对应关系。

全局内存可以动态声明或静态声明，在设备代码中可以使用以下修饰符静态声明变量：

```c
__device__
```

在之前的章节中，即到目前为止我们还未对内存进行任何优化，所有在 GPU 上访问的内存都属于全局内存。由于全局内存的特性，在多个核函数同时执行时，如果使用同一全局变量，需要注意内存竞争的问题。

全局内存访问是按字节对齐的，即一次读取指定大小的整数倍字节的内存（通常为32、64、128字节）。因此，在线程束执行内存加载/存储时，传输数量取决于以下两个因素：

-   **跨线程的内存地址分布**
-   **内存事务的对齐方式**

通常情况下，**满足最大内存请求的任务越多，未使用的字节被传输的可能性就越高，数据吞吐量就会降低**。换句话说，对齐的读写模式会导致不必要的数据传输，低利用率会降低吞吐量。在 1.1 版本以下的设备中，对内存访问的要求非常严格，因为当时还没有缓存。现在的设备都已经包含缓存，因此要求稍微宽松了一些。

接下来，我们将讨论如何优化全局内存访问，以最大程度地提高全局内存的数据吞吐量。

##### 1.2.7 GPU 缓存 GPU Caches

与 CPU 缓存类似，GPU 缓存是不可编程的内存，通常分为以下四种：

1.  一级缓存
2.  二级缓存
3.  只读常量缓存
4.  只读纹理缓存

**每个 SM 都拥有一个一级缓存，而所有 SM 共享一个二级缓存**。一级和二级缓存的作用是存储本地内存和全局内存中的数据，同时也包括寄存器溢出的部分。在Fermi、Kepler 以及后续的设备中，CUDA 允许我们配置读取操作是使用一级缓存和二级缓存，还是仅使用二级缓存。

与 CPU 不同的是，**在 GPU 中写操作不会被缓存，只有读取操作会受到缓存的影响**。每个 SM 还有一个专门用于提高设备内存中数据读取性能的**只读常量缓存**和**只读纹理缓存**。

##### 1.2.8. CUDA变量声明总结 CUDA Variable Declaration Summary

以下是用表格对各内存类型的总结：

| 修饰符       | 变量名称       | 存储器 | 作用域 | 生命周期 |
| ------------ | -------------- |:------:|:------:|:--------:|
|              | float var      | 寄存器 |  线程  |   线程   |
|              | float var[100] |  本地  |  线程  |   线程   |
| __share__    | float var*     |  共享  |   块   |    块    |
| __device__   | float var*     |  全局  |  全局  | 应用程序 |
| __constant__ | float var*     |  常量  |  全局  | 应用程序 |

设备存储器的重要特征：

| 存储器 | 片上/片外 |   缓存    | 存取 |     范围      | 生命周期 |
|:------:|:---------:|:---------:|:----:|:-------------:|:--------:|
| 寄存器 |   片上    |    n/a    | R/W  |   一个线程    |   线程   |
|  本地  |   片外    | 1.0以上有 | R/W  |   一个线程    |   线程   |
|  共享  |   片上    |    n/a    | R/W  | 块内所有线程  |    块    |
|  全局  |   片外    | 1.0以上有 | R/W  | 所有线程+主机 | 主机配置 |
|  常量  |   片外    |    Yes    |  R   | 所有线程+主机 | 主机配置 |
|  纹理  |   片外    |    Yes    |  R   | 所有线程+主机 | 主机配置 |

##### 1.2.9 静态全局内存 Static Global Memory

在计算机中，内存分配通常分为**静态分配**和**动态分配**两种类型。静态分配是指**在编译时或运行时根据固定的程序布局分配内存，通常发生在栈上**。动态分配则是**在运行时根据需要动态地分配内存空间，通常发生在堆上**，需要使用诸如`new`、 `malloc`等函数来动态分配内存，并使用`delete`、 `free`等函数来释放内存。

在 CUDA 编程中，同样存在静态分配和动态分配的概念。通常我们使用`cudaMalloc`函数来动态分配设备端内存。如果要进行静态分配，则需要将数据显式地从主机端拷贝到设备端。接下来我们将展示一个示例代码来说明程序的运行结果`chapter04/globalVariable.cu`：

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData;

__global__ void checkGlobalVariable() {
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData);

    // alter the value
    devData += 2.0f;
}

int main(void) {
    // initialize the global variable
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1, 1>>>();

    // copy the global variable back to the host
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
```

运行结果：

```shell
Host:   copied 3.140000 to the global variable
Device: the value of the global variable is 3.140000
Host:   the value changed by the kernel to 5.140000
```

需要注意的是，在代码:

```c
cudaMemcpyToSymbol(&devData, &value, sizeof(float));
```

函数`cudaMemcpyToSymbol`的第一个参数应该是`void*`类型，但在此处却使用了`__device__ float devData;`这种声明。实际上，设备上的变量定义与主机端变量定义不同。设备变量在代码中被声明时实际上是一个指针，指向的具体位置主机端是无法知晓的，也不知道指向的内容。要获取指向的内容，唯一的方法是通过显式的传输:

```c
cudaMemcpyFromSymbol(&value, devData, sizeof(float));
```

需要特别注意以下几点：

-   在主机端， `devData`只是一个标识符，而不是设备全局内存的变量地址。
-   在核函数中， `devData`是一个全局内存中的变量。
-   主机代码不能直接访问设备变量，设备也不能访问主机变量，这是 CUDA 编程与 CPU 编程最大的不同点。

在主机端，不可以对设备变量取地址运算，即`cudaMemcpy(&value, devData, sizeof(float));` 是无效的操作！你不能使用动态拷贝方式将值传给静态变量。 如果你一定要使用 `cudaMemcpy` 获取 `devData` 的地址，只能采用以下方式:

```c
float *dptr = NULL;
cudaGetSymbolAddress((void**)&dptr, devData);
cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice);
```

还有一种例外，可以直接从主机端引用 GPU 内存，即 CUDA 固定内存。后续我们将深入研究这方面内容。

CUDA 运行时 API 能够访问主机和设备端变量，但这取决于是否向正确的函数提供了正确的参数。当使用运行时 API 时，如果参数填写错误，尤其是主机和设备端的指针，结果是无法预测的。

### 2. 内存管理  MEMORY MANAGEMENT

CUDA 编程与 C 语言类似，需要程序员自行管理主机和设备之间的数据移动。 CUDA 编程的主要目的是为了加速程序运行，特别是在**机器学习**和**人工智能**等计算领域，这些任务在 CPU 上无法高效完成。控制硬件的语言属于底层语言，比如 C 语言，其中最为棘手的问题之一就是内存管理。相比之下，像 Python 和 PHP 这类语言具有自己的内存管理机制，而 C 语言的内存管理则需要程序员来负责。虽然这种方式学习起来比较困难，但一旦掌握会感到非常满足，因为拥有了自由控制计算机计算过程的能力。

**CUDA C 是 C 语言的扩展，其内存管理方面基本继承了C语言的方式**，要由程序员来控制 CUDA 内存。不同于 CPU 内存分配，GPU内存涉及到数据传输，**需要主机和设备之间进行数据传输**。

下一步我们要了解的内容包括：

1.  **设备内存的分配和释放**
2.  **在主机和设备之间传输内存数据**

为了实现最佳性能，CUDA 提供了在主机端准备设备内存的函数，同时要求手动向设备传递数据，并且手动从设备获取数据。

#### 2.1 内存分配和释放 Memory Allocation and Deallocation

在主机上进行全局内存分配使用以下函数：

```C
cudaError_t cudaMalloc(void ** devPtr,size_t count)
```

该函数在设备上分配了 `count` 字节的全局内存，并通过 `devptr` 指针返回该内存的地址。需要注意的是，第一个参数是**指针的指针**。通常的做法是首先声明一个指针变量，然后调用这个函数：

```C
float *devMem = nullptr;
cudaError_t error = cudaMalloc((void**)&devMem, count);
```

`devMem` 是一个指针变量，初始化为 `nullptr`，可以避免出现野指针。`cudaMalloc` 函数需要修改 `devMem` 的值，因此必须将其指针传递给函数。如果将 `devMem` 直接作为参数传递，经过函数后，指针的内容仍然是 `nullptr`。

内存分配函数所分配的内存支持任何变量类型，包括**整型**、**浮点型**、**布尔型等**。如果 `cudaMalloc` 函数执行失败，则会返回 `cudaErrorMemoryAllocation`。

当分配完地址后，可以使用下面函数进行初始化：

```C
cudaError_t cudaMemset(void * devPtr,int value,size_t count)
```

这个函数的用法类似于 `memset`，但需要注意的是，我们操作的内存是在 GPU 上对应的物理内存。

当已分配的内存不再被使用后，使用下面语句释放内存空间：

```C
cudaError_t cudaFree(void * devPtr)
```

值得注意的是，参数 `devPtr` 必须是之前使用 `cudaMalloc` 或其他类似函数分配的内存空间。如果传递了非法指针参数，会返回 `cudaErrorInvalidDevicePointer` 错误。另外，重复释放同一块内存空间也将导致错误。此外，如果尝试释放一个已经被释放的地址空间，则 `cudaFree` 也会返回错误。

#### 2.2. 内存传输 Memory Transfer

一旦分配好内存，我们就可以使用下列函数从主机向设备传输数据。C 语言的内存分配完成后可以直接读写，但在异构计算中，主机线程无法访问设备内存，设备线程也不能访问主机内存。因此，我们需要通过传输数据来实现数据在主机和设备之间的交互。

```C
cudaError_t cudaMemcpy(void *dst,const void * src,size_t count,enum cudaMemcpyKind kind)
```

这里需要注意的是，这些参数都是指针，而不是指针的指针。第一个参数`dst`表示目标地址，第二个参数`src`表示原始地址，然后是要拷贝的内存大小，最后是传输类型。传输类型包括以下几种：

-   cudaMemcpyHostToHost
-   cudaMemcpyHostToDevice
-   cudaMemcpyDeviceToHost
-   cudaMemcpyDeviceToDevice

在前面几篇中，已经详细描述了主机端和设备端代码传输的例子，这里不再赘述。下图是CPU 内存和 GPU 内存间的连接性能对比：

![Comparison between CPU transfer and GPU transfer](/images/Professional%20CUDA%20C%20Programming/Comparison%20between%20CPU%20transfer%20and%20GPU%20transfer.png)

GPU 采用 DDR5 内存，而 2011 年三星推出了 DDR4 主机内存，但 GPU 一直在使用DDR5。GPU 的内存具有非常高的理论峰值带宽，例如 Fermi C2050 的峰值带宽为 144GB/s，现在的 GPU 会更高。然而，**CPU 和 GPU 之间的通信通过 PCIe 总线，其理论带宽要低得多**，约为 8GB/s。如果对数据传输管理不当，应用程序的性能会受到影响，**数据传输效率会受限于 PCIe 总线的带宽**。因此，在 CUDA 编程中，一个基本原则就是要**尽量减少主机与设备之间的数据传输**。

#### 2.3 固定内存 Pinned Memory

GPU 无法安全地访问可分页主机内存，因为操作系统可能会移动数据的实际位置。主机物理内存被操作系统分成许多分页，**类似于把一本书分割成多页，但每次读取的时候不能保证每页的排列是连续的**。当数据需要从主机传输到设备时，如果数据的物理位置发生变化，传输操作就会受到影响。为了避免这种情况，**CUDA 驱动程序会提前锁定页面或分配固定的主机内存，将数据复制到固定内存，然后再传输数据到设备上**。这就好比你在准备做一道菜，需要将食材准备好，如果食材突然被搬动到厨房的另一个位置，你就无法顺利完成烹饪。CUDA 驱动就像在烹饪之前先把食材准备好**固定**在桌上，确保烹饪过程不受干扰。下面则是两种方式的示意图：


![Page Data Transfer and Pinned Data Transfer](/images/Professional%20CUDA%20C%20Programming/Page%20Data%20Transfer%20and%20Pinned%20Data%20Transfer.png)

- 左边是正常分配内存，传输过程是：**锁定内存页 -> 复制数据到固定内存 -> 将数据复制到设备上**
- 右边时分配时就是固定内存，直接传输到设备上

分配固定内存的函数如下：

```C
cudaError_t cudaMallocHost(void ** devPtr,size_t count)
```

该函数用于分配固定内存，大小为count字节。固定内存被锁定在内存页中，可以直接传输到设备上。由于设备可以直接访问固定内存，因此可以获得比可分页内存更高的读写带宽。

固定主机内存释放：

```C
cudaError_t cudaFreeHost(void *ptr)
```

现在我们用代码实际测试一下固定内存和分页内存的传输效率（`chapter04/pinMemTransfer.cu`）。此代码改写至前几篇中的 `sumArrays.cu`，不同之处在于将 `cudaMalloc` 和 `cudaFree` 改写为 `cudaMallocHost` 和 `cudaFreeHost`：

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
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    res[i] = a[i] + b[i];
}
int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);

    int nElem = 1 << 14;
    printf("Vector size:%d\n", nElem);
    int    nByte          = sizeof(float) * nElem;
    float* a_h            = (float*)malloc(nByte);
    float* b_h            = (float*)malloc(nByte);
    float* res_h          = (float*)malloc(nByte);
    float* res_from_gpu_h = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_d, *b_d, *res_d;
    CHECK(cudaMallocHost((float**)&a_d, nByte));
    CHECK(cudaMallocHost((float**)&b_d, nByte));
    CHECK(cudaMallocHost((float**)&res_d, nByte));

    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid(nElem / block.x);
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d);
    printf("Execution configuration %d,%d\n", block.x, grid.x);

    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    sumArrays(a_h, b_h, res_h, nElem);

    checkResult(res_h, res_from_gpu_h, nElem);
    cudaFreeHost(a_d);
    cudaFreeHost(b_d);
    cudaFreeHost(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return 0;
}
```

使用指令查看常规内存分配结果：

```shell
nvprof ./bin/chapter02/sumArrays
```

```shell
==6976== NVPROF is profiling process 6976, command: ./sumArrays
Vector size:16384
Execution configuration<<<1024,16>>>
Check result success!
==6976== Profiling application: ./sumArrays
==6976== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.52%  12.864us         2  6.4320us  6.4320us  6.4320us  [CUDA memcpy HtoD]
                   28.92%  5.9510us         1  5.9510us  5.9510us  5.9510us  [CUDA memcpy DtoH]
                    8.55%  1.7600us         1  1.7600us  1.7600us  1.7600us  sumArraysGPU(float*, float*, float*)
      API calls:   99.30%  82.289ms         1  82.289ms  82.289ms  82.289ms  cudaSetDevice
                    0.22%  181.31us       114  1.5900us      80ns  76.250us  cuDeviceGetAttribute
                    0.16%  130.84us         1  130.84us  130.84us  130.84us  cudaLaunchKernel
                    0.11%  94.199us         3  31.399us  2.1900us  87.619us  cudaMalloc
                    0.11%  90.989us         3  30.329us  2.8200us  82.009us  cudaFree
                    0.08%  65.580us         3  21.860us  14.660us  26.320us  cudaMemcpy
                    0.01%  9.9400us         1  9.9400us  9.9400us  9.9400us  cuDeviceGetName
                    0.01%  8.3800us         1  8.3800us  8.3800us  8.3800us  cuDeviceGetPCIBusId
                    0.00%     690ns         3     230ns     110ns     400ns  cuDeviceGetCount
                    0.00%     330ns         2     165ns      90ns     240ns  cuDeviceGet
                    0.00%     230ns         1     230ns     230ns     230ns  cuDeviceTotalMem
                    0.00%     160ns         1     160ns     160ns     160ns  cuModuleGetLoadingMode
                    0.00%     140ns         1     140ns     140ns     140ns  cuDeviceGetUuid
```

修改后的固定内存分配结果：

```shell
nvprof ./bin/chapter04/pinMemTransfer
```

```shell
==7685== NVPROF is profiling process 7685, command: ./pinMemTransfer
Vector size:16384
Execution configuration<<<1024,16>>>
Check result success!
==7685== Profiling application: ./pinMemTransfer
==7685== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.31%  31.170us         3  10.390us  4.9400us  15.700us  [CUDA memcpy HtoH]
                   32.69%  15.136us         1  15.136us  15.136us  15.136us  sumArraysGPU(float*, float*, float*)
      API calls:   97.01%  87.711ms         1  87.711ms  87.711ms  87.711ms  cudaSetDevice
                    1.78%  1.6074ms         3  535.79us  3.1800us  1.5990ms  cudaHostAlloc
                    0.77%  700.29us         3  233.43us  2.6000us  692.37us  cudaFreeHost
                    0.20%  184.41us       114  1.6170us      80ns  77.318us  cuDeviceGetAttribute
                    0.15%  138.30us         1  138.30us  138.30us  138.30us  cudaLaunchKernel
                    0.06%  51.459us         3  17.153us  6.2100us  27.309us  cudaMemcpy
                    0.01%  10.730us         1  10.730us  10.730us  10.730us  cuDeviceGetName
                    0.01%  6.7800us         1  6.7800us  6.7800us  6.7800us  cuDeviceGetPCIBusId
                    0.00%     780ns         3     260ns     120ns     530ns  cuDeviceGetCount
                    0.00%     270ns         2     135ns      80ns     190ns  cuDeviceGet
                    0.00%     210ns         1     210ns     210ns     210ns  cuDeviceTotalMem
                    0.00%     150ns         1     150ns     150ns     150ns  cuModuleGetLoadingMode
                    0.00%     130ns         1     130ns     130ns     130ns  cuDeviceGetUuid
```

固定内存的指标通常显示为HtoH，表示主机到主机的内存拷贝，而传统拷贝显示为HtoD。在使用 `pinMemTransfer` 时，由于采用固定内存，可以观察到其主要耗时在GPU活动上，尤其是主机到主机数据传输。相比之下，`sumArrays` 虽然总耗时较短，但在GPU活动方面的耗时比 `pinMemTransfer` 稍长。

综上所述： **固定内存的释放和分配成本通常比可分页内存高很多，但传输速度更快，因此对于大规模数据而言，固定内存的效率更高**。

#### 2.4 零拷贝内存 Zero-Copy Memory

通常情况下，主机不能直接访问设备内存，设备也不能直接访问主机内存。以前的设备一般遵守这个规则，但后来出现了一个例外——零拷贝内存。**零拷贝内存位于主机内存中，GPU线程可以直接访问**。通过使用 CUDA 零拷贝内存，可以实现在主机和GPU之间直接共享内存区域，从而避免不必要的数据拷贝。

CUDA 核函数使用零拷贝内存的情况包括：

1.  **当设备内存不足时**，可以利用主机内存
2.  **避免主机和设备之间的显式内存传输**
3.  提高 PCIe 传输速率

前面提到过要注意**线程间的内存竞争**，因为它们可能同时访问同一设备内存地址。现在，设备和主机也可以同时访问同一设备地址，因此在使用零拷贝内存时需要注意**主机和设备之间的内存竞争**。**零拷贝内存是固定内存，不可分页**，可以使用以下函数来创建零拷贝内存：

```c
cudaError_t cudaHostAlloc(void ** pHost,size_t count,unsigned int flags)
```

最后一个标志参数，可以选择以下值：

-   cudaHostAllocDefalt
-   cudaHostAllocPortable
-   cudaHostAllocWriteCombined
-   cudaHostAllocMapped

`cudaHostAllocDefault`和`cudaMallocHost`函数是一样的，`cudaHostAllocPortable`函数返回可被所有CUDA上下文使用的固定内存，`cudaHostAllocWriteCombined`返回写结合内存，在一些设备上这种内存传输效率更高。`cudaHostAllocMapped`可以创建零拷贝内存。

尽管零拷贝内存无需显式传输到设备，但设备不能直接通过`pHost`访问相应的内存地址。设备需要获得另一个地址，以便帮助设备访问主机上的零拷贝内存。具体方法是使用以下函数：

```C
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned flags);
```

`pDevice` 即为设备上访问主机零拷贝内存的指针。 在这里，`flags`必须设置为0，具体内容将在后面介绍。

**零拷贝内存可以被视为一个比设备主存储器更慢的设备**。 频繁的读写会显著降低零拷贝内存的效率。这一点很容易理解，因为每次都需要经过 PCIe 总线，速度肯定会变慢。就像千军万马要通过独木桥一样，速度必然受阻。如果还有不少人来回穿梭，情况就更糟了。但**在需要频繁传输大量数据的场景下，使用零拷贝可以提高数据传输效率**。

接下来，我们基于之前的代码进行数组相加操作，并观察效果。以下是主函数代码，核函数是上一节的代码（`chapter04/sumArrayZeroCopy.cu`）：

```C
int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);

    int power = 10;
    if (argc >= 2)
        power = atoi(argv[1]);
    int nElem = 1 << power;
    printf("Vector size:%d\n", nElem);
    int    nByte          = sizeof(float) * nElem;
    float* res_from_gpu_h = (float*)malloc(nByte);
    float* res_h          = (float*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_from_gpu_h, 0, nByte);

    float *a_host, *b_host, *res_d;
    double iStart, iElaps;
    dim3   block(1024);
    dim3   grid(nElem / block.x);
    res_from_gpu_h = (float*)malloc(nByte);
    float *a_dev, *b_dev;
    CHECK(cudaHostAlloc((float**)&a_host, nByte, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((float**)&b_host, nByte, cudaHostAllocMapped));
    CHECK(cudaMalloc((float**)&res_d, nByte));
    initialData(a_host, nElem);
    initialData(b_host, nElem);

    //=============================================================//
    iStart = cpuSecond();
    CHECK(cudaHostGetDevicePointer((void**)&a_dev, (void*)a_host, 0));
    CHECK(cudaHostGetDevicePointer((void**)&b_dev, (void*)b_host, 0));
    sumArraysGPU<<<grid, block>>>(a_dev, b_dev, res_d);
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    iElaps = cpuSecond() - iStart;
    //=============================================================//
    printf("zero copy memory elapsed %lf ms \n", iElaps);
    printf("Execution configuration %d,%d \n", grid.x, block.x);

    //-----------------------normal memory---------------------------
    float* a_h_n            = (float*)malloc(nByte);
    float* b_h_n            = (float*)malloc(nByte);
    float* res_h_n          = (float*)malloc(nByte);
    float* res_from_gpu_h_n = (float*)malloc(nByte);
    memset(res_h_n, 0, nByte);
    memset(res_from_gpu_h_n, 0, nByte);

    float *a_d_n, *b_d_n, *res_d_n;
    CHECK(cudaMalloc((float**)&a_d_n, nByte));
    CHECK(cudaMalloc((float**)&b_d_n, nByte));
    CHECK(cudaMalloc((float**)&res_d_n, nByte));

    initialData(a_h_n, nElem);
    initialData(b_h_n, nElem);

    //=============================================================//
    iStart = cpuSecond();
    CHECK(cudaMemcpy(a_d_n, a_h_n, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d_n, b_h_n, nByte, cudaMemcpyHostToDevice));
    sumArraysGPU<<<grid, block>>>(a_d_n, b_d_n, res_d_n);
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    iElaps = cpuSecond() - iStart;
    //=============================================================//
    printf("device memory elapsed %lf ms \n", iElaps);
    printf("Execution configuration %d,%d \n", grid.x, block.x);

    //-----------------------CPU Memory--------------------------------
    sumArrays(a_host, b_host, res_h, nElem);
    checkResult(res_h, res_from_gpu_h, nElem);

    cudaFreeHost(a_host);
    cudaFreeHost(b_host);
    cudaFree(res_d);
    free(res_h);
    free(res_from_gpu_h);

    cudaFree(a_d_n);
    cudaFree(b_d_n);
    cudaFree(res_d_n);

    free(a_h_n);
    free(b_h_n);
    free(res_h_n);
    free(res_from_gpu_h_n);
    return 0;
}
```

结果如下：

```shell
(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./sumArrayZeroCopy 10
Vector size:1024
zero copy memory elapsed 0.000154 ms 
Execution configuration<<<1,1024>>>
device memory elapsed 0.000020 ms 
Execution configuration<<<1,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./sumArrayZeroCopy 12
Vector size:4096
zero copy memory elapsed 0.000168 ms 
Execution configuration<<<4,1024>>>
device memory elapsed 0.000025 ms 
Execution configuration<<<4,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./sumArrayZeroCopy 14
Vector size:16384
zero copy memory elapsed 0.000192 ms 
Execution configuration<<<16,1024>>>
device memory elapsed 0.000050 ms 
Execution configuration<<<16,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./sumArrayZeroCopy 16
Vector size:65536
zero copy memory elapsed 0.000361 ms 
Execution configuration<<<64,1024>>>
device memory elapsed 0.000134 ms 
Execution configuration<<<64,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./sumArrayZeroCopy 18
Vector size:262144
zero copy memory elapsed 0.000993 ms 
Execution configuration<<<256,1024>>>
device memory elapsed 0.000509 ms 
Execution configuration<<<256,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./sumArrayZeroCopy 20
Vector size:1048576
zero copy memory elapsed 0.003397 ms 
Execution configuration<<<1024,1024>>>
device memory elapsed 0.001372 ms 
Execution configuration<<<1024,1024>>>
Check result success!
```

将结果统计为表格：

| 数据规模 n($2^n$) | 常规内存 (us) | 零拷贝内存 (us) |
|:-----------------:|:-------------:|:---------------:|
|        10         |      2.0      |       15.4       |
|        12         |      2.5      |       16.8       |
|        14         |      5.0      |       19.2       |
|        16         |     13.4      |       36.1       |
|        18         |     50.9      |      99.3       |
|        20         |     137.2     |      339.7      |

可以看出，零拷贝内存的执行时间比常规内存长很多，特别是在处理大规模数据时，两者的执行时间差距会进一步拉大。由于主机和设备之间通过 PCIe 连接，因此零拷贝内存的传输会非常耗时。从选择上来说，如果你想共享主机和设备端的少量数据，零拷贝内存可能会是一个不错的选择，因为它简化了编程并且性能表现较好。

异构计算系统有两种常见架构：

- **集成架构**：CPU 和 GPU 集成在同一芯片上，共享主存，因此零拷贝内存在性能和可编程性上表现更优
- **离散架构**：设备通过 PCIe 总线连接到主机，此时零拷贝内存只在特殊情况下有优势。

需要注意的是，为避免数据冲突，需要同步内存访问，尤其是在主机和设备共享固定内存时。最后，避免过度使用零拷贝内存，由于延迟较高，读取设备核函数可能会很慢。

#### 2.5. 统一虚拟寻址 Unified Virtual Addressing

**统一虚拟寻址(UVA)** 在 CUDA 4.0 中被引入，支持 64 位 Linux 系统。通过 UVA，主机内存和设备内存可以共享同一个虚拟地址空间，如下图所示：

![Unified Virtual Addressing](/images/Professional%20CUDA%20C%20Programming/Unified%20Virtual%20Addressing.png)

在没有 UVA 之前，我们需要管理指向主机内存和设备内存的指针。尤其是编写 C 语言代码的时候，多个指针指向不同的数据容易导致混乱，而在 CUDA C 中要经常处理主机和设备内存之间的复制操作，有时候处理起来比较麻烦。但是一旦有了 UVA，指向的内存空间对我们的应用程序代码来说就变得透明了。

通过 UVA，可以使用 `cudaHostAlloc` 分配的固定主机内存会同时具有相同的主机和设备地址。我们可以直接将返回的地址传递给核函数，而不必担心内存管理方面的复杂性。

前面的零拷贝内存，可以知道以下几个方面：

-   分配映射的固定主机内存
-   使用 CUDA 运行时函数获取映射到固定内存的设备指针
-   将设备指针传递给核函数

使用 UVA 后，我们不再需要使用 `cudaHostGetDevicePointer` 函数来获取设备上访问零拷贝内存的指针了，`chapter04/sumArrayZeroCopy.cu` 可以进一步简化（`chapter04/UVA.cu`）：

```C
  float *a_host,*b_host,*res_d;  
  CHECK(cudaHostAlloc((float**)&a_host,nByte,cudaHostAllocMapped));  
  CHECK(cudaHostAlloc((float**)&b_host,nByte,cudaHostAllocMapped));  
  CHECK(cudaMalloc((float**)&res_d,nByte));  
  res_from_gpu_h=(float*)malloc(nByte);  
  
  initialData(a_host,nElem);  
  initialData(b_host,nElem);  
  
  dim3 block(1024);  
  dim3 grid(nElem/block.x);  
  sumArraysGPU<<<grid,block>>>(a_host,b_host,res_d);
```

结果如下：

```shell
(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./UVA 10
Vector size:1024
zero copy memory elapsed 0.000151 ms 
Execution configuration<<<1,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./UVA 12
Vector size:4096
zero copy memory elapsed 0.000171 ms 
Execution configuration<<<4,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./UVA 14
Vector size:16384
zero copy memory elapsed 0.000190 ms 
Execution configuration<<<16,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./UVA 16
Vector size:65536
zero copy memory elapsed 0.000354 ms 
Execution configuration<<<64,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./UVA 18
Vector size:262144
zero copy memory elapsed 0.000998 ms 
Execution configuration<<<256,1024>>>
Check result success!

(DeepLearning) linxi@linxi1989:~/bin/chapter04$ ./UVA 20
Vector size:1048576
zero copy memory elapsed 0.003527 ms 
Execution configuration<<<1024,1024>>>
Check result success!
```

将结果统计为表格：

| 数据规模 n($2^n$) | 常规内存 (us) | 零拷贝内存 (us) | UVA (us) |
|:-----------------:|:-------------:|:---------------:|:--------:|
|        10         |      2.0      |      15.4       |   15.1   |
|        12         |      2.5      |      16.8       |   17.1   |
|        14         |      5.0      |      19.2       |   19.0   |
|        16         |     13.4      |      36.1       |   35.4   |
|        18         |     50.9      |      99.3       |   99.8   |
|        20         |     137.2     |      339.7      |  352.7   |

可以看出，使用 UVA 后基本和原来的代码执行时间上没区别。

#### 2.6 统一内存寻址 Unified Memory

CUDA 6.0 引入了**统一内存寻址**这一新特性，旨在**简化 CUDA 编程模型中的内存管理**。统一内存中包含一个**托管内存池**，这个内存池可以在 CPU 和 GPU 之间共享，**已分配的内存空间可以通过同一个指针在 CPU 和 GPU 上直接访问**。底层系统会自动处理主机和设备之间的数据传输，使得数据传输对应用程序透明化，从而大大简化了代码编写。

**统一内存寻址采用了一种“指针到数据”的编程模型，类似于零拷贝**。不同的是，零拷贝内存的分配由主机完成且需要手动传输数据，而统一内存寻址则由底层系统自动进行管理。

在使用统一内存时，我们可以传递两种类型的内存给核函数：**托管内存**和**未托管内存**。托管内存是由**底层系统自动分配**的，而未托管内存则是我们手动分配的。我们可以使用`__managed__` 关键字来修饰静态分配的托管内存变量，指定其作用域为整个文件。

```C
__device__ __managed__ int y;
```

还可以使用以下 API 动态分配托管内存：

```C
cudaError_t cudaMallocManaged(void ** devPtr,size_t size,unsigned int flags=0)
```

**在 CUDA 6.0 之前的版本中**，设备代码无法调用 `cudaMallocManaged` 函数，只能由主机代码调用。因此，所有的托管内存必须在主机代码中进行动态声明或者全局静态声明。在后续的内容中，我们将会深入研究统一内存寻址相关的知识。

### 3. 内存访问模式  MEMORY ACCESS PATTERNS

在 CUDA 执行模型中，线程束是执行的基本单位，而全局内存则是大部分设备端数据访问的起点。为了提高全局加载效率并避免内存带宽限制导致性能下降，我们需要最大程度地利用好全局内存带宽。因此，在优化核函数性能时，正确使用全局内存非常重要。

#### 3.1. 对齐与合并访问 Aligned and Coalesced Access

全局内存的加载和存储过程通常通过缓存实现，具体如下图所示:

![Global Memory Access](/images/Professional%20CUDA%20C%20Programming/Global%20Memory%20Access.png)

全局内存是在逻辑层面的模型，在编程时我们需要考虑两种不同的模型：

1.  **逻辑层面**：包括在写程序时使用的**一维（多维）数组**、**结构体**和定义的**变量**，这些是在逻辑层面上操作的。
2.  **硬件角度**：指代实际的硬件结构，包括一块 DRAM 上的**电信号**以及最底层内存驱动代码所处理的**数字信号**。

在 CUDA 中，每个 SM 都拥有自己的 L1 缓存，而 L2 缓存则是所有 SM 共享的。除了 L1 缓存之外，还有**只读缓存**和**常量缓存**，这些将在后续详细介绍。

当核函数需要从全局内存（DRAM）读取数据时，数据的读取以 **128 字节**或 **32 字节**内存事务的粒度实现，具体取决于访问方式：

-   如果**使用一级缓存**，每次加载数据的粒度为 128 字节。
-   如果**不使用一级缓存**，而是只使用二级缓存，那么加载数据的粒度为 32 字节。

**对于 CPU 而言，一级缓存或二级缓存不能被直接编程**，但 CUDA 支持通过编译指令来禁用一级缓存。

在优化内存时，我们要着重关注以下两个特性：

1.  **对齐内存访问**：当一个内存事务的**首个访问地址是缓存粒度（32或128字节）的偶数倍时**，例如二级缓存 32 字节的偶数倍为 64，一级缓存 128 字节的偶数倍为 256，这种情况被称为对齐内存访问。非对齐访问则是指除上述情况外的其他情况，非对齐内存访问会导致带宽浪费。
2.  **合并内存访问**：当一个线程束内的线程访问的内存都在一个内存块内时，就会出现合并访问的情况。

对齐合并内存访问是指**线程束从对齐的内存地址开始访问一个连续的内存块**。对齐合并访问是最理想化，也是最高速的访问方式。为了最大化全局内存吞吐量，应尽量将线程束访问内存组织成对齐合并的方式，以达到最高的效率。下面以一个例子来说明：

1. 一个线程束加载数据，使用一级缓存，并且这个事务所请求的所有数据都在一个 128 字节的对齐的地址段上。上面蓝色区域代表全局内存，下面橙色区域表示线程束需要的数据，绿色区域则代表对齐的地址段
  
![Aligned Coalesced Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Aligned%20Coalesced%20Memory%20Accesses.png)

2. 如果一个事务加载的数据分布在不同的对齐地址段上，会有以下两种情况：
   - **数据是连续的，但不在同一个对齐的段上**。例如，请求访问的数据分布在内存地址`1~128`，这种情况下 `0~127` 和 `128~255` 这两段数据将需要分别传递到 SM 两次
   - **数据是不连续的，也不在同一个对齐的段上**。例如，请求访问的数据分布在内存地址 `0~63` 和 `128~191`，这种情况下也需要两次加载。

![None Aligned Coalesced Memory Accesses](/images/Professional%20CUDA%20C%20Programming/None%20Aligned%20Coalesced%20Memory%20Accesses.png)

上图是一个一个典型的线程束，其中数据是分散的。例如，thread0 的请求在 128 之前，后续又有请求在 256 之后，因此需要进行**三个内存事务**。而利用率，即从主存取回的数据被实际使用的比例，为 $\frac {128}{128×3}$。利用率低会导致带宽浪费，最极端的情况是，如果每个线程的请求都位于不同的段上，也就是一个 128 字节的事务只有 1 个字节是有用的，那么利用率只有 $\frac {1}{128}$。

总结一下内存事务的优化关键：**用最少的事务次数满足最多的内存请求。事务数量和吞吐量的需求随设备的计算能力变化**。

#### 3.2. 全局内存读取 Global Memory Reads

在 SM 加载数据时，根据不同的设备和类型，可分为三种路径：

1.  一级和二级缓存
2.  常量缓存
3.  只读缓存

一级和二级缓存通常是默认路径，在代码中需要显式声明才能使用常量和只读缓存。另外，要提升性能还取决于使用的访问模式。在 Fermi 架构的 GPU（计算能力为 2.x）和 Kepler K40 以及更高架构的 GPU（计算能力为 3.5及以上），可以通过编译器标志来启用或禁用全局内存负载的一级缓存。

编译器**禁用**一级缓存的选项是：

```shwll
-Xptxas -dlcm=cg
```

编译器**启用**一级缓存的选项是：

```shwll
-Xptxas -dlcm=ca
```

**当一级缓存被禁用时，全局内存加载请求将直接进入二级缓存，如果二级缓存缺失，则由DRAM完成请求**。每次内存事务可以分为一个、两个或四个部分执行，每个部分有 32 个字节，也就是每次可以操作 32、64 或 128 字节。

启用一级缓存后，当 SM 有全局加载请求时，**首先会尝试调用一级缓存，如果一级缓存缺失，则尝试二级缓存，如果二级缓存也没有，则直接访问DRAM**。在某些设备上，一级缓存不用来缓存全局内存访问，而是用来存储寄存器溢出的本地数据，比如 Kepler 的 K10和K20。

内存加载可以分为两类：

-   缓存加载
-   无缓存加载

内存访问具有以下特点：

-   **是否使用缓存**：即一级缓存是否介入加载过程
-   **对齐与非对齐**：即访问的第一个地址是否是 32 的倍数
-   **合并与非合并**：即访问连续数据块时是否进行合并操作

##### 3.2.1. 缓存加载 Cached Loads

以下是使用一级缓存的加载过程，图片清晰易懂，无需过多文字解释

1. **对齐合并的访问**，利用率 100%

![Aligned and Coalesced Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Aligned%20and%20Coalesced%20Memory%20Accesses.png)

2. **对齐非合并的访问**，每个线程访问的数据都在一个块内，但是位置是交叉的，利用率100%

![Aligned None Coalesced Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Aligned%20None%20Coalesced%20Memory%20Accesses.png)

3. **非对齐合并的访问**，如果线程束请求一个连续的非对齐的32个4字节数据，这些数据会跨越两个内存块，同时没有对齐。在启用一级缓存的情况下，需要进行两个128字节的事务来完成加载操作。
   
![None Aligned but Coalesced Memory Accesses](/images/Professional%20CUDA%20C%20Programming/None%20Aligned%20but%20Coalesced%20Memory%20Accesses.png)

4. 当线程束中的所有线程请求同一个地址时，这些数据必定落在同一个缓存行范围内。缓存行是主存中可以一次读取到缓存的一段数据。若每个请求为4字节数据，那么在使用一级缓存时，利用率为 $\frac {4}{128}$

![Threads Request Same Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Threads%20Request%20Same%20Memory%20Accesses.png)

5. **最坏的情况**，线程束内的每个线程请求的数据都位于不同的缓存行上，其中 $1≤N≤32$。因此，当请求32个4字节的数据时，需要N个事务来完成加载操作，此时的利用率为 $\frac{1}{N}$。

![None Aligned None Coalesced Memory Accesses](/images/Professional%20CUDA%20C%20Programming/None%20Aligned%20None%20Coalesced%20Memory%20Accesses.png)

CPU 和 GPU 的一级缓存有明显的不同。GPU 的一级缓存可以通过编译选项等进行控制，而 CPU 的一级缓存不可控制。此外，CPU 的一级缓存使用一种替换算法，该算法考虑到了数据的使用频率和时间局部性，而 GPU 的一级缓存缺乏这种替换算法。

##### 3.2.2. 无缓存加载 Uncached Loads

没有使用一级缓存的加载，即直接从内存中读取数据，加载粒度变为32字节。**较细粒度的加载有助于提高内存利用率**，类比于选择喝水时可以选择 500ml 的大瓶装还是 250ml 的小瓶装。例如，当你需要 400ml 的水时，选择大瓶更方便，但如果需要 200ml 的水，则选择小瓶的利用率更高。细粒度的加载就像选择小瓶喝水，虽然每次加载量小，但利用率更高。在某些情况下，对于前文提到的缓存使用场景，细粒度加载可能会带来更好的效果。

1. **对齐合并的访问**，使用4个段，利用率 100%

![Aligned Coalesced Uncached Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Aligned%20Coalesced%20Uncached%20Memory%20Accesses.png)

2. **对齐非合并的访问**，都在四个段内，且互不相同，利用率100%

![Aligned None Coalesced Uncached Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Aligned%20None%20Coalesced%20Uncached%20Memory%20Accesses.png)


3. **非对齐合并的访问**，一个段的大小为32字节，因此，一个连续的128字节请求，即使没有按照段的边界对齐，最多也只涉及到5个段。因此，内存的利用率可以达到 5/6 ≈ 80%

![None Aligned Coalesced Uncached Memory Accesses](/images/Professional%20CUDA%20C%20Programming/None%20Aligned%20Coalesced%20Uncached%20Memory%20Accesses.png)

4. 所有线程访问一个4字节的数据，那么此时的利用率是 4/32=12.5%

![Threads Request Same Uncached Memory Accesses](/images/Professional%20CUDA%20C%20Programming/Threads%20Request%20Same%20Uncached%20Memory%20Accesses.png)

5.  **最坏的情况**，当目标数据分散在内存的各个角落时，可能需要N个内存段来获取数据。在这种情况下，与使用一级缓存相比仍然具有优势，因为 $N×128$ 相比于 $N×32$ 还是大很多的。

![None Aligned None Coalesced Uncached Memory Accesses](/images/Professional%20CUDA%20C%20Programming/None%20Aligned%20None%20Coalesced%20Uncached%20Memory%20Accesses.png)

##### 3.2.3. 非对齐读取示例 Example of Misaligned Reads

代码在 `chapter04/sumArrayOffset.cu`：

```C
#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"

void sumArrays(float* a, float* b, float* res, int offset, const int size) {
    for (int i = 0, k = offset; k < size; i++, k++) {
        res[i] = a[k] + b[k];
    }
}

__global__ void sumArraysGPU(float* a, float* b, float* res, int offset, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = i + offset;

    if (k < n)
        res[i] = a[k] + b[k];
}

int main(int argc, char** argv) {
    int dev = 0;
    cudaSetDevice(dev);

    int power  = 18;
    int offset = 0;
    if (argc >= 3) {
        power  = atoi(argv[1]);
        offset = atoi(argv[2]);
    }
    int nElem = 1 << power;
    printf("Vector size:%d\n", nElem);
    int nByte=sizeof(float)*nElem;
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
    CHECK(cudaMemset(res_d, 0, nByte));
    initialData(a_h, nElem);
    initialData(b_h, nElem);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3   block(1024);
    dim3   grid(nElem / block.x);
    double iStart, iElaps;
    iStart = cpuSecond();
    sumArraysGPU<<<grid, block>>>(a_d, b_d, res_d, offset, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(res_from_gpu_h, res_d, nByte, cudaMemcpyDeviceToHost));
    printf("Execution configuration %d,%d Time elapsed %f sec --offset:%d \n", grid.x, block.x, iElaps, offset);

    sumArrays(a_h, b_h, res_h, offset, nElem);

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

有 L1 缓存实验结果：

```shell
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-05-13_CUDA/bin/chapter04$ ./sumArrayOffset_with_L1cache 24 1
Vector size:16777216
Execution configuration<<<16384,1024>>> Time elapsed 0.001471 sec --offset:1 
Check result success!

(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-05-13_CUDA/bin/chapter04$ ./sumArrayOffset_with_L1cache 24 11
Vector size:16777216
Execution configuration<<<16384,1024>>> Time elapsed 0.001490 sec --offset:11 
Check result success!

(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-05-13_CUDA/bin/chapter04$ ./sumArrayOffset_with_L1cache 24 128
Vector size:16777216
Execution configuration<<<16384,1024>>> Time elapsed 0.001466 sec --offset:128 
Check result success!
```

```shell
root@linxi1989:/home/linxi/DevKit/Projects/2024-05-13_CUDA/bin/chapter04# nvprof --metrics gld_efficiency ./sumArrayOffset_with_L1cache 24 1
==25765== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumArraysGPU(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%

root@linxi1989:/home/linxi/DevKit/Projects/2024-05-13_CUDA/bin/chapter04# nvprof --metrics gld_efficiency ./sumArrayOffset_with_L1cache 24 11
==25837== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumArraysGPU(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%

root@linxi1989:/home/linxi/DevKit/Projects/2024-05-13_CUDA/bin/chapter04# nvprof --metrics gld_efficiency ./sumArrayOffset_with_L1cache 24 128
==25891== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumArraysGPU(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
```


没有 L1 缓存实验结果：

```shell
(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-05-13_CUDA/bin/chapter04$ ./sumArrayOffset_without_L1cache 24 1
Vector size:16777216
Execution configuration<<<16384,1024>>> Time elapsed 0.001486 sec --offset:1 
Check result success!

(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-05-13_CUDA/bin/chapter04$ ./sumArrayOffset_without_L1cache 24 11
Vector size:16777216
Execution configuration<<<16384,1024>>> Time elapsed 0.001522 sec --offset:11 
Check result success!

(DeepLearning) linxi@linxi1989:~/DevKit/Projects/2024-05-13_CUDA/bin/chapter04$ ./sumArrayOffset_without_L1cache 24 128
Vector size:16777216
Execution configuration<<<16384,1024>>> Time elapsed 0.001471 sec --offset:128 
Check result success!
```

```shell
root@linxi1989:/home/linxi/DevKit/Projects/2024-05-13_CUDA/bin/chapter04# nvprof --metrics gld_efficiency ./sumArrayOffset_without_L1cache 24 1
==25964== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumArraysGPU(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%
          
root@linxi1989:/home/linxi/DevKit/Projects/2024-05-13_CUDA/bin/chapter04# nvprof --metrics gld_efficiency ./sumArrayOffset_without_L1cache 24 11
==26018== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumArraysGPU(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency      80.00%      80.00%      80.00%

root@linxi1989:/home/linxi/DevKit/Projects/2024-05-13_CUDA/bin/chapter04# nvprof --metrics gld_efficiency ./sumArrayOffset_without_L1cache 24 128
==26053== Metric result:
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "NVIDIA GeForce GTX 1060 6GB (0)"
    Kernel: sumArraysGPU(float*, float*, float*, int, int)
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%
```

统计为表格如下：

| 偏移量 | 有 L1 缓存的时间（秒） | 有 L1 缓存的全局内存加载效率 | 无 L1 缓存的时间（秒） | 无 L1 缓存的 全局内存加载效率 |
|-----|---------------|-----------------|---------------|-----------------|
| 1   | 0.001471      | 80%             | 0.001486      | 80%             |
| 11  | 0.001490      | 80%             | 0.001522      | 80%             |
| 128 | 0.001466      | 100%            | 0.001471      | 100%            |


这里我们使用的指标是：

$$
全局加载效率 = \frac {请求的全局内存加载吞吐量}{所需的全局内存加载吞吐量}
$$

通过观察表格数据，可以得出结论：

- 偏移量为0时（128），效率最高，**有偏移量会导致全局内存加载效率降低**；
- **有 L1 缓存在所有偏移量下均略快于无 L1 缓存**，缓存缺失对**非对齐访问**的性能影响更大- 

#### 


### 4. 

### 5. 

### 6. 

---

## 参考引用

### 书籍出处

- [CUDA C编程权威指南](../../../asset/CUDA%20&%20GPU%20Programming/CUDA%20C编程权威指南.pdf)
- [Professional CUDA C Programming](../../../asset/CUDA%20&%20GPU%20Programming/Professional%20CUDA%20C%20Programming.pdf)

### 网页链接

- [人工智能编程 | 谭升的博客](https://face2ai.com/program-blog/)
- [Tony-Tan/CUDA_Freshman](https://github.com/Tony-Tan/CUDA_Freshman/tree/master)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)