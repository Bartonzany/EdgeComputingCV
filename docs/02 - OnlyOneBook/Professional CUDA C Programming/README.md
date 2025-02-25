# CUDA C 编程入门 0 - 碎碎念

---

### 0. 前言 “Only One Book”

之所以开这个系列，主要受一篇博客 [“Only One Book”计划](https://zhuanlan.zhihu.com/p/542488093) 启发，即**对每个细分领域/方向/阶段只推荐一本（个人认为）最好的入门书，免去读者在书海中挑书/选书的烦恼**（援引博主原话）。之前因为个人性格比较懒散，一直有计划和想法入门 CUDA，但一直没怎么付出实际行动，且积累的内容东一榔头西一锤子的，总不能形成一套完整的知识体系。看到了这篇博客，让我想到了之前积累了点看 《CUDA C编程权威指南》的内容，想着是否也可以整成个系列，放到 Github 或者知乎、CSDN 这些平台上，在自己后续回顾的时候可以有点参考的用处。

**“Only One Book”，这个想法是否可行？或者说是否可以稍微扩展些，在以某本书为主要参考的基础上，加入其他书/博客/论文的一些内容？** 这个想法应该可行，在之前的学习中，忙于在各种 筛选资料/找资料/资料整理/资料取舍 的问题上纠结，导致真正用在学这些资料的时间总是不够，学得迷迷糊糊，做烂尾了许多想做的事情，日后又懊悔没有认真学进去，后面学其他东西时又重新走一遍这样的流程，陷入了死循环。资料在精不在多，所以写下这篇博客，也是想着说就算选的资料真的没有那么好，脑袋比较愚笨不能吃透资料的内容，但能吸收到 70~80%，那也是好的，总比什么也没做，做什么都烂尾好的多，也能够督促自己可以坚持下去能够有所收获。

之前在实习的时候（地平线），公司刚出了最新的具身智能的部署板卡，后面也动了心思想学学 CUDA，积累了部分内容，那不如就直接再次总结成系列博客，也是另一个写下这篇入门指南的想法吧。当然最后没能实习留下来，部门没有hc，也有可能是自己太菜了 mentor 委婉的说法。秋招时简历也不好看，华子也没泡出来，至今也没 offer，菜狗属实了 o(╥﹏╥)o。已经秋招结束春招都准备开始了，不会真的要失业了吧。看着同门的大厂offer，说不羡慕那是假的，真就我最菜了。

扯远了说回来，以前在看博客的时候，总是觉得能输出内容，尤其是能输出高质量的内容，真的很厉害，羡慕那些专业博主和up主可以做到内容专业且有条理，在想我哪天可以不被动的输入内容，也输出点有用的东西呢？虽说可能做不到那么高质量，但如果可以帮到一些小忙，能给读者一些帮助，那我觉得算是比较成功了。

这是我写下的第一篇博客，哈哈，希望可以坚持下去不烂尾 O(∩_∩)O，把这个系列做完，学到些有用的东西。

### 1. 环境要求

既然使用 CUDA，那最后还是在黄矿老板的卡上动手动脚（这个逼赚麻了）。系统的话，建议能上 Linux 服务器还是尽量 Linux，毕竟深度学习炼丹还是用 Ubuntu 多的嘛。安装CUDA 驱动和一些需要的包，网上已经很多教程了，这里偷点懒就不详细说明啦。这个系列的代码是用 CMake 编译脚本来编译的，`build.sh` 执行代码文件。所以还是尽量不使用 windows 吧。

### 2. 代码目录结构

和大部分 C/C++ 工程一样，使用 CMake 管理并编译代码，每级子目录另有 CMakeLists.txt 将代码文件链接至根目录的 CMakeLists.txt，最后用 `sh build.sh` 脚本编译后，二进制文件就存放至 bin 文件夹中了。.clang-format 是个人参照 Google C/C++ 代码风格，进行了部分修改。需要注意的要用 clang-format-15，默认版本没法在 vscode 上进行格式化。

以下是编写好的部分代码文件树，代码会陆陆续续上传到 [个人 Github 仓库](https://github.com/Bartonzany/EdgeComputingCV/tree/main/docs/02%20-%20OnlyOneBook/Professional%20CUDA%20C%20Programming) 中：

```shell
├── bin
│   ├── chapter01
│   │   └── helloWorld
│   ├── chapter02
│   │   ├── checkDeviceInfor
│   │   ├── checkDimension
│   │   ├── checkThreadIndex
│   │   ├── checkThreadIndexFloat
│   │   ├── defineGridBlock
│   │   ├── sumArrays
│   │   ├── sumArraysOnGPU-small-case
│   │   ├── sumArraysTimer
│   │   └── sumMatrix
│   └── chapter03
│       ├── nestedHelloWorld
│       ├── reduceInteger
│       ├── reduceUnrolling
│       ├── simpleDeviceQuery
│       ├── simpleDivergence
│       └── sumMatrix2D
├── build
├── build.sh
├── chapter01
│   ├── CMakeLists.txt
│   └── helloWorld.cu
├── chapter02
│   ├── checkDeviceInfor.cu
│   ├── checkDimension.cu
│   ├── checkThreadIndex.cu
│   ├── checkThreadIndexFloat.cu
│   ├── CMakeLists.txt
│   ├── defineGridBlock.cu
│   ├── sumArrays.cu
│   ├── sumArraysOnGPU-small-case.cu
│   ├── sumArraysTimer.cu
│   └── sumMatrix.cu
├── chapter03
│   ├── CMakeLists.txt
│   ├── nestedHelloWorld.cu
│   ├── reduceInteger.cu
│   ├── reduceUnrolling.cu
│   ├── simpleDeviceQuery.cu
│   ├── simpleDivergence.cu
│   └── sumMatrix2D.cu
├── CMakeLists.txt
└── common
    └── common.h
```

### 3. 需要安装的软件

#### Vscode

应该是全宇宙最好用的轻量级 IDE 了，丰富的插件使它可以在任何平台上都能完成代码部署，且连接远程服务器拉代码也非常方便，甚至现在还集成了Copilot，强烈推荐！

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250131005340.png)

#### Obsidian

市面上许多的 Markdown 编辑器中，还是这款完全开源，插件丰富的用起来最舒服了 [Obsidian - Sharpen your thinking](https://obsidian.md/)。在 Vscode 中配置好 Git 和两边同步调整一下 Markdown 格式，就可以做到直接推上 Github，十分方便。并且插件市场有很多风格化的界面和小工具，方便内容排版和编辑，可以专心在内容编写上

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250131005902.png)

以下是个人风格化的界面，个人认为还是比较好看的：

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250131010018.png)

#### ChatGPT & DeepSeek

最近DeepSeek火出圈，俨然有种国内大模型的希望。DeepSeek 虽然在某些性能指标上达不到 ChatGPT 的效果，但毕竟用了才 600 多万美金实现了 95% 的性能，而且还是真正的开源，中美两大国的 AI 理念可以看出一斑。希望 DeepSeek 也可以日后成为操作系统的 Linux，成为世界开源大模型的标准和规则。后续的内容也会使用这两个大模型优化一下，毕竟某些时候我真的觉得GPT写得比我好....

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250131011300.png)

![](/images/Professional%20CUDA%20C%20Programming/Pasted%20image%2020250131011400.png)

主要就是这三四个软件了，其他在 Ubunutu 上的配置按需安装，这里也不在赘述了。

### 4. 目录

- [1 - 简介](1%20-%20简介.md)
- [2 - CUDA 编程模型](2%20-%20CUDA%20编程模型.md)
- [3 - CUDA 执行模型](3%20-%20CUDA%20执行模型.md)
- [4 - CUDA 全局内存](4%20-%20CUDA%20全局内存.md)

### 5. 结语

这篇算是自己的一些小小个人总结，CUDA 的学习就从这篇开始吧！自己比较喜欢拍照片，附上几张最近在家附近拍的些许照片，广东过年时天气一直不错，晒晒冬日的太阳，暖洋洋的。

![](/images/Professional%20CUDA%20C%20Programming/DSC_7610.jpg)

![](/images/Professional%20CUDA%20C%20Programming/DSC_7606.jpg)

![](/images/Professional%20CUDA%20C%20Programming/DSC_7635.jpg)

![](/images/Professional%20CUDA%20C%20Programming/DSC_7688.jpg)

![](/images/Professional%20CUDA%20C%20Programming/DSC_7710.jpg)

---

## 参考引用 

### 书籍出处

- [CUDA C编程权威指南](../../../asset/CUDA%20&%20GPU%20Programming/CUDA%20C编程权威指南.pdf)
- [Professional CUDA C Programming](../../../asset/CUDA%20&%20GPU%20Programming/Professional%20CUDA%20C%20Programming.pdf)

### 网页链接

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)

编辑于 2025.01.30 日大年初二晚凌晨1.16