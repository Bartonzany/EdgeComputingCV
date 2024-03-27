# Adaround

## 背景 Background

### 量化公式

根据 [模型量化简介](/docs/Model_Accelaration/模型量化简介.md) 中量化公式：

$$\begin{align*}
x &= sx_{int} \tag{1} \\
x_{int} &= clamp(round(\lfloor{\frac xs}\rceil+z);0,2^b-1) \tag{2} \\
\widehat{x} &= q(x;s,z,b)=s[clamp(round(\lfloor{\frac xs}\rceil+z);0,2^b-1)-z] \tag{3} 
\end{align*}$$

$x$ 和 $x_{int}$ 分别是浮点数和定点数, $s$ 是量化参数中的缩放系数，**反量化后的 $\widehat{x}$ 和 $x$ 之间的差异，就是量化引起的损失**。

量化算法的目标就是让量化损失越小越好。在这个原始的量化公式中，真正引起较大量化误差的是 **round 函数**。在大部分量化算法中，round 都是采用**四舍五入** (即最近邻取整)、或者完全向上取整、或者完全向下取整这三种策略实现的。但论文的作者发现，不同的取整策略，对量化精度的影响是不同的。比如，下面这张表就是对 ResNet18 的第一层卷积的权重采用不同的 round 策略后，得到的模型准确率：

![Adaround - Comparison of different rounding schemes](/images/Model_Accelaration/Adaround%20-%20Comparison%20of%20different%20rounding%20schemes.png)

可以看到，**四舍五入** (Nearest) 的效果要远远好于**完全向上取整** (Ceil) 或者**完全向下取整** (Floor)，但如果采用**随机取整** (Stochastic，随机向上或向下取整)，效果虽然有大的波动，但确实存在超过四舍五入的情况，最好的实验结果甚至超过 10 个点还多。说明四舍五入并不一定是最优的，存在一些 round 策略，可以让量化的损失更小。

### 数学论证

论文进一步给出了数学证明。既然不同的 round 函数会对模型的准确率产生较大的影响，那么对于模型训练的损失函数同样会带来比较大的差异。因此可以从**损失函数**的角度出发，来看一下 round 对损失函数的影响。

假设模型的损失函数是 $L$，输入是 $x$，输出是 $y$，模型权重参数是 $w$ (通常是矩阵或者张量)。量化就是对 $w$ 中每个元素进行 round，把这个 round 带来的差异记为 $\Delta w$，量化后的权重为 $w+\Delta w$。

对模型进行量化后，损失函数的变化可以表示为: 

$$L(x,y,w+\Delta w)-L(x,y,w) \tag{4}$$

如果公式 (4) 的结果很小，那么就可以认为 round 带来的差异不足以使模型的效果发生大的变化。如何在不知道 $L$ 的情况下研究这个式子呢？论文采用了泰勒展开这个数学工具。

具体泰勒展开可以自行查找，这里直接给出在 $z=w$ 这个位置的展开公式：

$$L(z) = L(w) + \nabla_z L(w)(z-w) + \frac{\nabla^2_z L(w)}{2!}(z-w)^2 \\
+\cdots+\frac{\nabla^n_z L(w)}{n!}(z-w)^n \tag{5}$$

其中 $\nabla_z^n L$ 表示 $L$ 的n阶导数。将 $z=w+\Delta w$ 代入公式 (4):

$$\begin{align*}
L(z=w+\Delta w) =& L(w) + \nabla_z L(w)(w+\Delta w-w) \\
& + \frac{\nabla^2_z L(w)}{2!}(w+\Delta w-w)^2 \\
& + \cdots+\frac{\nabla^n_z L(w)}{n!}(w+\Delta w-w)^n\\
=& L(w) + \nabla_z L(w)(\Delta w) + \frac{\nabla^2_z L(w)}{2!}(\Delta w)^2 \\
& +\cdots+\frac{\nabla^n_z L(w)}{n!}(\Delta w)^n \tag{6}
\end{align*}$$

由于 n 取到 2 阶的时候就足够分析问题了，所以可以把 $n>2$ 的部分舍弃掉，再把 $x$ 和 $y$ 也放入 $L$ 中: 

$$\begin{align*}
& L(x,y,w+\Delta w) - L(x,y,w) \\
\approx & \nabla_z L(x,y,w)(\Delta w) + \frac{\nabla^2_z L(x,y,w)}{2!}(\Delta w)^2 \\
=& \Delta w^T \nabla_z L(x,y,w) + \frac{1}{2} \Delta w^T \nabla^2_z L(x,y,w) \Delta w \\
=& \Delta w^T g^{(w)} + \frac{1}{2} \Delta w^T H^{(w)} \Delta w \tag{7}
\end{align*}$$

推导出这个公式的目的是想确认 round 带来的误差 $w$ 对 $L$ 的影响，即 $\Delta w$ 的变化会给公式 (7) 的结果带来什么变化。

对于损失函数 $L$ 来说，当模型训练收敛的时候，导数一般是趋于 0 的 (训练的时候通常是用一阶导数)，即 $w$ 在小范围内波动 (如 $\Delta w$ 很小时)，一阶导 $g^{(w)}$  都趋于 0，所以第一项可以忽略。

分析第二项 $H^{(w)}$，假设 $\Delta w^T = [\Delta w_1 \: \Delta w_2 ]$，且:

$$H^{(w)} = 
\begin{bmatrix}
1 & 0.5 \\
0.5 & 1 
\end{bmatrix}$$

$$\Delta w^T H^{(w)} \Delta w = \Delta w_1^2 + \Delta w_2^2 + \Delta w_1 \Delta w_2 \tag{8}$$

对于 $\Delta w_1 $ 和 $\Delta w_2$，四舍五入的 round 一定是最优的，因为四舍五入带来的 $\Delta w$ 在数值上是最小的。但对于 $\Delta w_1 \Delta w_2$，情况就跟它们的符号有比较大的关系。如果它们符号不同，相乘后数值就是负的，相当于公式 (7) 反而变小了，这样模型的量化误差也是减小的。而四舍五入并不能保证这一点。

## 研究目标 Research Objective

寻找到最好的 Round 函数权重取舍机制，更高效的用于后训练量化，简化流程并优化准确性而无需进行大量微调参数。

## 方法 Methods

### 寻找最好的 Round 函数 Task loss based rounding

最好的 round 函数需要让量化的权重给损失函数 $L$ 带来的变化最小，也就是让公式 (7) 尽可能的小。而从前面的分析来看，其实就是让 $\frac{1}{2} \Delta w^T H^{(w)} \Delta w$ 这一项尽可能的小。

假设量化前模型的权重是 $w$，量化后是 $\widehat{w}$。根据量化公式 (3)，可以推出: 

$$\widehat{w} = s[clamp(round(\lfloor{\frac ws}\rceil+z);0,2^b-1)-z] \tag{9}$$

对于 round 来说，无非是向上取整和向下取整两种情况。可以把这两种情况表示为: 

$$\begin{align*}
w^{floor} = s[clamp(round(\lfloor{\frac ws}\rceil+z);0,2^b-1)-z] \tag{10}  \\
w^{ceil} = s[clamp(round(\lfloor{\frac ws}\rceil+z);0,2^b-1)-z] \tag{11}  
\end{align*}$$

$\widehat{w} \in [w^{floor}, w^{ceil}]$, $\Delta w = w - \widehat{w}$，再结合前面的分析，可以把问题建模成:

$$\underset{\widehat{w}}{argmin} \: E[\frac{1}{2} \Delta w^T H^{(w)} \Delta w] \tag{12}$$

这个数学公式意思是要通过一堆样本，**求出期望最小的情况下对应的 $\widehat{w}$ 的值，并且这个值在区间 $[w^{floor}, w^{ceil}]$ 之间**。由于深度神经网络的结构通常很复杂，为了避免问题复杂化，论文限制这个优化函数只针对网络中的每一层单独进行，如果能把每一层的量化误差优化好，整体误差也会小很多。

**问题简化成求 $\frac{1}{2} \Delta w^T H^{(w)} \Delta w$ 会不会存在问题**？论文做了一个小实验来证实这一点。作者对 ResNet18 的第一个卷积层进行量化，图里每个点都代表不同 round 策略下的准确率以及 $\frac{1}{2} \Delta w^T H^{(w)} \Delta w$ 的数值大小。从图中可以看出一个大致的规律: **$\frac{1}{2} \Delta w^T H^{(w)} \Delta w$ 的值越大，准确率是逐渐下降的，二者大体上是一个线性的关系**，说明用 $\frac{1}{2} \Delta w^T H^{(w)} \Delta w$ 代替损失函数作为优化目标，没有大的问题。

![AdaRound Figure 7](/images/Model_Accelaration/AdaRound%20Figure%207.png)

### 从泰勒级数到损失函数 From Taylor expansion to local loss

#### Hessian 矩阵计算

首先第一个难点, **$H^{(w)}$ Hessian 矩阵计算复杂**。假设有一个一层的全连接网络，由于全连接层就是一个矩阵乘法，因此输出为 $z=wx$。那么，对于二阶导，它的计算方式是：

$$\begin{align*}
H^{(w)} &= \nabla_w^2L(w) \\
&= \frac {\partial}{\partial w}[\frac {\partial L}{\partial z} \frac {\partial z}{\partial w}] \quad (先求一阶导) \\
&= \frac {\partial}{\partial w}[\frac {\partial L}{\partial z}]x \\
&= \frac{\frac {\partial L}{\partial z}}{\partial z} \frac {\partial z}{\partial w}x \quad (用
一阶导对w继续求二阶导) \\
&= \frac {\partial^2 L}{\partial z \partial z}xx \tag{13}
\end{align*}$$

论文中的公式为:

$$H^{(w)} = E[x^{(l-1)}x^{(l-1)^T} \otimes \nabla^2_{z^{(l)}} \zeta] \tag{14}$$

这个公式中，计算比较复杂的是 $\nabla^2_z L$，它是 Loss 函数对网络中间每一层的特征求二阶导。反向传播 BP 是需要逐层求导计算的，如果网络的层数比较多，这个计算过程就会很复杂。因此，论文做了一个假设: **$\nabla^2_z L$ 是一个对角阵**，即矩阵中除了对角线上有数值外，其他位置的数值都是 0。不过，这样计算上还是很复杂，因此论文又做了进一步假设: **$\nabla^2_z L_{i,i} = c_i$，即这个对角阵中的元素都是常数**，这样它跟公式 (12) 的优化就没有关系了，可以直接丢掉。基于这两个假设, $H^{(w)}$ 可以直接去掉: 

![Hessian Decomposes Into Independent Sub-problem](/images/Model_Accelaration/Hessian%20Decomposes%20Into%20Independent%20Sub-problem.png)

最后得到简化后的优化目标: 

$$\underset{\widehat{w}}{argmin} E[(\Delta wx)^2] \tag{15}$$

#### NP 复杂度

前面虽然对优化目标简化了很多，但如果想直接优化还是存在很多困难的。**问题的关键就在于求解空间 $\widehat{w}$ 是离散的 (只有 $w^{floor}$, $w^{ceil}$ 两个解)，没法直接通过数学优化的方式求出最优解，只能暴力枚举所有的解，然后找出优化目标最小的解**。但神经网络的参数量巨大 (现代神经网络都是几十上百万的参数)，求解空间非常巨大，是一个 NP-Hard 难度的问题 (计算机也要算很久很久很久)。因此不能用一般方法解决这个求解问题。

### AdaRound

AdaRound 方法核心思想是：**既然离散空间不好优化，那就把它松弛成连续空间**。即可以把公式 (15) 替换成:

$$\begin{align*}
\underset{v}{argmin} ||wx - \tilde{w}x||^2_F + \lambda f_{reg}(v) \tag{16} \\
\tilde{w} = s \times clamp(\lfloor \frac{w}{s} \rfloor + h(v);n,p) \tag{17} \\  
\end{align*}
$$

$||wx - \hat{w}x||^2_F$ 表示两个特征矩阵之间的 Frobenius 范数，可以就简单把它当作公式 (15) 中的目标函数，使用 n 和 p 来表示整数网格限制 ($n = \frac{q_{min}}{s}$ 和 $p = \frac{q_{max}}{s}$)，另新引入辅助函数 $f_{reg}$。现在，需要优化的变量就不再是 $\widehat{w}$ 而是新变量 $v$，而后者是一个连续的变量，可用的优化方法更多。 $\tilde{w}$ 虽然也是带了量化误差的权重，但相比之前的 $\hat{w}$，这里面 round 部分只向下取整，通过 $h(v)$ 来调整最终取整的策略。 $h(v)$ 是一个数值范围在 $(0,1)$ 之间的函数，定义为：

$$h(v) = clip(\sigma(v)(\zeta - \gamma) + \gamma,0,1) \tag{18}$$

其中 $\sigma$ 是 sigmoid 函数, $h(v)$ 其实就是 sigmoid 函数的变体，它和 sigmoid 的函数图像如下：

![AdaRound取值函数](/images/Model_Accelaration/AdaRound取值函数.png)

$h(v)$ 和 sigmoid 的区别就是， $h(v)$ 在 $(0,1)$ 之间的部分梯度都比较抖，没有 sigmoid 那种平滑的走势。这种设计方式**可以避免 $h(v)$ 在需要优化的区域出现梯度弥散或消失的问题**。

再来看看 $f_{reg}$ 函数，图像如下图所示 (不同 $\beta$ 值对函数影响不同):

$$f_{reg}(v) = \sum_{i,j}(1 - |2h(v_{i,j}) - 1|^\beta) \tag{19}$$

![AdaRound取值函数2](/images/Model_Accelaration/AdaRound取值函数2.png)

现在分析如何最小化公式 (16) 的大小。对于 $||wx - \hat{w}x||^2_F$，$\hat{w}$ 中的 round 函数已经被松弛成连续的情况，但要保证 round 只能向上取整或者向下取整。由于 $h(v)$ 取值范围是 $(0,1)$， $\hat{w}$ 取值在区间 $[w^{floor}, w^{ceil}]$。要使 $\hat{w}$ 最终在 $w^{floor} w^{ceil}$ 中的一个，则需要观察 $f_{reg}$ 的图像。由上图可知， $f_{reg}$ 只有在 $h(v) = 0$ 或者 $h(v) =1$ 时，函数值才会等于 0。故只有最小化 $f_{reg}$，最后求解出来的 $h(v)$ 就是 0 或者 1， $\hat{w}$ 最后就会收敛到 $w^{floor} w^{ceil}$ 中的一个。

**$f_{reg}$ 在优化的过程中， $\beta$ 值需要从大到小慢慢调小**。在 $\beta$ 比较大的时候， $f_{reg}$ 有大片区域是比较平坦，该区域的梯度也比较小，即 $h(v)$ 最开始可以无限制向两边扩展，此时优化的主要是量化误差损失 $||wx - \hat{w}x||^2_F$。之后慢慢调小 $\beta$， $f_{reg}$ 平坦区域就越来越小，数值小于 0.5 的 $h(v)$ 就会往左边 0 的位置收敛，反之，则往右边 1 的位置靠拢。这种设计的好处是在最开始优化的时， $h(v)$ 可以自由地移动，不必拘泥于 $f_{reg}$ ，从而**把更多的优化空间留给量化误差损失。**

论文分析了优化前后 $h(v)$ 的变化。在优化前， $h(v)$ 是按照 nearest round 的方式初始化的，优化后 $h(v)$ 的情况发生了较大的变化。比如，如果是采用 nearest round，那么小于 0.5 的点 (横坐标)，应该都是要 round 到 0 的 (纵坐标)，而大于 0.5 的应该 round 到 1。但从图中发现，横坐标大于 0.5 的点，也有部分会被 round 到 0。这就说明，对于 AdaRound 来说，四舍五入未必是最优解。

![Adaround h(v)的变化](/images/Model_Accelaration/Adaround%20h(v)的变化.png)

#### 算法流程

AdaRound 是逐层顺序优化的，类似于 Bias Correction。首先，先对 $L-1$ 层进行优化。优化后把 $L-1$ 层的参数都用 AdaRound 求出来的最优解进行量化，并以此为基础对网络跑一遍前向推理直到 $L$ 层，然后继续用 AdaRound 优化，以此类推。这种逐层顺序优化的方法，**可以保证在优化 $L$ 层的时候，已经充分考虑了前面层的量化误差。**

此外，为了防止激活函数会方法量化误差，论文建议在优化的时候，把激活函数的影响也考虑进去：

$$\underset{v}{argmin} ||f_a(wx) - f_a(\tilde{w}x)||^2_F + \lambda f_{reg}(v) \tag{20}$$

即优化目标里面的量化损失部分，需要再包括激活函数后再计算

## 实验 Evaluation

论文使用 Adam 来优化作为 AdaRound 的优化器，以下是几个实验结果：

### 对比 Nearest rounding

论文针对 resnet18 这个通用的网络，在 ImageNet 上做了分类的实验。表格里的 First layer 表示只对第一层卷积做量化。首先，Nearest rounding 和其他方法相比差了 10 个点。其次，把 $H^{(w)}$ 扔掉的强假设，似乎并没有什么影响 (**Local MSE loss** v.s **$H^{(w)}$ task loss**)。第三，对目标进行松弛的方法也比直接暴力搜索的方式更有优势 (**Cont. relaxation** v.s. **Local MSE Loss**)，而且 **Cont. relaxation** 省时省力。

![Adaround优化算法对比](/images/Model_Accelaration/Adaround优化算法对比.png)

### 和直通估计器 (STE) 对比 Optimization using STE

其实 AdaRound 的目标函数 (公式 (16)) 是可以直接和 STE 结合进行量化训练的。因此论文也对比了用量化训练来优化该目标的效果。AdaRound 比 STE 优化的方式提高了两个点。论文解释说是量化训练的梯度带有一些 bias，从而影响了优化。

![Adaround和STE对比](/images/Model_Accelaration/Adaround和STE对比.png)

### 对训练数据的鲁棒性 Influence of quantization grid 

AdaRound 相比其他量化算法，需要进行小规模的训练，并且优化目标也更加复杂一些。因此研究数据样本对算法可能产生的影响，可以比较好地表现这个算法的稳定性。论文用了不同数据集的数据来优化 AdaRound，并对比了它们在 ImageNet 分类准确性上的效果。结果说明，用了跨数据集的数据并不会明显影响 AdaRound 的效果

![Adaround在不同数据集上的表现](/images/Model_Accelaration/Adaround在不同数据集上的表现.png)

## 结论 Conclusion

AdaRound 本身带来的提升也很大，几乎是离线量化的基准 baseline 。在之后的后量化训练算法中，不少是针对 AdaRound 进行改进优化的。因此搞懂这篇文章，对之后的学习也大有益处。

## 参考引用 Reference

### 论文地址 Papers

- [Up or Down? Adaptive Rounding for Post-Training Quantization (arxiv.org)](https://arxiv.org/abs/2004.10568)

### 博客 Blogs

- [AdaRound，一种全新的量化思路--问题篇 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/473815530)
- [AdaRound，一种全新的量化思路--解决篇 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/473820133)
- [AdaRound：一种自适应调整round的模型后量化算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/363941822)
- [AdaRound：训练后量化的自适应舍入-CSDN博客](https://blog.csdn.net/u012347027/article/details/114667698)