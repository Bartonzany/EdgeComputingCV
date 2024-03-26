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

推导出这个公式的目的是想确认 round 带来的误差 $w$ 对 $L$ 的影响，即 $\Delta w$ 的变化会给公式 (7) 的结果带来什么变化。对于损失函数 $L$ 来说，当模型训练收敛的时候，导数一般是趋于 0 的 (训练的时候通常是用一阶导数)，即 

## 研究目的 Research Objective

## 方法 Methods

## 实验 Evaluation

## 结论 Conclusion

## 参考引用 Reference

