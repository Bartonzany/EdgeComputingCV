---
aliases:
  - 矩阵消元
title: 第2讲 矩阵消元 Elimination with matrices
date: 2025-06-14 17:21:38
excerpt: 线性代数
tags:
  - 线性代数
---

## 第2讲 - 矩阵消元 Elimination with matrices

### 1. 知识概要

本节首先介绍消元法的初等变换实现，然后进一步介绍向量与矩阵的乘法，在此基础上，研究消元法的矩阵实现（消元矩阵）。最后简单地引入置换矩阵与逆矩阵。

具体内容如下：

- **Elimination（消元法）**
- **Back-Substitution （回代）**
- **Elimination Matrics（消元矩阵）**
- **Matrix Multiplication（矩阵乘法）**

事实上，计算机语言实现方程的求解，就是通过 **消元法** 实现的。

### 2. 消元法

消元法，也叫**高斯消元法(Gauss elimination)**，是计算机软件求解线形方程组所用的最常见的方法。**任何情况下，只要是矩阵 A 可逆，均可以通过消元法求得 $Ax=b$ 的解**。

消元法的核心思想是通过对方程组中的某两个方程进行适当的**数乘**和**加和**，以达到将某一未知数系数变为零，从而达到削减未知数个数的目的。

例如，二元一次方程组：

$$
\left\{
\begin{aligned}
x + 2y + z &= 2, \\
3x + 8y + z &= 12, \\
4y + z &= 2.
\end{aligned}
\right.
$$

其矩阵形式为：

$$
A = \begin{bmatrix} 
1 & 2 & 1 \\ 
3 & 8 & 1 \\ 
0 & 4 & 1 
\end{bmatrix}, \quad 
\mathbf{b} = \begin{bmatrix} 
2 \\ 
12 \\ 
2 
\end{bmatrix}.
$$

左上角的数字1叫做 **第一个主元(First pivot)** ，第一步要通过消元将第一列中除了主元之外的数字均变化为 0。

- 第一行乘 **-3** 加到第二行，将第二行第一列变化为 **0**
- 第三行第一个也变化为 **0**

$$
A = \begin{bmatrix} 
\boxed{1} & 2 & 1 \\ 
3 & 8 & 1 \\ 
0 & 4 & 1 
\end{bmatrix}
\xrightarrow{(2,1)}
\begin{bmatrix} 
\boxed{1} & 2 & 1 \\ 
0 & \boxed{2} & -2 \\ 
0 & 4 & 1 
\end{bmatrix}
\xrightarrow{(3,2)}
U = \begin{bmatrix} 
\boxed{1} & 2 & 1 \\ 
0 & \boxed{2} & -2 \\ 
0 & 0 & \boxed{5} 
\end{bmatrix}
$$

由于 _A_ 矩阵为可逆矩阵，消元结束后得到 **上三角阵 U（Uppertriangular matrix）**，其左侧下半部分的元素均为 0，而主元 1,2,5 分列在 _U_ 的对角线上，至此，消元结束，得到的 _U_ 即为我们想要化简的形式。

同样的，b 最后变成：

$$
\mathbf{b} = \begin{bmatrix} 
2 \\ 
6 \\ 
3
\end{bmatrix}.
$$

需要说明的是，**主元不能为0**。如果恰好消元至某行，0出现在了主元的位置上，应当通过与下方一行进行**行交换**，使得非零数字出现在主元位置上。

如果0出现在了主元位置上，并且下方没**有对等位置为非0数字的行**，则消元终止，并证明矩阵A为不可逆矩阵，且线性方程组**没有唯一解**。以下是三个例子：

$$
\begin{bmatrix} 
\boxed{0} & 2 & -2 \\ 
0 & \boxed{2} & 5 \\ 
0 & 0 & \boxed{5} 
\end{bmatrix},
\quad
\begin{bmatrix} 
\boxed{1} & 2 & -2 \\ 
0 & \boxed{0} & 5 \\ 
0 & 0 & \boxed{5} 
\end{bmatrix},
\quad
\begin{bmatrix} 
\boxed{1} & 2 & -2 \\ 
0 & \boxed{2} & 5 \\ 
0 & 0 & \boxed{0} 
\end{bmatrix}
$$

### 3. 回代求解

回带求解应该和消元法同时进行，需要对等式右侧的 **b** 做同样的乘法和加减法。手工计算时比较有效率的方法是使用 **增广矩阵（augmented matrix）**，将 **b** 插入矩阵 _A_ 之后形成最后一列，在消元过程中带着 **b** 一起操作。

$$
\begin{bmatrix} 
1 & 2 & 1 & \textcolor{blue}{2} \\ 
3 & 8 & 1 & \textcolor{blue}{12} \\ 
0 & 4 & 1 & \textcolor{blue}{2} 
\end{bmatrix}
\xrightarrow{(2,1)}
\begin{bmatrix} 
\textcolor{red}{1} & 2 & 1 & \textcolor{blue}{2} \\ 
0 & \textcolor{red}{2} & -2 & \textcolor{blue}{6} \\ 
0 & 4 & 1 & \textcolor{blue}{2} 
\end{bmatrix}
\xrightarrow{(3,2)}
\begin{bmatrix} 
\textcolor{red}{1} & 2 & 1 & \textcolor{blue}{2} \\ 
0 & \textcolor{red}{2} & -2 & \textcolor{blue}{6} \\ 
0 & 0 & \textcolor{red}{5} & \textcolor{blue}{-10} 
\end{bmatrix}.
$$

此时，便将原方程 $Ax=b$ 转化为了新的方程 $Ux=c$，其中 $\mathbf{b} = \begin{bmatrix} 2 \\ 6 \\ 3\end{bmatrix}.$

### 4. 消元矩阵

矩阵运算的核心内容就是对 **行** 或者 **列** 进行独立操作。

如前一节 **列图像** 部分，系数矩阵乘以列向量，相当于**对系数矩阵的列向量进行线性组合**。

$$
\begin{bmatrix} 
\textcolor{red}{*} & \textcolor{blue}{*} & \textcolor{green}{*} \\ 
\textcolor{red}{*} & \textcolor{blue}{*} & \textcolor{green}{*} \\ 
\textcolor{red}{*} & \textcolor{blue}{*} & \textcolor{green}{*}
\end{bmatrix}
\begin{bmatrix} 
3 \\ 
4 \\ 
5 
\end{bmatrix}
=
3 \begin{bmatrix} 
\textcolor{red}{*} \\ 
\textcolor{red}{*} \\ 
\textcolor{red}{*}
\end{bmatrix}
+
4 \begin{bmatrix} 
\textcolor{blue}{*} \\ 
\textcolor{blue}{*} \\ 
\textcolor{blue}{*}
\end{bmatrix}
+
5 \begin{bmatrix} 
\textcolor{green}{*} \\ 
\textcolor{green}{*} \\ 
\textcolor{green}{*} 
\end{bmatrix}
=
\begin{bmatrix} 
\otimes \\ 
\otimes \\ 
\otimes 
\end{bmatrix}.
$$

相应地，矩阵左乘行向量则是**对矩阵的行向量进行线性组合**。

$$
\begin{bmatrix} 
1 & 2 & 7 
\end{bmatrix}
\begin{bmatrix} 
\textcolor{red}{*} & \textcolor{red}{*} & \textcolor{red}{*} \\ 
\textcolor{blue}{*} & \textcolor{blue}{*} & \textcolor{blue}{*} \\ 
\textcolor{green}{*} & \textcolor{green}{*} & \textcolor{green}{*} 
\end{bmatrix}=
\begin{bmatrix} 
\textcolor{red}{*} & \textcolor{red}{*} & \textcolor{red}{*}
\end{bmatrix} + 2
\begin{bmatrix} 
\textcolor{blue}{*} & \textcolor{blue}{*} & \textcolor{blue}{*}
\end{bmatrix} + 7
\begin{bmatrix} 
\textcolor{green}{*} & \textcolor{green}{*} & \textcolor{green}{*} 
\end{bmatrix} =
\begin{bmatrix} 
\otimes & \otimes & \otimes 
\end{bmatrix}.
$$

矩阵消元的第一步是通过**左乘矩阵** $E_{21}$ 实现原矩阵 _A_ 的第二行减去第一行 3 倍这一过程。

$$
\underbrace{
\begin{bmatrix}
1 & 0 & 0 \\
-3 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}}_{E_{21}}
\underbrace{
\begin{bmatrix}
1 & 2 & 1 \\
3 & 8 & 1 \\
0 & 4 & 1
\end{bmatrix}}_{A}
=
\underbrace{
\begin{bmatrix}
1 & 2 & 1 \\
0 & 2 & -2 \\
0 & 4 & 1
\end{bmatrix}}_{E_{21}A}.
$$

$E_{21}$ 第二行使矩阵 _A_ 的行向量进行线性组合，而其它两行为了保持与原矩阵相同，采用同阶单位阵 _I_ 的行向量。左乘的这个矩阵为 **初等矩阵 （Elementary Matrix）**，因此记做 _E_。

- **第一行**：

$$
\begin{bmatrix} 
1 & 0 & 0 
\end{bmatrix}
\begin{bmatrix} 
1 & 2 & 1 \\ 
3 & 8 & 1 \\ 
0 & 4 & 1 
\end{bmatrix}
=
\begin{bmatrix} 
1 & 2 & 1 
\end{bmatrix}. 
$$
- **第三行**：

$$
\begin{bmatrix} 
0 & 0 & 1 
\end{bmatrix}
\begin{bmatrix} 
1 & 2 & 1 \\ 
3 & 8 & 1 \\ 
0 & 4 & 1 
\end{bmatrix}
=
\begin{bmatrix} 
0 & 4 & 1 
\end{bmatrix}.
$$
- **关键第二行**：

$$
\begin{aligned}
\begin{bmatrix} 
-3 & 1 & 0 
\end{bmatrix}
\begin{bmatrix} 
1 & 2 & 1 \\ 
3 & 8 & 1 \\ 
0 & 4 & 1 
\end{bmatrix} 
&= (-3) \begin{bmatrix} 1 & 2 & 1 \end{bmatrix}
+ (1) \begin{bmatrix} 3 & 8 & 1 \end{bmatrix}
+ (0) \begin{bmatrix} 0 & 4 & 1 \end{bmatrix} \\
&= \begin{bmatrix} 0 & 2 & -2 \end{bmatrix}.
\end{aligned}
$$

矩阵消元的第二步是矩阵 $E_{21}A$ 的第三行减去第二行的 2 倍，通过左乘矩阵 $E_{32}$ 实现。

$$
\underbrace{
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -2 & 1
\end{bmatrix}}_{E_{32}}
\underbrace{
\begin{bmatrix}
1 & 2 & 1 \\
0 & 2 & -2 \\
0 & 4 & 1
\end{bmatrix}}_{E_{21}A}
=
\underbrace{
\begin{bmatrix}
1 & 2 & 1 \\
0 & 2 & -2 \\
0 & 0 & 5
\end{bmatrix}}_{E_{32}(E_{21}A)}.
$$

$3 \times 3$ 矩阵最终得到 $E_{32}(E_{21}A)= U$。矩阵运算符合结合律，也可写作 $(E_{32}E_{21})A=U$，记为 $EA=U$。

方程 $Ax=b$ 的解也满足方程 $Ux=EAx=Eb=c$，因此我们将问题转化为 $Ux=c$。

### 5. 置换矩阵

**左乘**置换矩阵可以完成原矩阵的 **行变换**，**右乘**置换矩阵则为**列变换**

$$
\begin{bmatrix} 
0 & 1 \\ 
1 & 0 
\end{bmatrix}
\begin{bmatrix} 
a & b \\ 
c & d 
\end{bmatrix}
=
\begin{bmatrix} 
c & d \\ 
a & b 
\end{bmatrix},
\quad
\begin{bmatrix} 
a & b \\ 
c & d 
\end{bmatrix}
\begin{bmatrix} 
0 & 1 \\ 
1 & 0 
\end{bmatrix}
=
\begin{bmatrix} 
b & a \\ 
d & c 
\end{bmatrix}.
$$

构造 _P_ 矩阵是通过对 _I_ 矩阵进行行交换实现的，左右乘效果不同也展示了矩阵运算不符合交换律的性质

### 6. 逆矩阵

消元矩阵实施效果就是**抵消**原矩阵的消元操作。如原矩阵 _A_ 第二行行向量 $[3,8,1]$ 减掉了第一行 $[1,2,1]$ 的3倍变为 $[0,2,-2]$，则逆向操作为将第二行行向量 $[0,2,-2]$ 加上第一行 $[1,2,1]$ 的3倍，从而变回原来的第二行 $[3,8,1]$。

$$
E_{21} = 
\begin{bmatrix}
1 & 0 & 0 \\
-3 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix},
E_{21}^{-1} = 
\begin{bmatrix}
1 & 0 & 0 \\
3 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}.
$$

$$
E_{21}^{-1} E_{21} =
\begin{bmatrix}
1 & 0 & 0 \\
3 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 \\
-3 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}=I.
$$

### 7. 小结

本节从矩阵消元的角度，介绍解方程的通用做法，并介绍了消元矩阵，从矩阵乘法层面理解消元的过程，并延伸了消元矩阵的应用：基于单位阵 _I_ 的变化，对矩阵 A 进行行列变换的过程。

---