## 第1章 Python入门

### 1.5 NumPy

#### 1.5.1 导入NumPy

~~~python
import numpy as np
~~~

#### 1.5.2 生成NumPy数组

~~~python
>>> x = np.array([1.0, 2.0, 3.0])
>>> print(x)
[ 1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'>
~~~

#### 1.5.3 NumPy的算术运算

~~~python
>>> x = np.array([1.0, 2.0, 3.0])
>>> y = np.array([2.0, 4.0, 6.0])
>>> x + y  # 对应元素的加法
array([ 3.,  6., 9.])
>>> x - y
array([ -1.,  -2., -3.])
>>> x * y  # element-wise product
array([  2.,   8.,  18.])
>>> x / y
array([ 0.5,  0.5,  0.5])
~~~

这里需要注意的是，数组x和数组y的元素个数是相同的（两者均是元素个数为3的一维数组）。当x和y的元素个数相同时，可以对各个元素进行算术运算。如果元素个数不同，程序就会**报错**，所以**元素个数保持一致**非常重要。对应元素的”的英文是element-wise，比如“对应元素的乘法”就是element-wise product。

##### 广播

​		NumPy数组不仅可以进行element-wise运算，也可以和单一的数值（标量）组合起来进行运算。此时，需要**在NumPy数组的各个元素和标量之间进行运算**。这个功能也被称为广播。

~~~python
>>> x = np.array([1.0, 2.0, 3.0])
>>> x / 2.0
array([ 0.5,  1. ,  1.5])
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024091915583620240919155836.png" alt="image-20240919155829381" style="zoom:67%;" />

不同纬度的数组之间的乘法运算

~~~python
>>> A = np.array([[1, 2], [3, 4]])
>>> B = np.array([10, 20])
>>> A * B
array([[ 10, 40],
       [ 30, 80]])
~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024091915593520240919155935.png" alt="image-20240919155935598" style="zoom:67%;" />

#### 1.5.4 NumPy的N维数组

NumPy不仅可以生成一维数组（排成一列的数组），也可以生成多维数组。

~~~python
>>> A = np.array([[1, 2], [3, 4]])
>>> print(A)
[[1 2]
 [3 4]]
>>> A.shape
(2, 2)
>>> A.dtype
dtype('int64')
~~~

二维数组称为矩阵，矩阵的运算如下：

~~~python
>>> B = np.array([[1, 2],[3, 4]])
>>> B = np.array([[3, 0],[0, 6]])
>>> A + B
array([[ 4,  2],
       [ 3, 10]])
>>> A * B
array([[ 3,  0],
       [ 0, 24]])
>>> A * 10
array([[ 10, 20],
       [ 30, 40]])
~~~

数学上将一维数组称为**向量**，将二维数组称为**矩阵**。另外，可以将一般化之后的向量或矩阵等统称为**张量**(tensor)。

#### 1.5.6 访问元素

~~~python
>>> X = np.array([[51, 55], [14, 19], [0, 4]])
>>> print(X)
[[51 55]
 [14 19]
 [ 0 4]]
>>> X[0]    # 第0行
array([51, 55])
>>> X[0][1] # (0,1)的元素
55
>>> for row in X:
...     print(row)
...
[51 55]
[14 19]
[0 4]
~~~

使用数组访问各个元素

~~~python
>>> X = X.flatten()         # 将X转换为一维数组
>>> print(X)
[51 55 14 19  0  4]
>>> X[np.array([0, 2, 4])] # 获取索引为0、2、4的元素
array([51, 14,  0])
# 从X中抽出大于15的元素
>>> X > 15
array([ True,  True, False,  True, False, False], dtype=bool)
>>> X[X>15]
array([51, 55, 19])
~~~

Python等**动态类型语言**一般比C和C++等**静态类型语言**（编译型语言）运算速度慢。实际上，如果是运算量大的处理对象，用C/C++写程序更好。为此，当Python中追求性能时，人们会用C/C++来实现处理的内容。Python则承担“中间人”的角色，负责调用那些用C/ C++写的程序。**NumPy中，主要的处理也都是通过C或C++实现的**。因此，我们可以在不损失性能的情况下，使用Python便利的语法。

### 1.6 Matplotlib

在深度学习的实验中，图形的绘制和数据的可视化非常重要。Matplotlib是用于绘制图形的库，使用Matplotlib可以轻松地绘制图形和实现数据的可视化。

#### 1.6.1 绘制简单图形

~~~python
# 使用NumPy的arange方法生成了[0, 0.1, 0.2,…, 5.8, 5.9]的数据，将其设为x。对x的各个元素，应用NumPy的sin函数np.sin()，将x、y的数据传给plt.plot方法，然后绘制图形。最后，通过plt.show()显示图形。

import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()

~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024091916303020240919163030.png" alt="image-20240919163030367" style="zoom: 80%;" />

#### 1.6.2 pyplot的功能

~~~python
# 对1.6.1绘制的图形追加cos函数的图形，并使用pyplot的添加标题和x轴标签名等其他功能。
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.arange(0, 6, 0.1) # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # 用虚线绘制
plt.xlabel("x") # x轴标签
plt.ylabel("y") # y轴标签
plt.title('sin & cos') # 标题
plt.legend()  # 添加图形示例
plt.show()

~~~

![image-20240919165258765](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024091916525820240919165258.png)

#### 1.6.3 显示图像

~~~python
# 使用matplotlib.image模块的imread()方法读入图像，使用imshow()方法显示图像

import matplotlib.pyplot as plt
from matplotlib.image import imread
img = imread('practiceOfNumpy.png') # 读入图像（设定合适的路径！）
plt.imshow(img)

plt.show()

~~~

![image-20240919170139447](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024091917013920240919170139.png)

## 第2章 感知机

### 2.1 感知机是什么

感知机接收多个输入信号，输出一个信号。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092007523720240920075237.png" alt="image-20240920075230561" style="zoom:67%;" />

上图是一个接收两个输入信号的感知机的例子。x1、x2是输入信号，y是输出信号，w1、w2是权重（w是weight的首字母）。图中的○称为“神经元”或者“节点”。输入信号被送往神经元时，会被分别乘以固定的权重(w1x1、w2x2)。神经元会计算传送过来的信号的总和，只有当这个总和超过了某个界限值时，才会输出1。这也称为“神经元被激活”。这里将这个界限值称为阈值，用符号θ表示。

把上述内容用数学式来表示，就是式(2.1)

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092007571420240920075714.png" alt="image-20240920075714603" style="zoom:80%;" />

### 2.2 简单逻辑电路

#### 2.2.1 与门

与门（AND gate）是有两个输入和一个输出的门电路。与门仅在两个输入均为1时输出1，其他时候则输出0。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092008022320240920080223.png" alt="image-20240920080223368" style="zoom:80%;" />

​                                                                                                  **与门真值表**

#### 2.2.2 与非门和或门

与非门（NAND gate，NAND是Not AND的意思）就是颠倒了与门的输出。用真值表表示的话，仅当x1和x2同时为1时输出0。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092008064520240920080645.png" alt="image-20240920080645712" style="zoom:80%;" />

​                                                                                                 **与非门真值表**

或门是只要有一个输入信号是1，输出就为1的逻辑电路。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092008085220240920080852.png" alt="image-20240920080852426" style="zoom:80%;" />

​                                                                                                   **或门真值表**

### 2.3 感知机的实现

#### 2.3.1 简单的实现

~~~python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp =x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

'''
AND(0, 0) # 输出0
AND(1, 0) # 输出0
AND(0, 1) # 输出0
AND(1, 1) # 输出1
'''
~~~

#### 2.3.2 导入权重和偏置

把式2.1中的θ换成-b，b称为偏置，w1和w2称为权重。感知机会计算输入信号和权重的乘积，然后加上偏置，如果这个值大于0则输出1，否则输出0。

![image-20240920103242158](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015114520240920151145.png)

~~~python
>>> import numpy as np
>>> x = np.array([0, 1])     # 输入
>>> w = np.array([0.5, 0.5]) # 权重
>>> b = -0.7                 # 偏置
>>> w*x
array([ 0. , 0.5])
>>> np.sum(w*x)
0.5
>>> np.sum(w*x) + b
-0.19999999999999996   # 大约为-0.2（由浮点小数造成的运算误差）
~~~

#### 2.3.3 使用权重和偏置的实现

~~~python
# 使用权重和偏置实现与门
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

~~~

w1和w2是**控制输入信号的重要性**的参数，而偏置b是调整神经元**被激活的容易程度**（输出信号为1的程度）的参数。

~~~python
# 实现与非门和或门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
~~~

### 2.4 感知机的局限性

#### 2.4.1 异或门

使用感知机可以实现与门、与非门、或门三种逻辑电路。

异或门（XOR gate）也被称为逻辑异或电路。仅当x1或x2中的一方为1时，才会输出1。

![image-20240920110821625](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015124020240920151240.png)

感知机的局限性在于它只能表示由一条直线分割的空间。

![image-20240920143927397](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015123520240920151235.png)

![image-20240920144012598](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015123320240920151233.png)

由曲线分割而成的空间称为**非线性空间**，由直线分割而成的空间称为**线性空间**。

### 2.5 多层感知机

#### 2.5.1 已有门电路的组合

上节中，严格来说，应该是**单层感知机无法表示异或门**或者**单层感知机无法分离非线性空间**。

![image-20240920150243231](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015122920240920151229.png)

![image-20240920150306012](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015123220240920151232.png)

如上图通过组合与门、与非门、或门可以实现异或门

![image-20240920150346519](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015122420240920151224.png)

​                                                                      **异或门的真值表**

#### 2.5.2 异或门的实现

~~~python
# 结合前面定义的函数，定义新函数，实现异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
~~~

~~~python
XOR(0, 0) # 输出0
XOR(1, 0) # 输出1
XOR(0, 1) # 输出1
XOR(1, 1) # 输出0
~~~

使用感知机的方法表示异或门，如下图：

![image-20240920151420018](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092015142020240920151420.png)

与门、或门是**单层感知机**，而异或门是**2层感知机**。叠加了多层的感知机也称为**多层感知机**(multi-layered perceptron)。上图中的感知机总共由3层构成，但是因为拥有权重的层实质上只有2层（第0层和第1层之间，第1层和第2层之间），所以称为“2层感知机”。不过，有的文献认为图中的感知机是由3层构成的，因而将其称为“3层感知机”。

### 2.6 从与非门到计算机

