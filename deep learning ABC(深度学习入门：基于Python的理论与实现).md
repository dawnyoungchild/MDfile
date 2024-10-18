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

感知机通过叠加层能够进行非线性的表示，理论上还可以表示计算机进行的处理。

## 第3章 神经网络

### 3.1 从感知机到神经网络

#### 3.1.1 神经网络的例子

神经网络和感知机有很多共同点。

用图来表示神经网络的话，如下图所示。最左边的一列称为**输入层**，最右边的一列称为**输出层**，中间的一列称为**中间层**。中间层有时也称为**隐藏层**。“隐藏”一词的意思是，隐藏层的神经元（和输入层、输出层不同）肉眼看不见。输入层到输出层也依次称为第0层、第1层、第2层（层号之所以从0开始，是为了方便后面基于Python进行实现）。下图中，第0层对应输入层，第1层对应中间层，第2层对应输出层。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092311085420240923110854.png" alt="image-20240923110832935" style="zoom:80%;" />

上图网络一共由3层神经元构成，但实质上只有2层神经元有权重，因此称其为2层网络。有的术会根据构成网络的层数，称其为3层网络。

#### 3.1.2 复习感知机

![image-20240923170510525](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092317051020240923170510.png)

式3.1中，b为偏置参数，

在上图中，b没有明确画出来。可以如下图这样明确表示b。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092317081520240923170815.png" alt="image-20240923170815841" style="zoom:80%;" />

进而将式子3.1改写成式3.2

y = h(b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>)

和

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092317100020240923171000.png" alt="image-20240923171000495" style="zoom: 50%;" />

#### 3.1.3 激活函数登场

h(x)函数会将输入信号的总和转换为输出信号，这种函数一般称为**激活函数**(activation function)。如“激活”一词所示，激活函数的作用在于**决定如何来激活输入信号的总和**。

将式（3.2）y = h(b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>)分解为式3.4和3.5：

a = b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub>		（3.4）

y = h(a)							(3.5)

首先，式(3.4)计算加权输入信号和偏置的总和，记为a。然后，式(3.5)用h()函数将a转换为输出y。

通过下图可以明确式3.4和3.5的关系

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092317302420240923173024.png" alt="image-20240923173024608" style="zoom: 67%;" />

### 3.2 激活函数

式<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092317100020240923171000.png" alt="image-20240923171000495" style="zoom: 50%;" />表示的激活函数以阈值为界，一旦输入超过阈值，就切换输出。这样的函数称为“**阶跃函数**”。因此，可以说感知机中使用了阶跃函数作为激活函数。也就是说，在激活函数的众多候选函数中，感知机使用了阶跃函数。那么，如果感知机使用其他函数作为激活函数的话会怎么样呢？实际上，如果将激活函数从阶跃函数换成其他函数，就可以进入神经网络的世界了。下面我们就来介绍一下神经网络使用的激活函数。

#### 3.2.1 sigmoid函数

sigmoid函数（式3.6）是神经网络中经常使用的一个激活函数。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092317391920240923173919.png" alt="image-20240923173919266" style="zoom: 80%;" />

exp(-x)表示e<sup>-x</sup>次方的意思。e是皮纳尔常数2.7182...

神经网络中用sigmoid函数作为激活函数，进行信号的转换，转换后的信号被传送给下一个神经元。

#### 3.2.2 阶跃函数的实现

~~~python
# 使用Python表示式3.3，输入超过0时，输出1，否则输出0
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

~~~

以上函数只能接收实数而不能接收数组作为参数，对函数做一下修改让它支持NumPy数组的实现。

~~~python
import numpy as np
def step_function(x):
    y = x > 0
    return y.astype(int)

~~~

对上述函数的分解

~~~python
>>> import numpy as np
>>> x = np.array([-1.0, 1.0, 2.0])
>>> x
array([-1.,  1.,  2.])
>>> y = x > 0	# 数组x中大于0的元素被转换为True，小于等于0的元素被转换为False，从而生成一个新的数组y。
>>> y
array([False,  True,  True], dtype=bool)
'''
用astype()方法转换NumPy数组的类型。astype()方法通过参数指定期望的类型，这个例子中是np.int型。Python中将布尔型转换为int型后，True会转换为1，False会转换为0。
'''
>>> y = y.astype(int)
>>> y
array([0, 1, 1])
~~~

#### 3.2.3 阶跃函数的图形

~~~python
# 画出上面定义的阶跃函数的图形
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)   # 生成从-5.0到5.0，以0.1为阶长的数组
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()

~~~

![image-20240924144410631](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092414441720240924144417.png)

阶跃函数以0为界，输出从0切换为1（或者从1切换为0）。它的值呈阶梯式变化，所以称为阶跃函数。

#### 3.2.4 sigmoid函数的实现

~~~python
# 使用python定义sigmoid函数

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
~~~

~~~python
# 把sigmoid函数画在图上
import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)   # 生成从-5.0到5.0，以0.1为阶长的数组
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()

~~~

![image-20240924150155420](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092415015520240924150155.png)

#### 3.2.5 sigmoid函数和阶跃函数的比较

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092415024420240924150244.png" alt="image-20240924150244203" style="zoom:67%;" />

1、sigmoid函数输出是平滑的曲线，输出随着输入发生连续性变化；阶跃函数以0为界，输出发生急剧性变化。

2、感知机中神经元之间流动的是0或1的二元信号，而神经网络中流动的是连续的实数值信号。

3、当输入信号为重要信息时，阶跃函数和sigmoid函数都会输出较大的值；当输入信号为不重要的信息时，两者都输出较小的值。还有一个共同点是，不管输入信号有多小，或者有多大，输出信号的值都在0到1之间。

#### 3.2.6 非线性函数

函数本来是输入某个值后会返回一个值的转换器。向这个转换器输入某个值后，**输出值是输入值的常数倍的函数称为线性函数**（用数学式表示为h(x) = cx。c为常数）。因此，线性函数是一条笔直的直线。而非线性函数，顾名思义，指的是不像线性函数那样呈现出一条直线的函数。

神经网络的激活函数**必须使用非线性函数**。

#### 3.2.7 ReLU函数

ReLU函数在输入大于0时，直接输出该值；在输入小于等于0时，输出0。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092417013020240924170130.png" alt="image-20240924163310644" style="zoom:67%;" />

ReLU函数可以用下式表示

![image-20240924165128063](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092417012620240924170126.png)

~~~python
# python实现ReLU函数
def relu(x):
    return np.maximum(0, x)		# maximum函数会从输入的数值中选择较大的那个值进行输出。

~~~

### 3.3 多维数组的运算

#### 3.3.1 多维数组

~~~python
# A、B分别为一维数组和二维数组
>>> import numpy as np
>>> A = np.array([1, 2, 3, 4])
>>> print(A)
[1 2 3 4]
# 获取数组维数
>>> np.ndim(A)
1
# 获取数组元素个数，返回元组
>>> A.shape
(4,)
>>> A.shape[0]
4

>>> B = np.array([[1,2], [3,4], [5,6]])
>>> print(B)
[[1 2]
 [3 4]
 [5 6]]
>>> np.ndim(B)
2
>>> B.shape
(3, 2)
~~~

数组B是3*2的数组，第一维度有3个元素，第二维度有2个元素。第一个维度对应第0维，第二个维度对应第1维（Python的索引从0开始）。二维数组也称为矩阵(matrix)。如下图所示，数组的横向排列称为行(row)，纵向排列称为列(column)。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092417540020240924175400.png" alt="image-20240924175400669" style="zoom:67%;" />

#### 3.3.2 矩阵乘法

![image-20240924175726138](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092417572620240924175726.png)

~~~python
# python实现2*2矩阵相乘
>>> A = np.array([[1,2],[3,4]])
>>> B = np.array([[5,6],[7,8]])
>>> np.dot(A,B)
array([[19, 22],
       [43, 50]])
>>> np.dot(B,A)
array([[23, 34],
       [31, 46]])

~~~

要注意的是，**np.dot(A, B)和np.dot(B, A)的值可能不一样**。和一般的运算（+或*等）不同，矩阵的乘积运算中，操作数(A、B)的顺序不同，结果也会不同。

计算矩阵A和B的乘积，**A的第1维元素个数（列数）和B的第0维元素个数（行数）必须相等**，否则运算会出现错误。

~~~python
>>> A = np.array([[1,2,3],[4,5,6]])
>>> A.shape
(2, 3)
>>> B = np.array([[7,8],[9,10]])
>>> B.shape
(2, 2)
>>> np.dot(A,B)
Traceback (most recent call last):	# A列为3，B行为2，所以运算AB乘积出现错误
  File "<pyshell#9>", line 1, in <module>
    np.dot(A,B)
  File "<__array_function__ internals>", line 200, in dot
ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
>>> np.dot(B,A)	 # B列为2，A行为2，所以可以运算BA乘积
array([[39, 54, 69],
       [49, 68, 87]])

~~~

![image-20240927104047423](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092710404720240927104047.png)

由上图可以看出，A矩阵1维（列）与B矩阵0维（行）必须相等，而且运算结果**矩阵C的形状由A的0维（行）和B的1维（列）构成**。

![image-20240927104021962](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092710402220240927104022.png)

当A是二维矩阵，B是一维数组时，对应维度的元素个数保持一致的原则依然成立。

#### 3.3.3 神经网络的内积

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092714453020240927144530.png" alt="image-20240927144530502" style="zoom: 80%;" />

实现该神经网络时，要注意X、W、Y的形状，特别是X和W的对应维度的元素个数是否一致，这一点很重要。

~~~python
>>> X = np.array([1, 2])
>>> X.shape
(2,)
>>> W = np.array([[1, 3, 5], [2, 4, 6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape
(2, 3)
>>> Y = np.dot(X, W)
>>> print(Y)
[ 5  11  17]
~~~

使用np.dot（多维数组的点积），可以一次性计算出Y的结果。这意味着，即便[插图]的元素个数为100或1000，也可以通过一次运算就计算出结果。

### 3.4 3层神经网络的实现

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092714595520240927145955.png" alt="image-20240927145955376" style="zoom:80%;" />

图3-15 3层神经网络：输入层（第0层）有2个神经元，第1个隐藏层（第1层）有3个神经元，第2个隐藏层（第2层）有2个神经元，输出层（第3层）有2个神经元

#### 3.4.1 符号确认

![image-20240927160755486](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092716075520240927160755.png)

右上角（1）：表示权重和神经元的层号（即第1层的权重、第1层的神经元）

右下角1：后1层的神经元索引号

右下角2：前1层的神经元索引号

所以上图符号表示前一层的第2个神经元到后一层的第1个神经元的权重。权重右下角按照“**后一层的索引号、前一层的索引号**”的顺序排列。

#### 3.4.2 各层间信号传递的实现

![image-20240927163715104](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092716371520240927163715.png)

图3-17中增加了表示偏置的神经元“1”。请注意，偏置的右下角的索引号只有一个。这是因为前一层的偏置神经元（神经元“1”）只有一个。

a<sub>1</sub><sup>(1)</sup>通过加权信号和偏置的和按如下方式进行计算。

![image-20240927164247697](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092716424720240927164247.png)

如果使用矩阵的乘法运算，则可以将第1层的加权和表示成下面的式（3.9）。

![image-20240927165159186](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092716515920240927165159.png)

其中，A<sup>(1)</sup>、X、B<sup>(1)</sup>、W<sup>(1)</sup>如下所示

![image-20240927165526105](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024092716552620240927165526.png)

~~~python
# 使用NumPy实现式（3.9），将输入信号、权重、偏置值设置为任意值
import numpy as np

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape)
print(X.shape)
print(B1.shape)
A1 = np.dot(X, W1) + B1
print(A1)

'''
运行结果：
(2, 3)
(2,)
(3,)
[0.3 0.7 1.1]
'''
~~~

![image-20240930075849260](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093007585620240930075856.png)

上图3-18为**从输入层到第1层的信号传递**

a表示隐藏层的加权和（加权信号和偏置的总和）

z表示激活函数转换后的信号

h()表示激活函数，此处用sigmoid函数，使用Python实现如下

~~~python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
print(Z1)

'''
[0.57444252 0.66818777 0.75026011]
'''
~~~

![image-20240930080432843](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093008043220240930080432.png)

上图3-19为**第1层到第2层的信号传递**

~~~python
# Python实现第1层到第2层的传递，参数仍设置为任意值
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2,)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)	# [0.62624937 0.7710107 ]
~~~

这一传递的实现跟上一个传递的实现基本一样

![image-20240930081401085](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093008140120240930081401.png)

上图3-20为**第2层到输出层的信号传递**

输出层的实现也和之前的实现基本相同，但是**最后的激活函数（σ()）与之前的隐藏层（h()）不同**。

~~~python
'''
定义输出层的激活函数identity_function()函数（也称为“恒等函数”），恒等函数会将输入按原样输出。
'''
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # 或者Y = A3
~~~

**输出层所用的激活函数，要根据求解问题的性质决定。**一般地，回归问题可以使用恒等函数，二元分类问题可以使用sigmoid函数，多元分类问题可以使用softmax函数。

#### 3.4.3 代码实现小结

按照神经网络的实现惯例，权重记为大写字母W1，其他的（偏置或中间结果等）都用小写字母表示。

~~~python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708 0.69627909]

~~~

### 3.5 输出层的设计

机器学习的问题大致可以分为**分类问题**和**回归问题**。分类问题是数据属于哪一个类别的问题。比如，区分图像中的人是男性还是女性的问题就是分类问题。而回归问题是根据某个输入预测一个（连续的）数值的问题。比如，根据一个人的图像预测这个人的体重的问题就是回归问题（类似“57.4kg”这样的预测）。

神经网络可以用在分类问题和回归问题上，不过需要根据情况改变输出层的激活函数。一般而言，**回归问题用恒等函数，分类问题用softmax函数**。

#### 3.5.1 恒等函数和softmax函数

恒等函数会将输入按原样输出，对于输入的信息，不加以任何改动地直接输出。

softmax函数表达式

![image-20240930103118564](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093010311820240930103118.png)

其中，exp(x)是表示e<sup>x</sup>的指数函数（e是纳皮尔常数2.7182 ...）

式(3.10)表示假设输出层共有n个神经元，计算第k个神经元的输出y<sub>k</sub>。如式(3.10)所示，softmax函数的分子是输入信号[插图]的指数函数，分母是所有输入信号的指数函数的和。

用图表示softmax函数如下（3-22）：

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093010501220240930105012.png" alt="image-20240930105012779" style="zoom:67%;" />

softmax函数的输出通过箭头与所有的输入信号相连。这是因为，从式(3.10)可以看出，**输出层的各个神经元都受到所有输入信号的影响**。

~~~python
# 解释器确认softmax函数
>>> a = np.array([0.3, 2.9, 4.0])
>>>
>>> exp_a = np.exp(a) # 指数函数
>>> print(exp_a)
[  1.34985881  18.17414537  54.59815003]
>>>
>>> sum_exp_a = np.sum(exp_a) # 指数函数的和
>>> print(sum_exp_a)
74.1221542102
>>>
>>> y = exp_a / sum_exp_a
>>> print(y)
[ 0.01821127  0.24519181  0.73659691]

~~~

~~~python
# 定义softmax函数
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

~~~

#### 3.5.2 实现softmax函数时的注意事项

溢出问题：softmax函数的实现中要进行指数函数的运算，但是此时指数函数的值很容易变得非常大。e<sup>100</sup>的值超过40位，而e<sup>1000</sup>直接返回表示无穷大的inf。

计算机处理“数”时，数值必须在**4字节或8字节**的有限数据宽度内。这意味着数存在**有效位数**，也就是说，可以表示的数值范围是有限的。因此，会出现超大值无法表示的问题。这个问题称为**溢出**，在进行计算机的运算时必须注意。

对softmax函数的改进（3.11）：

![image-20240930110725469](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093011072520240930110725.png)

首先，式(3.11)在分子和分母上都乘上C这个任意的常数（因为同时对分母和分子乘以相同的常数，所以计算结果不变）。然后，把这个C移动到指数函数(exp)中，记为log C。最后，把log C替换为另一个符号C'。

式(3.11)说明，在进行softmax的指数函数的运算时，**加上（或者减去）某个常数并不会改变运算的结果**。这里的C'可以使用任何值，但是为了防止溢出，一般会**使用输入信号中的最大值**。

~~~python
>>> a = np.array([1010, 1000, 990])
>>> np.exp(a) / np.sum(np.exp(a)) # softmax函数的运算
array([ nan,  nan,  nan])         # 没有被正确计算
>>>
>>> c = np.max(a) # 1010
>>> a - c
array([  0, -10, -20])
>>>
>>> np.exp(a - c) / np.sum(np.exp(a - c))
array([  9.99954600e-01,   4.53978686e-05,   2.06106005e-09])
'''
通过减去输入信号中的最大值(上面的c)，原本不能正常计算的地方（nan）可以正确计算了。
'''
~~~

~~~python
# 由上面解释器的结果，我们可以优化softtmax函数
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

~~~

#### 3.5.3 softmax函数的特征

~~~python
>>> a = np.array([0.3, 2.9, 4.0])
>>> y = softmax(a)	# 上节定义的softmax函数
>>> print(y)
[ 0.01821127  0.24519181  0.73659691]
>>> np.sum(y)
1.0

~~~

由上面的代码可见：

1. softmax函数的输出是0.0到1.0之间的实数
2. softmax函数的输出值的总和是1

输出总和为1是softmax函数的一个重要性质。正因为有了这个性质，我们才可以把softmax函数的输出解释为“**概率**”。

上面的例子可以解释成y[0]的概率是0.018(1.8%)，y[1]的概率是0.245(24.5%)，y[2]的概率是0.737(73.7%)。从概率的结果来看，可以说“因为第2个元素的概率最高，所以答案是第2个类别”。或者说上面的例子可以解释成y[0]的概率是0.018(1.8%)，y[1]的概率是0.245(24.5%)，y[2]的概率是0.737(73.7%)。从概率的结果来看，可以说“因为第2个元素的概率最高，所以答案是第2个类别”。也就是说，通过使用softmax函数，我们可以用概率的（统计的）方法处理问题。

#### 3.5.4 输出层的神经元数量

输出层的神经元数量需要根据待解决的问题来决定。对于分类问题，**输出层的神经元数量一般设定为类别的数量**。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093014374620240930143746.png" alt="image-20240930143746440" style="zoom:67%;" />

### 3.6 手写数字识别

和求解机器学习问题的步骤（分成学习和推理两个阶段进行）一样，使用神经网络解决问题时，也需要首先使用训练数据（学习数据）进行**权重参数**的学习；进行推理时，使用刚才学习到的参数，对输入数据进行分类。

#### 3.6.1 MNIST数据集

MNIST是机器学习领域最有名的数据集之一，被应用于从简单的实验到发表的论文研究等各种场合。

MNIST数据集是由0到9的数字图像构成的。训练图像有6万张，测试图像有1万张，这些图像可以用于学习和推理。MNIST数据集的一般使用方法是，先用训练图像进行学习，再用学习到的模型度量能在多大程度上对测试图像进行正确的分类。

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093015041820240930150418.png" alt="image-20240930150418117" style="zoom:67%;" />

下载源码和训练测试文件

源码中的load_mnist函数以“（训练图像,训练标签），（测试图像,测试标签）”的形式返回读入的MNIST数据，还可以设置load_mnist(normalize=True, flatten=True,one_hot_label=False)三个参数。

- normalize：设置是否将输入图像正规化为0.0~1.0的值。设置为False则输入图像像素保持原来的0~255
- flatten：是否展开输入图像。设置为False则输入图像为1✖28✖28的三维数组；设置为True则输入图像会保存为由784个元素构成的一维数组
- one_hot_label：设置是否将标签保存为one-hot表示(one-hot representation)。one-hot表示是仅正确解标签为1，其余皆为0的数组，就像[0,0,1,0,0,0,0,0,0,0]这样。当one_hot_label为False时，只是像7、2这样简单保存正确解标签；当one_hot_label为True时，标签则保存为one-hot表示。

Python有pickle这个便利的功能。这个功能可以将程序运行中的对象保存为文件。如果加载保存过的pickle文件，可以立刻复原之前程序运行中的对象。用于读入MNIST数据集的load_mnist()函数内部也使用了pickle功能（在第2次及以后读入时）。利用pickle功能，可以高效地完成MNIST数据的准备工作。

~~~python
# 使用Python的PIL模块显示图像，同时确认一下数据
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))	# 把保存为NumPy数组的图像数据转换为PIL用的数据对象
    pil_img.show()

'''
x_train是mnist数据的训练图像，共有60000张28*28像素的图片组成。因此x_train[0]是取这60000张图片的第一张
'''
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False)
img = x_train[0]	# 获取mnist数据组的第一张训练图像
label = t_train[0]	# 获取第一张图像的标签
print(label) # 9

print(img.shape)          # (784,)
img = img.reshape(28, 28) # 把图像的形状变成原来的尺寸
print(img.shape)          # (28, 28)

img_show(img)

~~~

<img src="https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024093016345820240930163458.png" alt="image-20240930163457893" style="zoom:67%;" />

flatten=True时读入的图像是以一列（一维）NumPy数组的形式保存的。因此，显示图像时，需要把它变为原来的28像素×28像素的形状。可以**通过reshape()方法的参数指定期望的形状**，更改NumPy数组的形状。此外，还需要把保存为NumPy数组的图像数据转换为PIL用的数据对象，这个转换处理由Image.fromarray()来完成。

#### 3.6.2 神经网络的推理处理

MNIST数据集的神经网络输入层有784个神经元，输出层有10个神经元。

784这个数字来源于图像大小的28*28=784

10这个数字来源于10类被分类（数字0到9，共10类别）

此外，这个神经网络有2个隐藏层，第1个隐藏层有50个神经元，第2个隐藏层有100个神经元。这个50和100可以设置为任何值。

~~~python
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

'''
init_network()会读入保存在pickle文件sample_weight.pkl中的学习到的权重参数,这个文件中以字典变量的形式保存了权重和偏置参数。
'''
def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

def pridict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W1) + b3
    y = softmax(a3)

    return y
~~~

以上3个函数实现神经网络的推理处理。然后，评价它的识别精度(accuracy)，即能在多大程度上正确分类。

~~~python
'''
首先获得MNIST数据集，生成网络。接着，用for语句逐一取出保存在x中的图像数据，用predict()函数进行分类。predict()函数以NumPy数组的形式输出各个标签对应的概率。比如输出[0.1, 0.3, 0.2, ..., 0.04]的数组，该数组表示“0”的概率为0.1，​“1”的概率为0.3，等等。然后，我们取出这个概率列表中的最大值的索引（第几个元素的概率最高）​，作为预测结果。可以用np.argmax(x)函数取出数组中的最大值的索引，np.argmax(x)将获取被赋给参数x的数组中的最大值元素的索引。最后，比较神经网络所预测的答案和正确解标签，将回答正确的概率作为识别精度。
'''
'''
在这个例子中，我们把load_mnist函数的参数normalize设置成了True。将normalize设置成True后，函数内部会进行转换，将图像的各个像素值除以255，使得数据的值在0.0～1.0的范围内。像这样把数据限定到某个范围内的处理称为正规化(normalization)。
'''
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = pridict(network, x[i])
    p = np.argmax(y)	# 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print('Accuracy:' + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.0937

~~~

#### 3.6.3 批处理

~~~python
>>> x, _ = get_data()
>>> network = init_network()
>>> W1, W2, W3 = network['W1'], network['W2'], network['W3']
>>>
>>> x.shape
(10000, 784)
>>> x[0].shape
(784,)
>>> W1.shape
(784, 50)
>>> W2.shape
(50, 100)
>>> W3.shape
(100, 10)

~~~

从上述结果来看，多维数组的对应维度的元素个数确实是一致的。我们还可以确认最终的输出结果是一个元素个数为10的一维数组。

![image-20241002155806173](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024100215580620241002155806.png)

从整体的处理流程来看，输入一个由784个元素（原本是一个28×28的二维数组）构成的一维数组后，输出一个有10个元素的一维数组。这是只输入一张图像数据时的处理流程。

用predict()函数一次性打包处理100张图像。为此，可以把x的形状改为100×784，将100张图像打包作为输入数据。如下图所示

![image-20241002160128394](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024100216012820241002160128.png)

输入数据的形状为100×784，输出数据的形状为100×10。这表示输入的100张图像的结果被一次性输出了。比如，x[0]和y[0]中保存了第0张图像及其推理结果。

~~~python
# 基于批处理的代码实现
x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

~~~

range()函数若指定为range(start, end)，则会生成一个由start到end-1之间的整数构成的列表。若像range(start, end, step)这样指定3个整数，则生成的列表中的下一个元素会增加step指定的值。

~~~python
>>> list( range(0, 10) )
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> list( range(0, 10, 3) )
[0, 3, 6, 9]
~~~

在range()函数生成的列表的基础上，通过x[i:i+batch_size]从输入数据中抽出批数据。x[i:i+batch_n]会取出从第i个到第i+batch_n个之间的数据。本例中是像x[0:100]、x[100:200]……这样，从头开始以100为单位将数据提取为批数据。

然后，通过argmax()获取值最大的元素的索引。不过这里需要注意的是，我们给定了参数axis=1。这指定了在100×10的数组中，沿着第1维方向（以第1维为轴）找到值最大的元素的索引（第0维对应第1个维度）[插图]。这里也来看一个例子。

~~~python
>>> x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6],
...     [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
>>> y = np.argmax(x, axis=1)
>>> print(y)
[1 2 1 0]
~~~

最后，我们比较一下以批为单位进行分类的结果和实际的答案。为此，需要在NumPy数组之间使用比较运算符(==)生成由True/False构成的布尔型数组，并计算True的个数。我们通过下面的例子进行确认。

~~~python
>>> y = np.array([1, 2, 1, 0])
>>> t = np.array([1, 2, 0, 0])
>>> print(y==t)
[True True False True]
>>> np.sum(y==t)
3
~~~

## 第4章 神经网络的学习

这里所说的“学习”是指从训练数据中自动获取最优权重参数的过程。

### 4.1 从数据中学习

神经网络的特征就是可以从数据中学习。所谓“从数据中学习”，是指可以由数据自动决定权重参数的值。

#### 4.1.1 数据驱动

![image-20241009145146242](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024100914515320241009145153.png)

如何识别数字5

![image-20241009153354164](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024100915335420241009153354.png)

上图表示从人工设计规则转变为由机器从数据中学习：没有人为介入的方块用灰色表示

#### 4.1.2 训练数据和测试数据

使用训练数据进行学习，寻找最优的参数；

使用测试数据评价训练得到的模型的实际能力。

为什么需要将数据分为训练数据和测试数据呢？因为我们追求的是模型的**泛化能力**。为了正确评价模型的泛化能力，就必须划分训练数据和测试数据。另外，训练数据也可以称为**监督数据**。

**泛化能力**是指处理未被观察过的数据（不包含在训练数据中的数据）的能力。

### 4.2 损失函数

神经网络的学习中所用的指标称为**损失函数**(loss function)。这个损失函数可以使用任意函数，但一般用**均方误差**和**交叉熵误差**等。

损失函数是表示神经网络性能的“恶劣程度”的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致。

#### 4.2.1 均方误差

![image-20241009154141082](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024100915414120241009154141.png)

y<sub>k</sub>表示神经网络的输出，t<sub>k</sub>表示监督数据，k表示数据的维数。

在3.6节手写数字识别的例子中，y<sub>k</sub>、t<sub>k</sub>是由如下10个元素构成的数据

~~~python
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
~~~

数组元素的索引从第一个开始依次对应数字“0”“1”“2”……这里，神经网络的输出y是softmax函数的输出。由于softmax函数的输出可以理解为概率，因此上例表示“0”的概率是0.1，“1”的概率是0.05，“2”的概率是0.6等。t是监督数据，将正确解标签设为1，其他均设为0。这里，标签“2”为1，表示正确解是“2”。将正确解标签表示为1，其他标签表示为0的表示方法称为**one-hot**表示。

~~~python
# 使用Python实现均方误差函数
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)
~~~

~~~python
>>> # 设“2”为正确解
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>>
>>> # 例1：“2”的概率最高的情况（0.6）
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> mean_squared_error(np.array(y), np.array(t))
0.097500000000000031
>>>
>>> # 例2：“7”的概率最高的情况（0.6）
>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
>>> mean_squared_error(np.array(y), np.array(t))
0.59750000000000003
'''
这里举了两个例子。第一个例子中，正确解是“2”​，神经网络的输出的最大值是“2”​；第二个例子中，正确解是“2”​，神经网络的输出的最大值是“7”​。如实验结果所示，我们发现第一个例子的损失函数的值更小，和监督数据之间的误差较小。也就是说，均方误差显示第一个例子的输出结果与监督数据更加吻合。
'''
~~~

#### 4.2.2 交叉熵误差

![image-20241010073446162](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101007345320241010073453.png)

log表示以e为底数的自然对数log<sub>e</sub>。y<sub>k</sub>是神经网络的输出，t<sub>k</sub>是**正确解标签**。

t<sub>k</sub>中只有正确解标签的索引为1，其他均为0（one-hot表示）。

因此，式(4.2)实际上只计算对应正确解标签的输出的自然对数。比如，假设正确解标签的索引是“2”，与之对应的神经网络的输出是0.6，则交叉熵误差是-log 0.6 = 0.51；若“2”对应的输出是0.1，则交叉熵误差为-log 0.1 = 2.30。也就是说，交叉熵误差的值是由正确解标签所对应的输出结果决定的。

![image-20241010081903614](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101008190320241010081903.png)

上图为自然对数y=logx的图像，x等于1时，y为0；随着x向0靠近，y逐渐变小。因此，正确解标签对应的输出越大，式(4.2)的值越接近0；当输出为1时，交叉熵误差为0。此外，如果正确解标签对应的输出较小，则式(4.2)的值较大。

~~~python
# 代码实现交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))
~~~

这里，参数y和t是NumPy数组。函数内部在计算np.log时，加上了一个微小值delta。这是因为，当出现np.log(0)时，np.log(0)会变为负无限大的-inf，这样一来就会导致后续计算无法进行。作为保护性对策，添加一个微小值可以防止负无限大的发生。

~~~python
# 使用cross_entropy_error(y, t)进行一些简单的计算
>>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
>>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
>>> cross_entropy_error(np.array(y), np.array(t))
0.51082545709933802
>>>
>>> y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
>>> cross_entropy_error(np.array(y), np.array(t))
2.3025840929945458
~~~

第一个例子中，正确解标签对应的输出为0.6，此时的交叉熵误差大约为0.51。第二个例子中，正确解标签对应的输出为0.1的低值，此时的交叉熵误差大约为2.3。

#### 4.2.3 mini-batch学习

机器学习使用训练数据进行学习。使用训练数据进行学习，严格来说，就是**针对训练数据计算损失函数的值**，找出**使该值尽可能小**的参数。因此，计算损失函数时必须将所有的训练数据作为对象。也就是说，如果训练数据有100个的话，我们就要把这100个损失函数的总和作为学习的指标。

前面介绍的损失函数的例子中考虑的都是针对单个数据的损失函数。如果要求所有训练数据的损失函数的总和，以交叉熵误差为例，可以写成下面的式(4.3)。

![image-20241012103852028](https://gitee.com/fangdaxi/fangdaxi_img/raw/master/2024101210385920241012103859.png)

这里，假设数据有N个，t<sub>nk</sub>表示第n个数据的第k个元素的值（y<sub>nk</sub>是神经网络的输出，t<sub>nk</sub>是监督数据）。

MNIST数据集的训练数据有60000个，如果以全部数据为对象求损失函数的和，则计算过程需要花费较长的时间。再者，如果遇到大数据，数据量会有几百万、几千万之多，这种情况下以全部数据为对象计算损失函数是不现实的。因此，我们从全部数据中选出一部分，作为全部数据的“近似”。神经网络的学习也是**从训练数据中选出一批数据**（称为**mini-batch,小批量**），然后对每个mini-batch进行学习。比如，从60000个训练数据中**随机选择**100笔，再用这100笔数据进行学习。这种学习方式称为mini-batch学习。

~~~python
# 随机选定指定个数的数据进行mini-batch学习
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)  # 设定参数one_hot_label=True，可以得到one-hot表示（仅正确解标签为1，其余为0的数据结构）​

print(x_train.shape)  # (60000, 784)，输入数据为28*28=784的图像数据
print(t_train.shape)  # (60000, 10)，监督数据是10维的数据

train_size = x_train.shape[0]	# (60000, 784)[0]也就是60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # np.random.choice(60000, 10)会从0到59999之间随机选择10个数字。
x_batch = x_train[batch_mask]	# 从60000个图片中选出随机出来的10个图片
t_batch = t_train[batch_mask]	# 60000中10个图片的one-hot表示

# 运行np.random.choice(60000, 10)得到类似下面的结果
>>> np.random.choice(60000, 10)
array([ 8013, 14666, 58210, 23832, 52091, 10153, 8107, 19410, 27260,
21411])
~~~

#### 4.2.4 mini-batch版交叉熵误差的实现

~~~python
# 可以同时处理单个数据和批量数据两种情况的函数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
'''
这里，y是神经网络的输出，t是监督数据。y的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。并且，当输入为mini-batch时，要用batch的个数进行正规化，计算单个数据的平均交叉熵误差。
'''
~~~

