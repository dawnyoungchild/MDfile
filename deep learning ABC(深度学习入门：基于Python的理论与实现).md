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