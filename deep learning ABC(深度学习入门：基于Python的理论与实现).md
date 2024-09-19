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

​		这里需要注意的是，数组x和数组y的元素个数是相同的（两者均是元素个数为3的一维数组）。当x和y的元素个数相同时，可以对各个元素进行算术运算。如果元素个数不同，程序就会**报错**，所以**元素个数保持一致**非常重要。对应元素的”的英文是element-wise，比如“对应元素的乘法”就是element-wise product。

##### 广播

​		NumPy数组不仅可以进行element-wise运算，也可以和单一的数值（标量）组合起来进行运算。此时，需要**在NumPy数组的各个元素和标量之间进行运算**。这个功能也被称为广播。

~~~python
>>> x = np.array([1.0, 2.0, 3.0])
>>> x / 2.0
array([ 0.5,  1. ,  1.5])
~~~

#### 1.5.4 NumPy的N维数组

