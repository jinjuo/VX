# Numpy 学习
## 由于Numpy和Pandas是基于C语言的，所以计算速度会快很多

import numpy as np 

##把一个列表转为矩阵
array = np.array([[1,2,3],[4,5,6]])
print array

## 查看矩阵的维度
print('number of dim:', array.ndim)

## 查看行数和列数
print('shape:', array.shape)

## 查看元素个数
print('size:', array.size)

## 创建矩阵

# array：创建数组
# dtype：指定数据类型
# zeros：创建数据全为0
# ones：创建数据全为1
# empty：创建数据接近0
# arrange：按指定范围创建数据
# linspace：创建线段

a = np.array([1,2,3]) # 一般都是用list来创建矩阵

## 定义数据类型
a = np.array([1,2,3], dtype= np.int)
print(a.dtype)
### 除了int
# NumPy中的基本数据类型
# 名称	描述
# bool	用一个字节存储的布尔类型（True或False）
# inti	由所在平台决定其大小的整数（一般为int32或int64）
# int8	一个字节大小，-128 至 127
# int16	整数，-32768 至 32767
# int32	整数，-2 ** 31 至 2 ** 32 -1
# int64	整数，-2 ** 63 至 2 ** 63 - 1
# uint8	无符号整数，0 至 255
# uint16	无符号整数，0 至 65535
# uint32	无符号整数，0 至 2 ** 32 - 1
# uint64	无符号整数，0 至 2 ** 64 - 1
# float16	半精度浮点数：16位，正负号1位，指数5位，精度10位
# float32	单精度浮点数：32位，正负号1位，指数8位，精度23位
# float64或float	双精度浮点数：64位，正负号1位，指数11位，精度52位
# complex64	复数，分别用两个32位浮点数表示实部和虚部
# complex128或complex	复数，分别用两个64位浮点数表示实部和虚部
### 默认都是64位的，精度最高，但占用空间最大；要是想多存点数同时对精度没那么高要求，可以用16（eg. float16）或者8

## 创建一个n行，m列的全零矩阵
a = np.zeros((n,m))

## 创建一个n行，m列的全1矩阵
a = np.ones((n,m), dtype= int16)

## 创建一个n行，m列的全部数字接近0的矩阵
a = np.empty((n,m))

## random
a = np.random.random((2,4)) # 创建一个2行4列的随机矩阵，其中数值都是0~1的随机浮点数
### 更多关于random类的教程：https://blog.csdn.net/akadiao/article/details/78252840?locationNum=9&fps=1

## arange
a = np.arange(10,20,2) #创建一个从10到19的，2为步长有序的序列
a = np.arange(12) #从0到10的有序序列
a = np.arange(12).reshape((3,4)) #再把这个数列变成3行4列的矩阵
a = np.linspace(1,10,5) #生成线段，第一个数字是1，最后一个数字是10,一共5个数字


# Numpy中的运算
## Numpy中的加减乘除，都是对应位置的元素，进行运算
import numpy as np
a = np.array([10,20,30,40])
b = np.arange(4)
print(a + b )
print(a - b)
print(a*b) ##注意Python中的次方运算是双*号，比如a的平方是a**2,a的5次方是a**5
print(a/b)

## 进行三角函数运算，就直接
c = np.sin(a) # np.cos(), np.tan()

## 矩阵中元素的判断，会逐个返回TRUE/FALUS值
print(b == 3) #注意1个等号是赋值，2个等号才是等于的意思
print(b<3)

## 矩阵的乘法有2种
a = np.array([[1,1],[1,0]])
b = np.arange(4).reshape((2,2))
c = a*b #元素逐个相乘
c = np.dot(a,b) #矩阵的乘法法则，点积; 表达式也可以写成
c_2 = a.dot(b) #两者是等价的

## 找到矩阵中最大值、最小值，或者矩阵中数值全部求和；同时可以用axis参数来控制按行或列来进行操作：
## 当axis的值为0的时候，将会以列作为查找单元， 当axis的值为1的时候，将会以行作为查找单元
np.sum(a, axis = 0) #按列，求和
np.min(a, axis = 1) #按行，求最小值
np.max() #最大值

## 找到最大或最小的索引
np.argmin()
np.argmax()

## 其他处理
np.mean() #均值；都可以写成a.mean()，其中a是对象矩阵
np.median() #中位数
np.cumsum() #累加
np.diff() #累减/差
np.nonzero() #非零的数的位置，会给出2个array，一个是非零的行位置，一个是非零的列位置
np.sort() #排序
np.transpose() #矩阵倒置；也可以写成 a.T
np.clip(a,5,9) #a矩阵中，所有小于5的数都转换成5,所有大于9的数，都转换成9，5~9的数字不变

## Numpy的索引
### 先创建一个矩阵
A = np.arange(3,15).reshape((3,4))
### 通过切片取数值
print([2][1]) # 等价于 print([2,1])
print([2,:]) # 可以用：代表该行的所有，例如：print([:,1]) or print([1,1:3])
### 用for循环来取数值
#### 按行取数值
for row in A:
	print(row)
#### 按列取数值
for column in A.T: #python没有内置的取列功能，所以我们先把矩阵转置一下，把原来的列变成行，再按行取出来
	print(column)
#### 按item取数值
for item in A.flat: #先把矩阵拉平，变成一行的数列，再一个一个取出来
	print(item)
print(A.flatten()) #可以看到矩阵拉平后的效果。.flatten()与.flat的区别在于，后者只是一个迭代器，不会return具体的数值

## 矩阵的合并与分割
### 先准备2个array
A = np.array([1,1,1])
B = np.array([2,2,2])
### 合并操作
print(np.vstack((A,B))) #上下合并；可以用.shape查看合并后的行列数
print(np.hstack((A,B))) #左右合并；
print(A[n p.newaxis,:]) #在行上加一个维度；ps，无法对单行序列使用.T转置
print(A[:,np.newaxis]) #在列上加一个维度
#### 所以，要想把A，B纵向合并，那就要先把A，B都用np.newaxis做序列转置，然后np.hstack()，操作如下：
A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]
print(np.hstack((A,B)))
------
print(np.concatenate((A,B,A,B), axis = 0)) #np.concatenate()是对以上合并方法的一个集合，可以先合并，然后用axis参数来控制是按行还是按列。axis=0是按列，在垂直方向上合并，相当于vstack，axis=1是按行，是在水平方向上合并，相当于hstack

### 分割操作
A = np.arange(12).reshape((3,4))
print(np.split(A, 2, axis = 1)) #纵向切割，分成2块，把每一列单独切出来
print(np.split(A, 3, axis = 0)) #横向切割，分成3块，把每一行单独切出来
print(np.array_split(A, 3, axis = 1)) #纵向不等量切割，该函数会把array的前两列作为第一个切片，后面两列分别2个切片
#### 也可以：
print(np.vsplit(A,3)) #纵向切割，结果是原来的每一行成为新的array
print(np.hsplit(A,2)) #横向切割，结果是原来前两列和后两列，分别成为新的array


## array的赋值
### = 的赋值方法会带有关联性
### copy()的赋值方法没有关联性
b = a.copy() #也就是deep copy

# 啥时候复习哈：0514.。。
