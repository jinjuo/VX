# Pandas 学习
## pandas更像一个字典的numpy，比如可以对每一列设置一个名字

# pandas 是基于 Numpy 构建的，让以 Numpy 为中心的应用变得更加简单。
# pandas主要包括三类数据结构，分别是：
# Series：一维数组，与Numpy中的一维array类似。二者与Python基本的数据结构List也很相近，其区别是：List中的元素可以是不同的数据类型，而Array和Series中则只允许存储相同的数据类型，这样可以更有效的使用内存，提高运算效率。
# DataFrame：二维的表格型数据结构。很多功能与R中的data.frame类似。可以将DataFrame理解为Series的容器。以下的内容主要以DataFrame为主。
# Panel ：三维的数组，可以理解为DataFrame的容器。
# Pandas官网，更多功能请参考http://pandas-docs.github.io/pandas-docs-travis/index.html

import numpy as np
import pandas as pd

s = pd.Series([1,3,6,np.nan,44,1]) # 设置一个序列，注意S是大写
dates = pd.date_range('20180101', periods = 6) #生成一个DatetimeIndex['2018-01-01','2018-01-02','2018-01-03','2018-01-04','2018-01-05','2018-01-06']

## 定义DataFrame的方法一：指定行、列标签
df = pd.DataFrame(np.random.randn(6,4), index = dates, columns = ['a','b','c','d']) # 建一个DataFrame，类似Numpy中的二维矩阵，每行是一个日期，列分别是abcd

## 定义DataFrame的方法二：用字典
df_2 = pd.DataFrame({
	'A':1., 
	'B':pd.Timestamp('20130102'), 
	'C':pd.Series(1, index= list(range(4)),dtype = 'float32'), 
	'D':np.array([3]*4, dtype='int32'),
	'E':pd.Categorical(["test","train","test","train"]),
	'F':'foo'
	})

## 查看DataFrame的类型
df.dtypes # 不用括号，因为这是dataframe的属性

## 查看行的序号
df.index

## 查看列的名字
df.columns

## 查看数值
df.values

## 查看统计数据
df.describe() #会返回每一列的count,mean,std,min,25%,50%,75%,max；日期、字符串等非数字values，该命令不会显示

## Dateframe的转置
df.T

## 排序
df.sort_index(axis = 1, ascending = False) #对列按倒序排序，结果列标：FEDCBA
df.sort_index(axis = 0, ascending = False) #对行按倒序排序，结果行标：3210
df.sort_values(by='E') #按E列的值排序

## pandas取值
dates = pd.date_range('20180101', periods = 6)
df = pd.DataFrame(np.arange(24).reshape((6,4), index = dates, columns = ['A','B','C','D'])
### 简单的选择
print(df['A'], df.A) # 两者等同
print(df[0:3],df['20180101':'20180103'])  # 两者等同
### 高级的选择
#### select  by label:loc
print(df.loc['20180101']) # 选择该日期的一行
print(df.loc[:,['A','B']]) # 选择所有行和A，B两列
print(df.loc['20180101', ['A', 'B']]) # 只选择20180101这一行的A，B两列；注意，loc都是标签，所以都是用标签去选择
#### select by position: iloc
print(df.iloc[3]) # 选择第三行的数据
print(df.iloc[3,0]) # 选择第三行第一列的数据
print(df.iloc[3:5, 1:3]) #切片筛选
print(df.iloc[[1,3,5],1:3]) #非连续的逐个筛选
#### mixed select by label & position:ix
print(df.ix[3:, ['A', 'C']]) # 行是用position选，列是用label选的mixed select
#### Boolean indexing select
print(df[df.A > 8]) #筛选所有A列数值大于8的dataframe
### 使用isin()方法过滤在指定列中的数据
df[df['high'].isin([0.00, 9.00])] #筛选df数据中high列中是0或者9的数据

## pandas 设置值
dates = pd.data_range('20180101', periods = 6)
df = pd.DataFrame(np.arange(24).reshape((6,4), index = dates, columns = ['A','B','C','D'])
### 用iloc，赋值
df.iloc[2,2] = 111 ## 第3行第3列赋值为111
### 用loc，赋值
df.loc['20180102', 'B'] = 222
### 用ix赋值，同理
### boolean indexing
df[df.A > 4] = 0 #所有A列大于4的dataframe，每行每列的数值都变成0
df.A[df.A > 4] = 0 #所有A列数值大于4的A列数值，都变成0，其他列数据不变
df['F'] = np.nan #新增F列，并都设值为NAN
df['E'] = pd.Series([1,2,3,4,5,6], index= pd.date_range('20180101', periods = 6)) # 新增E列，要用原来dataframe的相同行标签，这样才能对齐，然后用一个list把每一行的数值传进去

## 处理缺省值
dates = pd.data_range('20180101', periods = 6)
df = pd.DataFrame(np.arange(24).reshape((6,4), index = dates, columns = ['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
### 丢掉nan：dropna
print(df.dropna(axis = 0, how = 'any')) # 丢掉有any（任何）缺省值的行；how={'any','all'}，其中any是有一个nan就触发，all是只有所有都nan才触发；默认值是any
print(df.dropna(axis = 1)) #丢掉有任何nan的列
### 继续用nan：fillna
print(df.fillna(value = 0)) #把nan填成0
### 检查有没有nan
print(df.isnull()) 
print(np.any(df.isnull()==True) #检查所有数据中是否有nan

## 查看重复的数据
z.duplicated()


## 导入导出数据
### pd.read的文档：http://pandas.pydata.org/pandas-docs/stable/io.html

import pandas as pd

data = pd.read_csv(~/desktop/student.csv)
print(data)
##保存到pickle格式
data.to_pickle('student.pickle')


## 合并多个dataframe：concatenating


import numpy as np
import pandas as pd

### 首先准备几个dataframe
df1 = pd.DataFrame(np.ones((3,4))*0, columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns = ['a','b','c','d'])
### 方法1
res = pd.concat([df1,df2,df3], axis = 0) # 行的序号保持原来的状态（会有重复）
res = pd.concat([df1,df2,df3], axis = 0, ignore_index = True) # 重置index（没有重复）

### 方法2：join，有'inner'和'outer'两种
#### 准备数据
df4 = pd.DataFrame(np.ones((3,4))*0, columns = ['a','b','c','d'], index = [1,2,3])
df5 = pd.DataFrame(np.ones((3,4))*1, columns = ['b','c','d','e'], index = [2,3,4])
#### outer join
res2 = pd.concat([df4,df5], join = 'outer')
# or
res2 = pd.concat([df4,df5]) # 默认是outer
# outer join 会把所有列都保留，没有数值的位置自动填充NaN
### inner join
res3 = pd.concat([df4, df5], join = 'inner')
# inner join 只会保留共有的列，其余直接删掉

### 方法3：join_axes
res4 = pd.concat([df4, df5], axis = 1, join_axes = [df4.index])
# 做左右合并的时候，只考虑df4的index，df5中与df4有重合的保留，缺少的补NaN，多余的删掉

### 方法4：append
df1 = pd.DataFrame(np.ones((3,4))*0, columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1, columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2, columns = ['a','b','c','d'])

res5 = df1.append(df2, ignore_index = True)
res6 = df1.append([df2, df3], ignore_index = True) #用list可以同时append多个dataframe
# 也可以每次添加一条数据
s1 = pd.Series([1,2,3,4], index = ['a','b','c','d'])
res7 = df1.append(s1, ignore_index = True)

## 合并 merge by key，主要用中database中
#数据准备
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

res = pd.merge(left, right, on="key") # 基于某行或列合并
## 基于2个或多个key
#定义资料集并打印出，且key中的values不一样
left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
re = pd.merge(left, right, on = ['key1','key2']) #默认是inner join的方法
#等价于
re = pd.merge(left, right, on = ['key1','key2'], how = 'inner')
## how 有4种方式['left', 'right', 'inner', 'outer']

## indicator,merge的同时会把merge的信息写下来
#定义资料集
df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
res = pd.merge(df1, df2, on = "col1", how = "outer",indicator = True)
# 可以给indicator那一列命名，把True改成相应的名字，比如 ,indicator = 'indicator_colunms'

## merged by index
# 定义数据集
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
# left_index 和 right_index
res = pd.merge(left, right, left_index = True, right_index = True, how = 'outer')

## 解决重叠overlapping的问题
## 准备数据集
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})

re = pd.merge(boys, girls, on = 'k', suffixes = ['_boys', '_girls'], how = 'outer')

## join的功能，后续再学习

# Plot 数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
## Series
data = pd.Series(np.random.randn(1000), index = np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()

## DataFrame
data = pd.DataFrame(np.random.randn(1000,4),
					index = np.arange(1000),
					columns = ['A','B','C','D'])
# print(data.head())
data = data.cumsum()
data.plot() # plot()中有很多样式参数，之后再细看
plt.show()

## plot 的方法:bar, hist, box, scatter, kde, area, pie, hexbin
### 尝试一个散点图
ax = data.plot.scatter(x = 'A', y = 'B', color = 'DarkBlue', label = 'Class1')
data.plot.scatter(x = 'A', y = 'C', color = 'LightGreen', label = 'Class2', ax = ax)
plot.show()



# 0514不知道为什么总是提交不了