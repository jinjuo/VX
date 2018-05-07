# Matplotlib Plt学习

## 画一条直线
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)
y = 2*x + 1
plt.plot(x,y)
plt.show()

## figure：窗口
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure() #每个figure下的内容，就是该窗口中绘图内容
plt.plot(x,y1)

plt.figure(num = 3, figsize = (8,5)) #可以给figure设置编号、大小等参数
plt.plot(x, y2)
plt.plot(x, y1, color = 'red', linewidth = 1.0, linestyle = '--') #绘图参数设置

plt.show()


## axis：坐标轴
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2
plt.figure()
## 设置坐标轴的取值范围
plt.xlim((-1,1))
plt.ylim((-2,3))
## 设置坐标轴的描述标签
plt.xlabel('I am X')
plt.ylabel('I am Y')
## 更换坐标值的刻度:ticks
new_ticks = np.linspace(-2,2,11)
plt.xticks(new_ticks)
### 把y轴的刻度换成文字，注意一一对应
plt.yticks(
	[-2,-1,0,3],
	['really bad','bad','normal','really good']
	)
#### 更改y轴文字的字体，并打印出特殊数学字符alpha
# plt.yticks(
# 	[-2,-1,0,3],
# 	[r'$really\ bad$',r'$bad\ \alpha$',r'$normal$',r'$really\ good$']
# 	)
# # 其中，r是正则的意思，要把所有文字用$框起来，空格直接读不出来，所以要在所有空格前加\
# # 特殊数学符号前，要加\
plt.plot(x, y2)
plt.plot(x, y1, color = 'red', linewidth = 1.0, linestyle = '--') 
plt.show()


## 修改坐标轴的位置，用到gca= get current axis
ax = plt.gca()
ax.spines['right'].set_color('none') #spines是边框，把右边框隐掉
ax.spines['top'].set_color('none') #把上边框隐掉
ax.xaxis.set_ticks_position('bottom') #把下边框设置为X轴（默认是没有指定X轴和Y轴的）
ax.yaxis.set_ticks_position('left') #把左边框设置为Y轴
ax.spines['bottom'].set_position(('data',0)) #把X轴的位置，设定在y轴数值是0的位置；除了用data，还可以用很多参数来设置，之后再研究吧
ax.spines['left'].set_position(('data',0)) #把Y轴的位置，设定中X轴数值是0的位置

## 设置图例 legend
plt.plot(x, y2, label = 'up')
plt.plot(x, y1, color = 'red', linewidth = 1.0, linestyle = '--', label = 'down')
plt.legend() #legend 有handles,labels，loc参数可以用，loc默认值是best
plt.show()

## 标注 annotation
x0 = 1
y0 = 2*x0+1
plt.scatter(x0,y0,size= 50,color = 'b')
plt.plot([x0,x0],[y0,0],'k--',linewidth = 2.5)
# method 1：annotate
plt.annotate(r'$2x+1=%s$'%y0, xy=(x0,y0), xycoords = 'data', xytext = (+30, -30), textcoords = 'offset points',
				fontsize = 16, arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3, rad=.2'))	
# method 2：text
plt.text(-4,4,r'$This\ is\ a\ text\ with \mu\ \sigam_i\ \alpha_t$',
		fontdict={'size':16, 'color':'r'})

## 画散点图
import matplotlib.pyplot as plt
import numpy as np

n = 1024
X = np.random.normal(0,1,n) #n个均值为0，方差为1的随机数
Y = np.random.normal(0,1,n)
T = np.arctan2(Y,X) #for color value
plt.scatter(X,Y,s = 75, c=T, alpha = 0.5)
plt.xlim((-1.5, 1.5))
plt.ylim((-1.5, 1.5))
plt.xticks(()) #隐藏坐标轴刻度
plt.yticks(())


## 画柱状图
import matplotlib.pyplot as plt
import numpy as np

n = 12

X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5, 1, n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5, 1, n)

plt.bar(X,+Y1, facecolor = '#9999ff', edgecolor = 'white')
plt.bar(X,-Y2,facecolor = '#ff9999', edgecolor = 'white')
for x,y in zip(X,Y1):
	plt.text(x + 0.4, y + 0.05, '%.2f' %y, ha = 'center', va = 'bottom')# ha = horizontal alignment 
for x,y in zip(X,Y2):
	plt.text(x + 0.4, -y - 0.05, '-%.2f' %y, ha = 'center', va = 'top')

plt.xlim((-0.5, n))
plt.xticks(()) #隐藏坐标轴刻度
plt.ylim((-1.25, 1.25))
plt.yticks(())

plt.show()


## 画等高线图
import matplotlib.pyplot as plt
import numpy as np
def f(x,y): # 设置一个高度的function
	return (1-x/2+ x**5 + y**3) * np.exp(-x**2-y**2)

n =256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x,y) #设置网格

plt.contourf(X,Y,f(X,Y),8,cmap = plt.cm.hot, alpha = 0.75) #给点加上颜色

C = plt.contour(X,Y,f(X,Y),8,colors = 'black', linewidth = 0.5) #画等高线
plt.clabel(C, inline = True, fontsize = 10) # 线上加数字标签

plt.xticks(())
plt.yticks(())

plt.show()



## 画3d图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
# X, Y values
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2+Y**2)
# height value
Z = np.sin(R)

# 绘图
ax.plot_surface(X,Y,Z,rstride = 1, cstride = 1, cmap= plt.get_cmap('rainbow')) # rstride和cstride是行密度和列密度 
ax.contourf(X,Y,Z, offset = -2, zdir= 'z', cmap = 'rainbow') # zdir是把3d图按哪个轴的方向压下去
ax.set_zlim(-2, 2)
plt.show()

## subplot中一个页面中同时画多个图
import matplotlib.pyplot as plt

plt.figure()
# method 1：均匀构图
plt.subplot(2,2,1)
plt.plot([0,1],[0,1])
plt.subplot(2,2,2)
plt.plot([0,1],[0,2])
plt.subplot(2,2,3)
plt.plot([0,1],[0,3])
plt.subplot(2,2,4)
plt.plot([0,1],[0,4])
plt.show()

# method 2：不均匀构图
import matplotlib.pyplot as plt
plt.figure()

plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.subplot(2,3,4)
plt.plot([0,1],[0,2])
plt.subplot(2,3,5)
plt.plot([0,1],[0,3])
plt.subplot(2,3,6)
plt.plot([0,1],[0,4])
plt.show()

# method3：subplot2grid
import matplotlib.pyplot as plt
plt.figure()
ax1 = plt.subplot2grid((3,3),(0,0),colspan = 3, rowspan = 1)
ax1.set_title('ax1_title')
ax1.plot([1,1],[2,2])
ax2 = plt.subplot2grid((3,3),(1,0),colspan = 2, rowspan = 1)
ax3 = plt.subplot2grid((3,3),(1,2),colspan = 1, rowspan = 2)
ax4 = plt.subplot2grid((3,3),(2,0),colspan = 1, rowspan = 1)
ax5 = plt.subplot2grid((3,3),(2,1),colspan = 1, rowspan = 1)
plt.show()

# method 4:gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure()
gs = gridspec.GridSpec(3,3) #3行3列
ax1 = plt.subplot(gs[0, :])
ax2 = plt.subplot(gs[1, :2])
ax3 = plt.subplot(gs[1:, 2])
ax4 = plt.subplot(gs[-1, 0])
ax5 = plt.subplot(gs[-1, -2])
plt.show()

# method 5: subplots
import matplotlib.pyplot as plt

plt.figure()
f,((ax11, ax12),(ax21, ax22)) = plt.subplots(2,2, sharex = True, sharey = True)
ax11.scatter([1,2],[2,4])
plt.tight_layout()
plt.show()



## 图中图
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure()
# 创建数据
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]
# 画大图
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')
# 左上角小图
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(y, x, 'b')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('title inside 1')

# 右小角小图
plt.axes([0.6, 0.2, 0.25, 0.25])
plt.plot(y[::-1], x, 'g') # 注意对y进行了逆序处理
plt.xlabel('x')
plt.ylabel('y')
plt.title('title inside 2')

plt.show()


## 次坐标
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y1 = 0.05 * x**2
y2 = -1 * y1

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, y1, 'g-')   # green, solid line
ax2.plot(x, y2, 'b-') # blue

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')
ax2.set_ylabel('Y2 data', color='b')

plt.show()

## 动画 animation
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
fig, ax = plt.subplots()
# 数据准备
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x)) #加逗号是因为这里是一个list，我们要选第一位

def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,
def init():
    line.set_ydata(np.sin(x))
    return line,
ani = animation.FuncAnimation(fig=fig,
                              func=animate,
                              frames=100,
                              init_func=init,
                              interval=20,
                              blit=False)
# 参数说明：
# fig 进行动画绘制的figure
# func 自定义动画函数，即传入刚定义的函数animate
# frames 动画长度，一次循环包含的帧数
# init_func 自定义开始帧，即传入刚定义的函数init
# interval 更新频率，以ms计
# blit 选择更新所有点，还是仅更新产生变化的点。应选择True，但mac用户请选择False，否则无法显示动画

plt.show()

ani.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])# 如果要保存视频



# first learn on May 7th,2018
