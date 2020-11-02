import matplotlib.pyplot as plt
#使用jupyter,請加入以下這一行。
# %matplotlib inline
data=[-1,-4.3,15,21,31]
plt.figure(figsize=(8,5))
plt.plot(data*2)
plt.show()

x=[1,2,3,4,5]
y1=[1,2,5.5,6,7]
y2=[4,5,6,7,8]
plt.figure(num=3,figsize=(8,5))
plt.plot(x,y1,'r-o',x,y2,'g--')
plt.grid()              # 網格線
plt.show()

plt.figure(num=3,figsize=(8,5))
plt.plot([-1,2,3,4],[1,-4,9,16],'*')
plt.axis([0,6,0,20])    # 座標顯示範圍
plt.grid()
plt.show()

import numpy as np
plt.figure(figsize=(8,5))
x=np.linspace(-10,10,50)         # 50 代表分點數; x 是 array !
y1=2*x+1
y2=x**2-3*x-11
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=2.0,linestyle='dashdot')
plt.grid()
plt.xlim(-2,6)                   # x 軸範圍
plt.ylim(-5,10)                  # y 軸範圍
plt.title('plot')                # 圖形標題
plt.xlabel('x')                  # x 軸名稱
plt.ylabel('y')                  # y 軸名稱
plt.show()

plt.figure(figsize=(8,5))
x=np.linspace(-10,10,50)
y1=2*x+1
y2=x**2-3*x-11
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=2.0,linestyle='dashdot')
plt.grid()
plt.xlim(-2,6)
plt.ylim(-5,10)
plt.xticks(np.arange(9),('A','B','C',"D","E"))  # 重新設定 x 軸的刻度; xticks(要調整的位置，位置的內容)
print(np.linspace(-5,10,5))
plt.yticks([-5,-1.25,2.5,6.25,10],
[r'$\beta$',r'$\alpha$',r'$\mu$',r'$\sigma^2$','T'])  # 重新設定 y 軸的刻度搭配 Latex 
plt.title('plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()




