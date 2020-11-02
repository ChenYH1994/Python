import datetime              # 加入日期
now1=datetime.datetime.now( )
print(now1)
print(now1.year)
print(now1.month)
print(now1.date( ))
print(now1.day)
print(now1.hour)
print(now1.minute)
print(now1.second)
print(now1.time( ))

import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-10,10,50)
y1=2*x+1
y2=x**2-3*x-11
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')   # 畫圖
import datetime
now1=datetime.datetime.now()
a=(now1.year)
b=(now1.month)
c=(now1.day)
d=(now1.hour)
f=(now1.minute)
g=(now1.second)
filename=str(a)+str(b)+str(c)+str(d)+str(f)+str(g)+".png" # 含有時間的檔案名
plt.savefig(filename)                                     # 將畫好的圖，以上面檔名儲存

