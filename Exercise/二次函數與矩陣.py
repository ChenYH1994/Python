import matplotlib.pyplot as plt
import numpy as np
x=np.arange(-100,101,1)
y=3*np.power(x,2)-12*x+4
print('因為平方前面的係數是正數,所以會有最小值')
print(np.min(y))
plt.plot(x,y)
print(x[y.argmin()])   # 找出極小值點
print(x[y.argmax()])   # 找出極大值點，但此函數無極大值點
plt.grid()
plt.show()

import numpy as np
f = np.array([3, -1])
a = np.array([3, -1]).reshape(2, 1)
b = np.array([8, 4, 0]).reshape(3, 1)
c = np.array([-1, 0, 0, 3]).reshape(4, 1)
d = np.array([4, 0, -3, 2, 1]).reshape(5, 1)
print(a);print("--------------")
print(a.shape)
print(a.ndim) # a是二維的
print(b);print("--------------")
print(c);print("--------------")
print(d);print("--------------")
print(f)
print(f.shape)
print(f.ndim)

import numpy as np
A = np.array([
[4, -7],
[2, -3]
])
A_inv = np.linalg.inv(A)   #得到反矩陣，若不可逆，則會產生錯誤
print(A.dot(A_inv))

import numpy as np
B = np.array([
[8, 2],
[12, 3]
])
try:
   B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError:
   print('矩陣 B 是不可逆矩陣')



import numpy as np
from scipy import stats
np1=np.array([15,10,30,30,30,25,25,55,57,54,80,75,77,84,91,92,88,90,95,86])
print(np1)
print("中位數:",np.median(np1))
print("平均值:",np.mean(np1))
print("筆數:",len(np1))
print("眾數:",stats.mode(np1))
print('非連續資料,使用bar顯示個別資料')
x=np.arange(0,len(np1))
print(x)
import matplotlib.pyplot as plt
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax1.bar(x,np1)
ax2=fig.add_subplot(1,2,2)
ax2.hist(np1,bins=5)
ax2.axvline(np.median(np1),0,1,c='r')
ax2.axvline(np.mean(np1),0,1,c='g')
plt.show()

import numpy as np
a=np.array([70,60,30,66,80,90,20,60,54,70])
b=np.array([50,60,40,66,70,80,50,60,64,60])
c=np.array([150,160,140,166,170,180,150,160,164,160])
print("a班級平均值:",np.mean(a))
print("b班級平均值:",np.mean(b))
print("c班級平均值:",np.mean(c))
print("a班級中位數:",np.median(a))
print("b班級中位數:",np.median(b))
print("c班級中位數:",np.median(c))
print("a班的變異數:",np.var(a));print("a班的標準差:",np.std(a))
print("b班的變異數:",np.var(b));print("b班的標準差:",np.std(b))
print("c班的變異數:",np.var(c));print("c班的標準差:",np.std(c))
cv1=np.std(a)/np.mean(a);cv2=np.std(b)/np.mean(b)
cv3=np.std(c)/np.mean(c)
print("a班的變異係數:",cv1);print("b班的變異係數:",cv2)
print("c班的變異係數:",cv3)

import matplotlib.pyplot as plt
import numpy as np
city = ['Delhi', 'Beijing', 'Washington', 'Tokyo', 'Moscow']
pos = np.arange(len(city))
Happiness_Index = [60, 40, 70, 65, 85]
plt.bar(pos, Happiness_Index, color='blue', edgecolor='black')
plt.xticks(pos, city)
plt.xlabel('City', fontsize=16)
plt.ylabel('Index', fontsize=16)
plt.title('Barchart ', fontsize=20)
plt.show( )



