import random
import numpy as np
for i in range(20):
  a=random.randint(1,10)         # 在 [1,10] 這個範圍之間隨機產生一個整數
  print(a,end=' ')
print()
print("-----------------")
for i in range(20):
  c = random.randrange(1,10)     # 在 [1,10) 這個範圍之間隨機產生一個整數
  print(c,end=' ')
print()
print("-----------------")
for i in range(20):
  b = random.random()            # 隨機產生一個浮點數
  print(b,end=' ')
print()
print("-----------------")
for i in range(20):
  d = random.uniform(1,10)       # 在 [1,10) 之間隨機產生一個浮點數
  print(d,end=' ')
print()
print("-----------------")
z=np.random.randint(0,100,(5,5))    # 在 [0,100) 中隨機挑選整數形成 5*5 矩陣
print(z)
print()
print("-----------------")
y=np.random.randint(0,100,20)       # 在 [0,100) 中隨機挑選產生20個整數
print(y)
print()
print("-----------------")

import numpy.random as ra1
aa=['a','b','c','d']
for a in range(5):
  np1=ra1.choice(aa,3)           # 在 aa 中隨機挑選 3 個
  print(np1,end='')
print()
print('--1-----')
for a in range(5):
  np1=ra1.choice(aa,3,p=[0.1,0.3,0.6,0])      # 在 aa 中隨機挑選 3 個，並且分配機率給 aa 的元素
  print(np1,end='')
print()
print('---2-----')
for a in range(5):
  np1=ra1.choice(aa,3,p=[0.1,0.3,0.6,0],replace=False)  # replace=False,代表資料不要重複
  print(np1,end='')
print()
print('---3-----')
print(aa)
ra1.shuffle(aa)       # 將 aa 內資料順序打亂
print(aa)
print()
print("-----------------")

lst1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
slice = random.sample(lst1, 5)       # 在 lst1 中隨機挑 5 個，並返回 lst1
print(slice)

print()
print("-----------------")

f=np.random.rand(100)*40+30      # 產生 100 個 [30,70] 之間的實數
print(f)

print()
print("-----------------")


import numpy.random as ra1             # 頭彩機
a=ra1.choice(10,3)                     # 中獎號碼
print(a)
money=0
for b in range(1000):
   np1 = ra1.choice(10, 3)             # 1000位來賓得到的號碼
   if ((np1 == a).all() == True):      # 核對號碼，當 np1、a 兩組資料完全相同時，則這位來賓就中獎
       print(a)
       print("中獎!!")
       money += 1
print("獎金為",money*10000)             # 迴圈停止，來賓獎金為 money*10000

print()
print("-----------------")

a=np.random.binomial(100,0.5,20)       # 二項分配，20組實驗，每組丟公平硬幣 100 次的正面次數
print(a)                               # 實驗次數愈多， 次數 / 100 就越接近 1/2

print()
print("-----------------二項分配")

import matplotlib.pyplot as plt
flg=plt.figure()
count1,total=1,0
for i in range(8):                    # 共 8 大組實驗
  a=np.random.binomial(100,0.5,20)    # 一大組實驗，有 20 組小實驗，每小組丟 100 次硬幣，產生 20 個數據
  ax1=flg.add_subplot(4,2,count1)     # 分成 4x2 個子圖，count1 是圖的位置
  ax1.hist(a,bins=5)                  # 每大組實驗的直方圖
  print(np.mean(a))                   # 一組實驗的正面次數平均值
  total+=np.mean(a)                   # 每大組實驗平均值加總
  ax1.axvline(x=np.mean(a),ymin=0,ymax=1,c='red')    # 每大組子圖的平均值垂直線
  count1+=1
plt.show()
print(total/8)      # 每組實驗的總平均數 ~ 50次

print()
print("-----------------")

total=np.empty(100)      # 隨機產生 1*100 陣列，依照總數據的數量去輸入
start,end=0,0
for i in range(5):       # 共 5 大組實驗
  a=np.random.binomial(100,0.5,20)   # 每大組的 a 有 20 筆數據
  end=start+20
  total[start:end]=a      # 將產生的 a 存入 total 前 20 個位置裡，第二十個位置是 a[19]
  start=start+20          # 接下來一組 a 就是存入 [20:40]
print(total)              # 得到 20 組實驗，每組 100 次的總數據
plt.xlabel('numbers of positive')     # 直方圖 x 軸為正面次數
plt.ylabel('Frequency')               # y 軸為次數出現的頻率估計
plt.hist(total,bins=50)   # 畫出總數據直方圖， bins 代表箱子圖個數，個數越多，分佈越精細
plt.axvline(x=50,ymin=0,ymax=10,c='red')   # 平均值垂直線，位置 x=50次
plt.show()

print()
print("-----------------")

total=[]
total1=0
for i in range(50):                               # 四個恰好兩個消費者喜歡，機率是 0.2109
   a=sum(np.random.binomial(4,0.25,100)==2)/100   # True 的次數
   total.append(a)
   total1+=a
   print(a)
total2=total1/50
print("計算結果:",total1/50)
plt.hist(total,bins=5)
plt.axvline(x=0.2109,ymin=0,ymax=10,c='red')
plt.axvline(x=total2,ymin=0,ymax=10,c='black')
plt.show()

print()
print("-----------------")

total=[]
total1=0
for i in range(10000):         # 四個至少兩個消費者喜歡，機率 = 0.2617
  a=sum(np.random.binomial(4,0.25,100)>=2)/100
  total.append(a)
  total1+=a
  #print(a)
print("計算結果:",total1/10000)
total2=total1/10000
plt.hist(total,bins=5)
plt.axvline(x=0.2617,ymin=0,ymax=10,c='red')
plt.axvline(x=total2,ymin=0,ymax=10,c='black')
plt.show()

print()
print("-----------------超幾何分配")

import numpy.random as nr
all=[]
total=0                               # 這個問題機率是 0.3687
for i in range(1000):                 # 1000 次大實驗，共有 1000x100 個數據
   s=nr.hypergeometric(20,15,5,100)   # 總人數 :35，男:20、女:15，隨機抽五人，裏頭的男生數
   #print('樣本為5個,5個裡面男生的數量:')
   #print(s)
   p=sum(s==3)/100                    # 恰好男生數是 3 的組數再做平均
   total+=p                           # 1000 次平均的總和
   all.append(p)                      # 每次平均都存進 all，共 1000 次
print(total/1000)                     # 100000 個中恰好男生數是 3 的數據平均
x=np.arange(1000)
plt.hist(all,bins=10)
plt.axvline(x=total/1000,ymin=0,ymax=1,c='red')
plt.show()

print()
print("-----------------常態分配")
x=np.random.normal(5,1,8)          # normal(平均值,標準差,size)
y=np.random.normal(5,2,8)
z=np.random.normal(5,0.1,8)
print(x)
print(y)
print(z)
a=np.arange(8)
plt.plot(a,x,c='r')
plt.plot(a,y,c='g')
plt.plot(a,z,c='b')
plt.show()

x=np.random.normal(5,1,10000)
y=np.random.normal(5,2,10000)
z=np.random.normal(5,3,10000)
fig=plt.figure()
ax1=fig.add_subplot(3,1,1)         # 3x1 個子圖當中的第 1 個
ax1.hist(x)                        # 子圖用直方圖呈現 x 數據的分布情況
ax1.title.set_text('1')            # 子圖名字
ax1=fig.add_subplot(3,1,2)         # 第 2　個子圖
ax1.hist(y)
ax1.title.set_text('2')
ax1=fig.add_subplot(3,1,3)         # 第 3 個子圖
ax1.hist(z)
ax1.title.set_text('3')
plt.show()


