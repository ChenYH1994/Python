import numpy as np
a=np.arange(4)
print(a)
print('---------------')

b=np.arange(4,8)
print(b)

print('---------------')

print(a*np.sin(b))     # Numpy 的計算可以搭配它的內建函數
print(a*np.sqrt(b))

print('---------------')

a+=2                   # 如果想連續的運算，可以使用 +=、-=、*=、...，但這會改變原本的 a
print(a)
a*=3
print(a)               # 算到這裡 a 從 [ 0 1 2 3 ] 變成 [ 6 9 12 15 ]

print('---------------')

a=np.array([3.3,4.5,1.2,5.7,0.3])
print(a.sum())         # 元素和
print(a.min())
print(a.max())
print(a.mean())        # 平均值
print(a.std())         # 標準差

print('---------------')

A=np.arange(0,9).reshape(3,3)  # 用 np.arange.reshape 生成矩陣 A (多維陣列)
print(A)
B=np.ones((3,3))
print(B)

print('---------------')

print(A*B)             # 注意 !! 這邊的 * 是元素對元素相乘的意思，不是矩陣相乘
print()
print(np.dot(A,B))     # 這裡 np.dot 才是矩陣相乘
print()
print(A.dot(B))        # 矩陣相乘的另一個寫法




