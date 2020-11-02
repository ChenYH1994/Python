import numpy as np
print('---------------索引值搜尋')
a=np.arange(10,16)
print(a[0])           # a 的第一個元素的索引值是 0
print(a[-1])          # 索引值<0 代表倒數第幾個， -1 就是倒數第一個
print(a[[1,3,4]])     # 也可以同時取出好幾個指定元素
print('---------------')

print(a[1:5])         # 索引值 1~4 的元素
print(a[1:5:2])       # 從 [1] 開始取，接著取彼此間隔差 2 的元素

print('---------------')

A=np.arange(10,19).reshape(3,3)
print(A[1,2])         # 取出 A 中 第 2 列、第 3 行 的元素
print()
print(A[0,:])         # 取出第 1 列
print()
print(A[:,0])         # 取出第 1 行
print()
print(A[0:2,0:2])     # 逗號左邊表示 列的範圍，右邊是 行的範圍，所以取出左上角 2x2 的矩陣
print()
print(A[[0,2],0:2])   # 由第 1、 3列的 [0]、[1] 元素構成的矩陣

print('---------------迴圈遍歷，包含搭配函數計算')

for item in A.flat:   # 先左至右，然後換列，依序遍歷出所以 A 的元素
    print(item)
print()
print(np.apply_along_axis(np.mean,axis=0,arr=A)) # 每一行在平均函數上作用，當然函數可以自取。
print(np.apply_along_axis(np.mean,axis=1,arr=A)) # 每一列在平均函數上作用
print()

print('---------------條件判定')

A=np.random.random((4,4))
print(A)
print()
print(A<0.5)       # 如果元素 < 0.5，則在那個位子印出 True，否則印出 Flase
print()
print(A[A<0.5])    # 取出 A 中 < 0.5 的元素形成陣列

print('---------------形狀變換')

a=np.random.random(12)
print(a)
print()
a.shape=(3,4)      # 直接改變 a 的形狀
print(a)
print()
a=a.ravel()        # 恢復成一維陣列
print(a)
print()
a.shape=(3,4)
print(a)
print()
a.transpose()      # a 轉置
print(a)

print('---------------陣列連接')

C=np.ones((3,3))
D=np.zeros((3,3))
v=np.vstack((C,D))    # 將 C, D 垂直連接
print(v)
print()
h=np.hstack((C,D))    # 將 C, D 水平連接
print(h)
print()
a=np.array([0,1,2])
b=np.array([3,4,5])
c=np.array([6,7,8])
V=np.column_stack((a,b,c)) # 將 a,b,c 以直的方方向放入矩陣的列
print(V)
print()
H=np.row_stack((a,b,c))    # 將 a,b,c 以橫的方式放入矩陣的列
print(H)

print('---------------陣列切分')

A=np.arange(16).reshape((4,4))
print(A)
print()
[B,C]=np.hsplit(A,2)       # 將每列分成兩半，將 A 分成左右兩個矩陣
print(B)
print()
print(C)
print()
[B,C]=np.vsplit(A,2)       # 將每行分成兩半，將 A 分成上下兩個矩陣
print(B)
print()
print(C)
print()

[B,C,D]=np.split(A,[1,3],axis=1)    # 將列的第 1, 2 行與其他部分做劃分，分成三個部分。水平劃分依此類推
print(B)
print()
print(C)
print()
print(D)
print()

print('---------------副本與視圖')

# 給一個陣列，無論將它儲存在另一個變數或者做切分...位址都不變。
# 想要生成一個位置不同的副本，要用 copy() 函數

a=np.array([1,2,3,4])
c=a.copy()
print('a =',a)
print('c =',c)
a[0]=13
print('a變成 =',a)
print('c仍然不變 =',c)
print('所以用了 copy() 函數，生成了與 a 位址不同的副本 c')

print('-'*15,'文件讀寫')
print()
data=np.random.random(12)
print(data)
data.shape=(4,3)
print(data)
np.save('Saved_data',data)






