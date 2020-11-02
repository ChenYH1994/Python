import 複迴歸1
import numpy as np
import pandas as pd
waist_heights=複迴歸1.waist_heights
weights=複迴歸1.weights
print("我們想要預估下面waist and heights的weights")
new_waist_heights=pd.DataFrame(np.array([[66,164],[82,172]]))
print(new_waist_heights)
print('依上述資料預估出兩者的體重')
predicted_weights=複迴歸1.lm.predict(new_waist_heights)
print(predicted_weights)

x=waist_heights[:,0]    # 所有腰圍
y=waist_heights[:,1]    # 所有身高

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x,y,weights,c='r',marker='o')
ax.set_xlabel('waist')     # 腰圍
ax.set_ylabel('heights')   # 身高
ax.set_zlabel('weights')   # 體重
plt.show()

fig=plt.figure()           # 紅色是訓練組，藍色是測試組
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x,y,weights,c='r',marker='o')

x1=np.array(new_waist_heights[0])
y1=np.array(new_waist_heights[1])
ax.scatter(x1,y1,predicted_weights,c='b',marker='o')
ax.set_xlabel('waist')     # 腰圍
ax.set_ylabel('heights')   # 身高
ax.set_zlabel('weights')   # 體重
plt.show()