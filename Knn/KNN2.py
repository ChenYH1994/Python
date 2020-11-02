import matplotlib.pyplot as plt
import KNN1
iris2=KNN1.iris2
print('3-圖表分析')
print('後續將要進行顏色區分,將三種不同類型鳶尾花分別顯示')
from matplotlib.colors import ListedColormap
classes = ['setosa', 'versicolor', 'virginica']
colours = ListedColormap(['r','b','g'])
fig=plt.figure(figsize=(12,6))
ax1=fig.add_subplot(2,2,1)
ax2=ax1.scatter(iris2['sepal_length'],iris2['sepal_width'],c=iris2['level'],cmap=colours)
ax1.set_xlabel('sepal_length')
ax1.set_ylabel('sepal_width')
ax1.legend(handles=ax2.legend_elements()[0], labels=classes)  # 各名稱的顏色
ax1=fig.add_subplot(2,2,2)
ax2=ax1.scatter(iris2['petal_length'],iris2['petal_width'],c=iris2['level'],cmap=colours)
ax1.set_xlabel('petal_length')
ax1.set_ylabel('petal_width')
ax1.legend(handles=ax2.legend_elements()[0], labels=classes)
ax1=fig.add_subplot(2,2,3)
ax2=ax1.scatter(iris2['sepal_length'],iris2['petal_width'],c=iris2['level'],cmap=colours)
ax1.set_xlabel('sepal_length')
ax1.set_ylabel('petal_width')
ax1.legend(handles=ax2.legend_elements()[0], labels=classes)
ax1=fig.add_subplot(2,2,4)
ax2=ax1.scatter(iris2['petal_length'],iris2['sepal_width'],c=iris2['level'],cmap=colours)
ax1.set_xlabel('petal_length')
ax1.set_ylabel('sepal_width')
ax1.legend(handles=ax2.legend_elements()[0], labels=classes)
plt.show()