import matplotlib.pyplot as plt
import KNN2
import seaborn
iris2=KNN2.iris2
print('4-圖表分析--分別用不同4個數據的圖表顯示')
def plot_iris(iris,col1,col2):
   seaborn.lmplot(x=col1,y=col2,data=iris,hue='species',fit_reg=False)
   plt.xlabel(col1)
   plt.ylabel(col2)
   plt.show()
plot_iris(iris2,'sepal_length','sepal_width')
plot_iris(iris2,'petal_length','petal_width')
plot_iris(iris2,'sepal_length','petal_width')
plot_iris(iris2,'petal_length','sepal_width')