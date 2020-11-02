import matplotlib.pyplot as plt
import numpy as np
import itertools
'''
itertools.product:產生多個列表和迭代器的笛卡爾積
'''

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    '''
    imshow(cm,cmap=none):像素表
    cm:要繪製的圖像或數據，其中cm的數據對應顏色數據
    cmap:顏色圖譜
    interpolation='nearest': 
    '''

    plt.title(title)

    plt.colorbar()
    '''漸變色條'''

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        '''
        plt.text:在指定的位置加上文字
        horizontalalignment: 選擇垂直對齊的方式
        color: 如果是深藍色，則文字顯示白色；如果是淺藍色，則文字顯示黑色
        '''

    plt.tight_layout()
    '''
    tight_layout:
    主要用於自動調整繪圖區的大小及間距，
    使所有的繪圖區及其標題、坐標軸標籤等都可以不重疊的完整顯示在畫布上。
    '''

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
