import KNN_canner_1
from sklearn.neighbors import KNeighborsClassifier
data1=KNN_canner_1.dataCancer.data             # 數據
target1=KNN_canner_1.dataCancer.target         # 標籤
print(target1)
print('Q 那要怎麼分類才是最佳?')
from sklearn.model_selection import GridSearchCV,train_test_split  # 最佳參數、分類模擬器
params1={'n_neighbors':[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]}
X_train,X_test,y_train,y_test=train_test_split(data1,target1,test_size=0.3)
grid1=GridSearchCV(KNeighborsClassifier(),params1,verbose=3)   # 從 params1 挑出最優參數
print('開始訓練')
grid1.fit(data1,target1)
print('顯示最佳參數組合')
print(grid1.best_params_)
print('顯示最佳的函數內容')
print(grid1.best_estimator_)