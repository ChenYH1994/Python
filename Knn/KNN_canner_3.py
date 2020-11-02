import KNN_canner_1
from sklearn.neighbors import KNeighborsClassifier
data1=KNN_canner_1.dataCancer.data
target1=KNN_canner_1.dataCancer.target
print(target1)
print('Q 那要怎麼分類才是最佳?')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data1,target1,test_size=0.3)
print('開始訓練')
model=KNeighborsClassifier(n_neighbors=9)
print(X_train)
print()
print(y_train)
model.fit(X_train,y_train)       # 進行訓練
print('進行預測')
predicted=model.predict(X_test)
print(predicted)
print('顯示真實資料')
print(y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predicted))    # 答對率
print('也可以找測試的特徵X_test與測試的目標y_test,還有預測predicted進行圖表呈現')