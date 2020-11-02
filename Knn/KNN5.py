import KNN4
X_train=KNN4.X_train
y_train=KNN4.y_train
X_test=KNN4.X_test
y_test=KNN4.y_test
print('6-終於要開始進行分類')
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
print('開始訓練')
model.fit(X_train,y_train)          # fit 可以把資料、答案一併帶入模型內訓練
print('開始預測')
predict1=model.predict(X_test)      # 用 predict 來預測
print('預測的結果')
print(predict1)
print('實際的結果')
print(y_test)
print('算準確率')
from sklearn.metrics import accuracy_score    # 把預測結果跟實際答案做比對，算出答對率
print(accuracy_score(y_test,predict1))
print('或者做方差，值越接近 1，說明預測愈準確')
print(model.score(X_test,y_test))
