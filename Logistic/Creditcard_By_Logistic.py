import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print('主題:信用卡詐騙檢測')
data=pd.read_csv('creditcard.csv')
print(data.head())

print('特徵 V1，V2，...，V28 是經過PCA轉換後的數據')
print('Amount 代表交易金額')
print('Class=0 表示該筆交易紀錄正常，Class=1 表示交易例外也就是被欺騙')
print()

print('想先了解正常跟欺詐的比例')
class1=data['Class'].value_counts()
print(class1)
print('以上數據顯示正常占絕大多數')
print()

'''圖視化'''
class1.plot(kind='bar')
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

'''查看是否有遺失值'''
print(data.isnull().sum())

print()
print('接下來，改進資料不平衡問題')
print()
print('方法1: 下取樣，讓正常跟欺詐一樣少')
print('方法2: 過取樣，假造欺詐樣本，使得數量跟正常樣本一樣多')
print('兩個方法都有疑慮。進行實驗比較，再看採用哪個方法。')
print()

print('下取樣法實驗:')

print('1.先對數值變動大的資料作標準化，像是 Amount 對於其他特徵，數值差距就大')
from sklearn.preprocessing import StandardScaler
data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Time','Amount'],axis=1)

print('2.資料劃分')
'''分開原始資料的 feature 跟 label'''
X=data.iloc[:,data.columns!='Class']    # 資料只有特徵，無標籤
Y=data.iloc[:,data.columns=='Class']    # 資料只有標籤
number_records_fraud=len(data[data.Class==1])   # 例外資料數量
print(X)

'''分開正常跟欺詐的index'''
fraud_indices=np.array(data[data.Class==1].index)
normal_indices=np.array(data[data.Class==0].index)

'''隨機選取跟欺詐同數量的正常的index'''
random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
random_normal_indices=np.array(random_normal_indices)

'''把正常跟欺詐的 index 拼接起來'''
under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])

'''用 index 找出特徵數據，共 984 筆的下取樣資料'''
under_sample_data=data.iloc[under_sample_indices,:]

'''分開 下取樣資料 的 feature 跟 label'''
X_undersample=under_sample_data.iloc[:,under_sample_data.columns!='Class']
Y_undersample=under_sample_data.iloc[:,under_sample_data.columns=='Class']

print('3.劃分訓練集、測試集，再從訓練集裡面劃分出驗證集，用來調參數')
from sklearn.model_selection import train_test_split

'''對原始的特徵跟標籤進行劃分'''
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

'''對下取樣資料進行劃分'''
X_train_us,X_test_us,Y_train_us,Y_test_us=train_test_split(X_undersample,Y_undersample,test_size=0.3,random_state=0)


print('4.用 L1 正規化懲罰、交叉驗證進行調參，其中正規化是為了避免過擬合')
print('5.模型評估:混淆矩陣')
from sklearn.linear_model import LogisticRegression
'''呼叫邏輯迴歸模組'''
from sklearn.metrics import confusion_matrix
'''混淆矩陣'''

from 交叉驗證 import printing_Kfold_scores
print()
best_c = printing_Kfold_scores(X_train_us,Y_train_us)


lr = LogisticRegression(C = best_c, penalty = 'l1',solver='liblinear')
lr.fit(X_train_us,Y_train_us.values.ravel())
y_pred_undersample = lr.predict(X_test_us.values)

# 計算混淆矩陣
print('代入 下取樣本 的答案與預測結果')
cnf_matrix = confusion_matrix(Y_test_us,y_pred_undersample)
print('混淆矩陣:')
print(cnf_matrix)
print()
print("召回率: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# 畫圖
from Confusion_Matrix_plt import plot_confusion_matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

print()
print('接著拿 原始資料 來做測試')
Y_pred=lr.predict(X_test.values)

# 計算混淆矩陣
cnf_matrix=confusion_matrix(Y_test,Y_pred)

print('原始資料測試的召回率:',cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

#繪圖
class_names=[0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

print()
print('發現到正常樣本被誤判成詐欺的比例極高!')
print('嘗試方法2:過取樣法')

from imblearn.over_sampling import SMOTE

credit_cards=pd.read_csv("creditcard.csv")   #重新讀取數據，因為之前原數據在下取樣時已經被更改

'''下面是把特徵 Amount 標準化的 code,它把 normAmount 放在最後一行，並且刪去了 Time 這一行'''

# print()
# print('對 Amount 做特徵標準化')
# from sklearn.preprocessing import StandardScaler
# credit_cards['normAmount']=StandardScaler().fit_transform(credit_cards['Amount'].values.reshape(-1,1))
# credit_cards=credit_cards.drop(['Time','Amount'],axis=1)

print()
print('這邊有個發現! 如果對 Amount 做標準化，則誤判率沒有比標準化前來得低')
columns=credit_cards.columns
''' 想訓練 Amount 標準化後的模型，下面請改成 len(columns)-2，這樣才是刪除倒數第 2 行的 Class '''
features_columns=columns.delete(len(columns)-1)   # 特徵數據的 column

features=credit_cards[features_columns]     # 特徵數據
labels=credit_cards['Class']                # 標籤數據

features_train, features_test, labels_train, labels_test = train_test_split(features,labels
                                                                            ,test_size=0.3,random_state=0)
oversampler=SMOTE(random_state=0)


'''利用 SMOTE 進行資料生成，使得正常跟詐欺樣本的資料數相等'''
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)
os_features = pd.DataFrame(os_features)
os_labels = pd.DataFrame(os_labels)
print()
print('過取樣後的樣本數:')
print('詐欺樣本數量:',len(os_labels[os_labels.values==1]))
print('正常樣本數量:',len(os_labels[os_labels.values!=1]))

'''一樣用交叉驗證得出最佳的懲罰力度'''
best_c = printing_Kfold_scores(os_features,os_labels)

'''進行訓練'''
lr = LogisticRegression(C = best_c, penalty = 'l1', solver='liblinear')
lr.fit(os_features,os_labels.values.ravel())

'''進行預測'''
y_pred1 = lr.predict(features_test.values)

''''計算並畫出混淆矩陣'''
cnf_matrix = confusion_matrix(labels_test,y_pred1)
np.set_printoptions(precision=2)

print("召回率:", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

print()
print('結論:針對這個專案')
print('1. 對於過取樣，先對 Amount 做標準化，對模型誤判率的改善沒有比較優。')
print('2. 過取樣比下取樣的整體預測結果好')

