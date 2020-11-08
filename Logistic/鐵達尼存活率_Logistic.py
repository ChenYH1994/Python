import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train1=pd.read_csv('titanic_train.csv',encoding='utf-8-sig')
test1=pd.read_csv('titanic_test.csv',encoding='utf-8-sig')
train1.to_html('titan_train.html')
test1.to_html('titan_test.html')
print('了解欄位型態')
print(train1.dtypes)
print()
print('查看是否有遺失值')
print(train1.isnull().sum())
print()
print('透過圖表查看train遺失值')
sns.heatmap(train1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
print('透過圖表查看test遺失值')
sns.heatmap(test1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
print('顯示train資料集資料:',train1.shape)
print('顯示test資料集資料:',test1.shape)

print()
print('對於羅吉斯迴歸不能有遺失值存在,我們必須得處理遺失值')
print()
print('從熱力圖看到 Cabin 太多遺失值，所以移除此欄位')
train1.drop('Cabin',axis=1,inplace=True)
test1.drop('Cabin',axis=1,inplace=True)
print()
print('只要有遺失值,那筆紀錄就不要')
train1.dropna(inplace=True)
test1.dropna(inplace=True)
print()
print('透過圖表查看train遺失值')
sns.heatmap(train1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
print('透過圖表查看test遺失值')
sns.heatmap(test1.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
print()
print('顯示train資料集資料:',train1.shape)
print('顯示test資料集資料:',test1.shape)

print()
print('存活率是指0代表死亡 1代表存活')

'''長條圖'''
print('性別與存活率')
sns.countplot(x='Survived',hue='Sex',data=train1)
plt.show()

print()
print('票務等級與存活率')
sns.countplot(x='Survived',hue='Pclass',data=train1)
plt.show()

print()
print('登船港口與存活率')
sns.countplot(x='Survived',hue='Embarked',data=train1)
plt.show()

print()
print('年齡與存活率')
sns.countplot(x='Survived',hue='Age',data=train1)
plt.show()

print()
print('票務等級與港口')
sns.countplot(x='Embarked',hue='Pclass',data=train1)
plt.show()


print()
#print('兄弟姊妹與配偶')
#sns.countplot(x='Survived',hue='SibSp',data=train1)
#plt.show()
print('父母與子女')
sns.countplot(x='Survived',hue='Parch',data=train1)
plt.show()

print()
print('家庭存活率')
train1['family']=train1['Parch']+train1['SibSp']
sns.countplot(x='Survived',hue='family',data=train1)
plt.show()

print()
print('刪除不需要的欄位以及object型態欄位，只留數值型態欄位')
train2=train1.drop(['PassengerId','Survived','Name'],axis=1)

print()
'''
1. 以下四個欄位為字串,若要進行羅吉斯迴歸,必須做encoder動作。
     • Embarked, Ticket, Sex, Name
2. Sex欄位資料失去平衡,男性死亡率是女性的7倍

3. Embarked欄位為什麼S港死亡率很高?
     後續進行票務等級的分析,發現S港的是票價最便宜的旅客。
'''

print('train1_X代表訓練的內容')

'''特徵'''
train1_X=train2.drop(['Sex','Ticket','Embarked','family'],axis=1)
print(train1_X.head())

print()
print('train1_y代表訓練的目標,就是存活與否')

'''標籤'''
train1_y=train1['Survived']
print(train1_y.head())

'''開始訓練'''
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')

model.fit(train1_X,train1_y)

print()
print('開始進行預估')


print('test資料一樣要刪除不需要的欄位以及object型態欄位')
print('了解欄位的資料型態')
print(test1.dtypes)

print('test1_y代表預測的目標,就是這個乘客的ID')
test1_y=test1['PassengerId']

print('test1_X代表想要進行預測的內容')
test1_X=test1.drop(['PassengerId','Name','Sex','Ticket','Embarked'],axis=1)
print(test1_X)

print('開始預測')
pred=model.predict(test1_X)

Survived=[(ID,pred) for ID,pred in zip(test1_y,pred)]

'''印出 乘客ID 與 是否存活'''
[print('乘客ID:{}    預測是否存活:{:2}'.format(*pair)) for pair in Survived]

'''儲存成 csv 檔'''
finish1=pd.DataFrame({"乘客ID":test1_y,"存活率":pred})
finish1.to_csv('finish.csv',encoding='utf-8-sig',index=False)

