import pandas as pd
import class_1
import bank_note
print('2-了解分類以及是否有分類不平衡')
credit=class_1.credit
print('查看數值資料的基礎統計量')
print(credit.describe())
credit5=credit.describe()
credit5.to_html('credit5.html')
print('查看字串資料的統計量')
print(credit.describe(include=['object']))
credit6=credit.describe(include=['object'])
credit6.to_html('credit6.html')
print('想了解是否有分類不平衡,這是字串資料,我們得把欄位找出來進行分析')

cols3=[
       'checking_account_status','credit_history',
       'purpose','savings_account_balance',
       'time_employed_yrs','gender_status',
       'other_signators','property',
       'other_credit_outstanding',
       'home_ownership','job_category',
       'telephone','foreign_worker',
       'bad_credit'
       ]    # 利用 credit6 複製貼上
for column in cols3:
    print('這是:',column)
    print(credit[column].value_counts())

import seaborn as sns
import matplotlib.pyplot as plt
print('3-以信用良好與否為X軸,進行各種資料分類與圖表顯示')
def plot_box(credit,cols,col_x='bad_credit'):    # 畫出信用度跟各資料的箱型圖
    for col in cols:
           sns.set_style('whitegrid')            # 設定圖的底色是白色
           sns.boxplot(col_x, col, data=credit)  # x: 信用度， y: 跟信用度有關聯的資料
           # auto_prices[col].plot.hist(bins=10)
           plt.ylabel(col)                       # 軸的名稱
           plt.show()
num_cols=['loan_duration_mo', 'loan_amount', 'payment_pcnt_income',
          'age_yrs', 'number_loans', 'dependents']  # 貸款期限、貸款額度、年齡...
plot_box(credit,num_cols)

'''
loan_duration_mo : 貸款期限
loan_amount : 貸款額度
payment_pcnt_income : 付款占個人所得的比例
age_yrs : 年齡
number_loans : 貸款次數
dependents : 家屬
'''


