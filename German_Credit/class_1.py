import pandas as pd
import class_0
import bank_note
print('1-整理欄位、欄位內容進行轉換')
credit=class_0.credit
print('用不到customer_id,所以將這column刪除')
credit.drop(['customer_id'],axis=1,inplace=True) # axis=1 代表行，inplace 代表是否要改變原來的值
credit.to_html('credit3.html')
code_list=bank_note.code_list
print(code_list)
print('將欄位內容進行轉換')
for dict1 in code_list:
    col = dict1[0]         # 欄位名稱
    group1 = dict1[1]      # 欄位的替代值
    credit[col] = [group1[x] for x in credit[col]]  #將原先行的值，用替代值替換
credit.to_html('credit4.html')