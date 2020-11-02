from sklearn.datasets import load_breast_cancer          # 癌症統計資料: 良性、惡性
import pandas as pd
dataCancer = load_breast_cancer()
print("第一步就是先了解有哪些資料")
print(dataCancer)
print(dataCancer.keys())
print("資料必須轉換為DataFrame")
df=pd.DataFrame(dataCancer['data'],columns=dataCancer['feature_names'])
df.to_html('df1.html')

