import pandas as pd
import KNN0
iris=KNN0.iris
print('2-轉換為dataframe')
print('將target轉換為target_name')
species=[iris.target_names[x] for x in iris.target]  # 0、1、2 依序有對應的 target_name
print(species)
print('將iris.data轉換為dataframe')
iris2=pd.DataFrame(iris.data,columns=['sepal_length','sepal_width','petal_length','petal_width'])
print(iris2)
print('產生datafrane後就可以加入欄位')
iris2['species']=species         # 新增各組數據的種類名
iris2['level']=iris.target       # 新增種類代號 0、1、2
print(iris2)