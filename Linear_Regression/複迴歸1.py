import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
waist_heights = np.array([[67,160], [68,165], [70,167],        # 腰圍、身高數據
                          [65,170], [80,165], [85,167],
                          [78,178], [79,182], [95,175],
                          [89,172]])

weights = np.array([50, 60, 65, 65,70, 75, 80, 85,90, 81])     # 體重
X = pd.DataFrame(waist_heights, columns=["Waist", "Height"])   # 腰圍、身高的 Dataframe
target = pd.DataFrame(weights, columns=["Weight"])             # 體重的 series
y = target["Weight"]
lm = LinearRegression()

lm.fit(X, y)      # 由 腰圍、身高 得出複迴歸的參數
new_waist_heights = pd.DataFrame(np.array([[66, 164], [82, 172]]))   # 測試組
predicted_weights = lm.predict(new_waist_heights)      # 模型測試
print(predicted_weights)      # 測試組的體重預估