import 線性回歸1
import matplotlib.pyplot as plt
new_temperatures = 線性回歸1.pd.DataFrame(線性回歸1.np.array([26, 30]))
predicted_sales = 線性回歸1.lm.predict(new_temperatures)         # 對溫度 26、30 預估銷售量
print(predicted_sales)
plt.scatter(線性回歸1.temperatures, 線性回歸1.drink_sales)  # 線性預測的繪點
regression_sales = 線性回歸1.lm.predict(線性回歸1.X)        # 對整個數據做線性預估
plt.plot(線性回歸1.temperatures, regression_sales, color="blue")
plt.plot(new_temperatures, predicted_sales,color="red", marker="o",
markersize=10)                                                        # 整個回歸線的圖
plt.show( )