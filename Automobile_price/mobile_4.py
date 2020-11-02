import matplotlib.pyplot as plt
import mobile_3
print('4-長條圖顯示')
auto_prices=mobile_3.auto_prices
cols3=mobile_3.cols3
print(len(cols3))
#print(cols3[1])
cols4=['make','body_style','num_of_cylinders']
print('十個欄位,我們只取三個')
#fig=plt.figure(figsize=(12,9))
n=1
for cols in cols4:
    counts=auto_prices[cols].value_counts()
    counts.plot.bar(color='b')      # 利用長條圖顯示各類資料的數量
    plt.title(cols)
    plt.show()                      # 發現光這三項，資料的數量差異很大! 不適合機器學習的分類
    n+=1
print('以上是字串類型的資料，所以可以用長條圖統計，但是數值資料就無法，只能用直方圖')