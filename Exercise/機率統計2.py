import numpy as np
import matplotlib.pyplot as plt
x=np.random.normal(200,10,800)
#normal(平均值,標準差,size)
print(x)
plt.hist(x)
plt.axvline(x=210,c='r')
plt.axvline(x=190,c='r')
plt.axvline(x=220,c='g')
plt.axvline(x=180,c='g')
plt.axvline(x=230,c='black')
plt.axvline(x=170,c='black')
#plt.plot(a,x,c='r')
plt.show()

print()
print('-'*70)

import scipy.stats as st
print("信賴水準、常態分配下的面積 所對應的Z分數:",st.norm.ppf(.95))
print("由Z分數推導出信賴水準或常態分配下的面積 :",st.norm.cdf(1.64))

