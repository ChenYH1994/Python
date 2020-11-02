import class_2
credit=class_2.credit
print('查看信用好壞的統計數')
credit_counts=credit['bad_credit'].value_counts()
print(credit_counts)
print('0代表信用良好 1代表信用不好')