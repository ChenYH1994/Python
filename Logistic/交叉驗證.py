import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
def printing_Kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)
         # 交叉驗證函數 KFold(等份數量，shuffle=False 表示不管重複幾次，劃分結果都一樣)

    # 正則化懲罰力道
    c_param_range = [0.01, 0.1, 1, 10, 100]

    # 待存放每個參數的平均召回率的空 DataFrame
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # k-fold表示K折的交叉驗證，這裡會得到兩個索引集合：訓練集=索引[0]，驗證集=索引[1]
    j = 0
    # 用不同的懲罰力道去實驗，從中找出最佳懲罰參數
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('正則化懲罰力道: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []

        # 選定參數，再執行交叉驗證
        for iteration, indices in enumerate(fold.split(x_train_data)):

            '''
            fold : 將客戶的數據均分成 n 組
            .split : 把均分的數據再分成訓練集和驗證集，然後返回索引生成器
            iteration = n (第 n 次實驗)
            indices = array[array[訓練集索引],array[驗證集索引]]
            '''

            # 用到 L1 正則化
            lr = LogisticRegression(C=c_param, penalty='l1',solver='liblinear')

            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            # 用驗證集預測
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)

            # 計算召回率
            recall_acc = recall_score(y_train_data.iloc[indices[1],:].values, y_pred_undersample)

            # 把交叉實驗的召回率做收集，最後再算平均
            recall_accs.append(recall_acc)

            #每次交叉實驗的結果展示
            print('Iteration ', iteration, ': 召回率 = ', recall_acc)

        # 計算完平均後，再丟到最開始的空DataFrame:results_table 裡面
        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        '''
        results_table:
           '懲罰力道'        '平均召回率'
        0     0.01      0.9577468977515414
        1     0.1       0.9033036329066049
        .      .                .
        .      .                .
        .      .                .
        '''
        j += 1
        print('')
        print('平均召回率 ', np.mean(recall_accs))
        print('')

    print()
    print('每個懲罰力道的實驗結果:')
    print(results_table)

    # 找到最好的参数，哪一个Recall高，自然就是最好的了。
    best_c = results_table.loc[results_table['Mean recall score'].astype('float32').idxmax()]['C_parameter']

    print()
    # 印出最好的结果
    print('*********************************************************************************')
    print('效果最好的模型所选参数 = ', best_c)
    print('*********************************************************************************')

    return best_c