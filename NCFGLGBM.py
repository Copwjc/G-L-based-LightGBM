import lightgbm as lgb
import pandas as pd
import random
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
ordered = 0.6
k = 10

def coefficient(k,ordered):
    t = 1
    for i in range(0,k):
        t *= (ordered-i)
    t *= (-1)**k
    t = t/gamma(k+1)
    return t

#计算系数
coef = []
for i in range(0,k):
    coef.append(coefficient(i,ordered))

X = pd.read_csv('dataset/feature.csv', index_col = [0])
Y = pd.read_csv('dataset/price.csv', index_col = [0])
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=4)
test_num = x_valid.shape[0]
train_num = x_train.shape[0]
coef = [coef]*train_num

def f(x):
    return (x**2)/2

def ncasual(coef,x,f):
        global k
        p4 = []
        p5 = []
        for i in range(0,k):
                temp1 = f(x+i)
                temp2 = f(x-i)
                p4.append(temp1)
                p5.append(temp2)
        out = (sum(np.array(coef).T*(np.array(p5)-np.array(p4))))
        return out

def custom_train(y_true, y_pred):
    global t, epsilon
    residual = (y_true - y_pred).astype("float")
    grad = -ncasual(coef, residual, f)
    hess = np.where(residual<0, 1.0, 1.0)
    return grad, hess

# default lightgbm model with sklearn api
gbm = lgb.LGBMRegressor() 



evals_results = {}
# updating objective function to custom
# default is "regression"
# also adding metrics to check different scores
gbm.set_params(**{'objective': custom_train}, 
    metrics = ["mae", 'rmse'], 
    boosting_type = 'gbdt',
    learning_rate = 0.076,
    max_depth = 12,
    reg_alpha = 0.109160205318194,
    reg_lambda = 3.17246659541789,
    min_data_in_leaf = 4,
    num_leaves = 168,
    n_estimators = 400)

# fitting model 
gbm.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])

pred_y = gbm.predict(x_valid)
y_valid = y_valid.to_numpy().reshape(test_num,1)


a = gbm.evals_result_.pop('valid_0')

out = pd.DataFrame(a['rmse'])
out1 = pd.DataFrame(a['l1'])
out.to_csv('output/NCFGLGBM.csv')
out1.to_csv('output1/NCFGLGBM.csv')
             
from sklearn.metrics import mean_absolute_percentage_error as mape
print(f"MAE is {mean_absolute_error(y_valid, pred_y)}")
print(f"R2 is {r2_score(y_valid, pred_y)}")
print(f"mape is {mape(y_valid, pred_y)}")
print(f"rmse is {mean_squared_error(y_valid, pred_y)**(1/2)}")
 
# = pd.DataFrame(evals_result)
#a.to_excel('./1.xlsx')



#out1 = pd.DataFrame(pred_y)
#out.to_csv('output/fom.csv')
#out1.to_csv('output/fopred.csv')
#plt.plot(pred_y,label = 'pred')
#plt.plot(y_valid, label = 'true')
#plt.legend()
#plt.show()


# = pd.DataFrame(evals_result)
#a.to_excel('./1.xlsx')

    