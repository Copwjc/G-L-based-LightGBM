from lightgbm import LGBMRegressor
import pandas as pd
import random
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

ordered = 0.8
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

X = pd.read_csv('x.csv', index_col = [0])
Y = pd.read_csv('y.csv', index_col = [0])
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


def objective(trial):
	pruning_callback = optuna.integration.LightGBMPruningCallback(trial, mse)
	params = {
		'verbose': -1,
        'boosting_type':trial.suggest_categorical('boosting_type',['gbdt']),
#		"subsample": trial.suggest_float("subsample", 0.1, 1.0),
#		"colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 10, 500),
	}
	model = LGBMRegressor(**params,objective = custom_train, n_estimators = 400,learning_rate = 0.036,max_depth = 16 ,reg_alpha = 6.19287469865381E-05, reg_lambda =4.55298928390744,min_child_samples = 6)
	model.fit(x_train, y_train)
	pred_y = model.predict(x_valid)
	score = mae(y_valid, pred_y)
	return score

study = optuna.create_study(direction="minimize", pruner = optuna.pruners.MedianPruner)

study.optimize(objective, n_trials = 300)

print("Number of finished trials: ({})".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
	print("    {}: {}".format(key, value))
    