import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from config import my_config
from xgboost import XGBRegressor
from scipy.stats import uniform as sp_uniform
from dataprocess import Textprocess
import matplotlib.pyplot as plt
import pickle


if __name__=="__main__":
    #获得数据
    T = Textprocess()
    tr_x, tr_y, te_x, te_y = T.getData()
    model = XGBRegressor(n_jobs=-1)
    #寻找超参数
    param_dist = {"learning_rate": [1E-1, 1E-2, 1E-3, 1E-4, 1E-5], "n_estimators": [100, 500, 1000], "max_depth":[5, 10, 20],
                  "reg_lambda":sp_uniform(0.1, 5.0)}
    n_iter_search = 256
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,scoring='r2',
                                       n_iter=n_iter_search, cv=5)
    sresult = random_search.fit(tr_x, tr_y)
    print("finish research hyper parameters")
    print(sresult)
    param = {"learning_rate":sresult.best_params_["learning_rate"], "n_estimators":sresult.best_params_["n_estimators"],
             "max_depth":sresult.best_params_["max_depth"], "reg_lambda":sresult.best_params_["reg_lambda"]}
    with open("parameters_jerk.txt", "w", encoding="utf-8") as f:
        f.write(str(param))
    f.close()
    #设置参数开始跑
    model = XGBRegressor(learning_rate=sresult.best_params_["learning_rate"], max_depth=sresult.best_params_["max_depth"], n_estimators=sresult.best_params_["n_estimators"],
                                      reg_lambda=sresult.best_params_["reg_lambda"], n_jobs=-1)
    model.fit(tr_x, tr_y)

    print(model.score(te_x, te_y))

    res = model.predict(te_x)

    model.save_model('./model/xgboost_jerk.json')

    result = []
    for i in range(len(te_y)):
        result.append([te_y[i], res[i]])

    result = pd.DataFrame(result)
    result.to_csv("./result/predict_jerk.csv", header=False, index=False, encoding="utf-8")
