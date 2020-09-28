import csv
import sys

sys.path.append('../')
import math
import time
from sys import stdout
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from WEVREG.WEVREG import WEVREG
from experiments.external_methods.WkNN.WkNN import WkNN


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def total_sum(y_pred, y_true):
    return y_pred.sum() / y_true.sum()


num_experiments = 100


model_name = ["WkNN", "WEVREG", "WEVREG1", "GB", "SVR", "RT", "KNN", "RF", "ANN", "LR"]

df = pd.read_csv("datasets/Boston.csv").drop(["id"],
                                                axis=1)
selections = [["Normal"], ["WEVREG", "WEVREG1", "WkNN", "RT", "RF", "GB", "SVR"]]

scaler = MinMaxScaler()
columns = df.columns




input = df.drop(['medv'], axis=1)
cols = input.columns
input = scaler.fit_transform(input.values)
input = pd.DataFrame(input, columns=cols)

target = df['medv'].values

def my_func(selects):
    weights_wevreg = []
    weights_wevreg1 = []
    weights_gb = []
    weights_svr = []
    weights_rf = []
    weights_rt = []
    weights_wknn = []
    for n in range(num_experiments):
        x_train_help, x_test_help, y_train, y_test = train_test_split(input, target, test_size=0.2, shuffle=True)
        for s in selects:
            fs_name = s
            from pathlib import Path
            file_name = "results/boston_results_{}.csv".format(fs_name)
            my_file = Path(file_name)
            if not my_file.is_file():
                with open(file_name, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["model", "num_experiment", "num_feature", "train_time", "validation_time", "train_rmse", "train_mae",
                         "train_mape",
                         "train_r2", "eval_rmse", "eval_mae", "eval_mape", "eval_r2"])

            for i in range(x_test_help.shape[1]):

                if s != "Normal":
                    num_feature = i + 1
                    fs_name = s
                    weights = np.loadtxt("results/boston_weights_{}.csv".format(fs_name), delimiter=",")
                    weights = np.abs(weights)
                    weights = np.median(weights, axis=0)
                    sorted_indices = weights.argsort()[::-1]
                    top_5 = sorted_indices[:num_feature]
                    columns = df.columns
                    selected_columns = columns[top_5]
                    x_train = x_train_help[selected_columns].values
                    x_test = x_test_help[selected_columns].values
                else:
                    if i >= 1:
                        break
                    num_feature = -1
                    x_train = x_train_help.values
                    x_test = x_test_help.values

                try:
                    for name in model_name:
                        stdout.write("\rFS: {}, Model: {}, Num_experiment: {}                           ".format(s, name, n))
                        stdout.flush()
                        model = None
                        if name == "WEVREG":
                            model = WEVREG(n_neighbors=20, max_iter=30, learning_rate=0.1, optimizer='adam', scale=False,
                                                columns=cols)
                        elif name == "WEVREG1":
                            model = WEVREG(n_neighbors=20, max_iter=30, learning_rate=0.1, optimizer='adam',
                                                scale=False,
                                                columns=cols,
                                                p=1)
                        elif name == "GB":
                            model = GradientBoostingRegressor(n_estimators=100, max_depth=4)
                        elif name == "WkNN":
                            model = WkNN(n_neighbors=20, max_iter=30, learning_rate=0.1, optimizer='adam', scale=False)
                        elif name == "SVR":
                            model = LinearSVR(C=1000.0, max_iter=1000, dual=False, loss='squared_epsilon_insensitive', random_state=int(time.time()))
                        elif name == "RT":
                            model = DecisionTreeRegressor()
                        elif name == "KNN":
                            model = KNeighborsRegressor(n_neighbors=20, weights='distance', n_jobs=9)
                        elif name == "RF":
                            model = RandomForestRegressor(n_estimators=100, max_depth=4)
                        elif name == "ANN":
                            model = MLPRegressor(hidden_layer_sizes=(50, 25, 10), max_iter=2000)
                        elif name == "GLM":
                            model = None
                        elif name == "LR":
                            model = LinearRegression()
                        elif name == "LASSO":
                            model = Lasso(alpha=.2)

                        start = time.time()
                        if name != "GLM":
                            model.fit(x_train, y_train)

                        else:
                            model = sm.GLM(endog=y_train, exog=x_train, family=sm.families.Gaussian())
                            model = model.fit()
                        train_time = time.time() - start

                        y_train_pred = model.predict(x_train)
                        start = time.time()
                        y_pred = model.predict(x_test)
                        pred_time = time.time() - start
                        if name == "WEVREG":
                            weights_wevreg.append(model.model.dimension_weights.detach().cpu().numpy())
                        elif name == "WEVREG1":
                            weights_wevreg1.append(model.model.dimension_weights.detach().cpu().numpy())
                        elif name == "SVR":
                            weights_svr.append(model.coef_)
                        elif name == "GB":
                            weights_gb.append(model.feature_importances_)
                        elif name == "RF":
                            weights_rf.append(model.feature_importances_)
                        elif name == "RT":
                            weights_rt.append(model.feature_importances_)
                        elif name == "WkNN":
                            weights_wknn.append(model.model.dimension_weights.detach().cpu().numpy())

                        rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pred))
                        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
                        mae_train = mean_absolute_error(y_train, y_train_pred)
                        r2_train = r2_score(y_train, y_train_pred)

                        rmse_val = math.sqrt(mean_squared_error(y_test, y_pred))
                        mape_val = mean_absolute_percentage_error(y_test, y_pred)
                        mae_val = mean_absolute_error(y_test, y_pred)
                        r2_val = r2_score(y_test, y_pred)

                        with open(file_name, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(
                                [name, n, num_feature, train_time, pred_time, rmse_train, mae_train, mape_train, r2_train, rmse_val, mae_val,
                                 mape_val, r2_val])
                except Exception as e:
                    raise e

    if s == "Normal":
         weights_wevreg = np.array(weights_wevreg)
         weights_wevreg1 = np.array(weights_wevreg1)
         weights_gb = np.array(weights_gb)
         weights_svr = np.array(weights_svr)
         weights_rf = np.array(weights_rf)
         weights_rt = np.array(weights_rt)
         weights_wknn = np.array(weights_wknn)
         np.savetxt("results/boston_weights_{}.csv".format("WEVREG"), weights_wevreg, delimiter=",")
         np.savetxt("results/boston_weights_{}.csv".format("WEVREG1"), weights_wevreg1, delimiter=",")
         np.savetxt("results/boston_weights_{}.csv".format("GB"), weights_gb, delimiter=",")
         np.savetxt("results/boston_weights_{}.csv".format("SVR"), weights_svr, delimiter=",")
         np.savetxt("results/boston_weights_{}.csv".format("RF"), weights_rf, delimiter=",")
         np.savetxt("results/boston_weights_{}.csv".format("RT"), weights_rt, delimiter=",")
         np.savetxt("results/boston_weights_{}.csv".format("WkNN"), weights_wknn, delimiter=",")
        
for s in selections:
    my_func(s)


