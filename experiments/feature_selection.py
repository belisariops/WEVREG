import time

import numpy as np
from sklearn import datasets
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFE, f_regression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
import csv
import sys
sys.path.append('../')

from WEVREG.WEVREG import WEVREG
from datetime import datetime

from experiments.external_methods.WkNN.WkNN import WkNN

examples = [25, 50, 75, 100]
repetitions = 100
n_neighbors = 20
epochs = 100
n_features = 50
min = 0
max = 1
file_name = "fs_experiment.csv"
file_colums = ["function", "model_name", "n_examples", "success", "success_var", "precision", "total_time"]
np.random.seed(int(datetime.now().timestamp()))
models = ["WEVREG_1", "WEVREG_2", "GB", "RF", "WkNN", "Corr", "RT"]


experiments = [
    {
    "num_features": 1,
    "function_name": "square",
    "func": lambda X: X[:, 0]
    },
     {
     "num_features": 1,
     "function_name": "sin",
     "func": lambda X: np.sin(2 * np.pi * X[:, 0] + np.pi / 2)
     },
     {
     "num_features": 2,
     "function_name": "sin + x_2",
     "func": lambda X: np.sin(2 * np.pi * X[:, 0] + np.pi / 2) + X[:, 1]
     },
     {
     "num_features": 2,
     "function_name": "xor_like",
     "func": lambda X: np.sin(2 * np.pi * X[:, 0]) * np.sin(2 * np.pi * X[:, 1])
     },
    {
    "num_features": 2,
    "function_name": "xor",
    "func": lambda X: np.logical_xor(X[:, 0], X[:, 1])
    },
{
    "num_features": 4,
    "function_name": "friedman",
    },
]
def run_model(model_name, X, Y, num_important_features):
    indices = None
    if model_name == "WEVREG_2":
        model = WEVREG(n_neighbors=n_neighbors, max_iter=epochs, learning_rate=0.1, scale=False, optimizer='adam', columns=["x_0", "x_1", "x_2", "x_3", "x_4"])
        model.fit(X, Y)
        importance = model.model.dimension_weights.detach().cpu().numpy()
        indices = (-importance).argsort()[:]
    elif model_name == "WEVREG_1":
        model = WEVREG(n_neighbors=n_neighbors, max_iter=epochs, learning_rate=0.1, scale=False, optimizer='adam', columns=["x_0", "x_1", "x_2", "x_3", "x_4"], p=1)
        model.fit(X, Y)
        importance = model.model.dimension_weights.detach().cpu().numpy()
        indices = (-importance).argsort()[:]
    elif model_name == "WkNN":
        model = WkNN(n_neighbors=n_neighbors, max_iter=epochs, learning_rate=0.1, scale=False, optimizer='adam')
        model.fit(X, Y)
        importance = model.model.dimension_weights.detach().cpu().numpy()
        indices = (-importance).argsort()[:]
    elif model_name == "Corr":
        data = np.hstack((X, np.expand_dims(Y, axis=1)))
        corr = np.corrcoef(data, rowvar=False)[-1, :]
        indices = (-corr).argsort()[:]
    elif model_name == "BaseLine":
        indices = np.array(range(X.shape[1]))
    elif model_name == "L1-SVR":
        model = LinearSVR()
        select_model = SelectFromModel(model)
        select_model.fit(X, Y)
        importance = select_model.estimator_.coef_
        indices = (-importance).argsort()[:]
    elif model_name == "f-score":
        importance = f_regression(X, Y)
        indices = (-importance[0]).argsort()[:]
    elif model_name == "RF":
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, Y)
        importance = model.feature_importances_
        indices = (-importance).argsort()[:]
    elif model_name == "GB":
        model = RandomForestRegressor(n_estimators=200)
        model.fit(X, Y)
        importance = model.feature_importances_
        indices = (-importance).argsort()[:]
    elif model_name == "RT":
        model = DecisionTreeRegressor()
        model.fit(X, Y)
        importance = model.feature_importances_
        indices = (-importance).argsort()[:]
    return indices[:5]


with open(file_name, 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(file_colums)

for model_name in models:
    for exp in experiments:
        for e in examples:
            count = 0
            results = []
            start = time.time()
            for i in range(repetitions):
                num_important_features = exp["num_features"]
                function_name = exp["function_name"]

                if function_name == "friedman":
                    X, Y = datasets.make_friedman1(n_samples=e, n_features=n_features, noise=0.02, random_state=None)

                else:
                    if function_name != "xor":
                        X = np.random.uniform(min, max, size=(e, n_features))
                        Y = exp["func"](X)
                        noise = np.random.normal(0, 0.02, len(Y))
                        Y = Y + noise
                    else:
                        X = np.random.randint(0, 2, size=(e, n_features))
                        Y = exp["func"](X)

                perm = np.random.permutation(n_features)
                important_indices = perm[:num_important_features]
                idx = np.empty_like(perm)
                idx[perm] = np.arange(len(perm))
                X[:] = X[:, idx]

                indices = run_model(model_name, X, Y, num_important_features)

                r_s = 0
                indices = indices[: 5]
                for i in important_indices:
                    if i in indices:
                        r_s += 1

                all_detected = True

                for j in range(len(important_indices)):
                    if important_indices[j] not in indices:
                        all_detected = False
                if all_detected:
                    count += 1
                r_t = num_important_features
                i_s = len(indices) - r_s
                i_t = n_features - r_t
                alpha = np.amin(np.array([0.5, r_t / i_t]))
                if all_detected:
                    results.append(1.0)
                else:
                    results.append(r_s / r_t + alpha * i_s / i_t)
            total_time = time.time() - start
            results = np.array(results)
            line = [function_name, model_name, e, results.mean(), np.var(results.mean()), (count / repetitions), total_time]
            with open(file_name, 'a') as file:
                writer = csv.writer(file, delimiter=',')
                writer.writerow(line)
