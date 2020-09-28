import sys
import time
from functools import wraps

import numpy as np
import faiss
import torch
import warnings
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn
from joblib import dump, load

from WEVREG import utils
from WEVREG.EuclideanModel import EuclideanModel


class WEVREG:
    def __init__(self, n_neighbors=10, learning_rate=0.001, discounting_model='euclidean', max_iter=20, scale=False,
                 columns=None, optimizer='adam', p=2):
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.discounting_model = discounting_model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.column_names = None
        self.training_set = None
        self.target_set = None
        self.model = None
        self.nearest_neighbors = None
        self.optimizer = None
        self.n_chunks = 1
        self.loss = nn.MSELoss()
        self.rules = []
        self.mse = []
        self.mae = []
        self.r2_score = []
        self.relative_mae = []
        self.input_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scale = scale
        self.columns = columns
        self.old_x = None
        self.old_y = None
        self.deleted_vectors = False
        self.phi = None
        self.weights = None
        self.loss = torch.nn.MSELoss()
        self.optimizer_name = optimizer
        self.neighbors_index = None
        self.beta = None
        self.p = p

    def get_time(function):
        @wraps(function)
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = function(self, *args, **kwargs)
            end = time.time()
            self.times[function.__name__] = end - start
            print("Elapsed time {}: {}".format(function.__name__, end - start))
            return result

        return wrapper

    def fit(self, x, y, column_names=None, chunk_size=None, grad_desc=True, save=False, load=False, create_rules=False):
        self.column_names = column_names
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        if self.scale:
            x = self.input_scaler.fit_transform(x)
            y = self.target_scaler.fit_transform(y.reshape(-1, 1))[:, 0]

        if self.training_set is not None:
            self.add_vectors(x, y)
            self._gradient_descent(save=save, create_rules=create_rules, chunk_size=chunk_size)
            return
        self.training_set = x.astype(np.float32)
        self.target_set = y.astype(np.float32)

        self.neighbors_index = faiss.IndexFlatL2(np.size(self.training_set, 1))
        self.neighbors_index.add(np.ascontiguousarray(self.training_set))

        if self.n_neighbors >= len(self.training_set):
            self.n_neighbors = len(self.training_set)
            warnings.warn("The number of neighbors exceeds the size of the dataset, the number of neighbors "
                          "it's going to be the size of the dataset. ")

        if self.discounting_model == 'euclidean':
            self.model = EuclideanModel(self.training_set.shape[1], weights=self.weights, phi=self.phi, p=self.p)
            # self.model.phi = nn.Parameter(torch.tensor(self.beta))
        else:
            raise AttributeError(
                'The discounting model {} does not exists in the current implementation'.format(self.discounting_model))
        self.model = self.model.to(self.device)
        if self.optimizer_name == "rprop":
            self.optimizer = optim.Rprop(list(self.model.parameters()), lr=self.learning_rate)
        elif self.optimizer_name == "adam":
            self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(list(self.model.parameters()), lr=self.learning_rate)
        else:
            raise AttributeError('The optimizer {} does not exist in the current implementation. '
                                 .format(self.optimizer_name))

        if load:
            self.model = torch.load('model.torch')
        if grad_desc:
            self._gradient_descent(save=save, create_rules=create_rules, chunk_size=chunk_size)

    def _get_loo_nn(self, x, weights, train):
        n_neighbors = self.n_neighbors + 1 if train else self.n_neighbors
        self.neighbors_index = faiss.IndexFlat(np.size(self.training_set, 1), faiss.METRIC_L2)
        self.neighbors_index.add(np.ascontiguousarray((self.training_set * weights).astype(np.float32)))
        _, indices = self.neighbors_index.search(
            np.ascontiguousarray((x * weights).astype(np.float32)), n_neighbors)  # actual search
        if not train:
            response = indices
        else:
            response = []
            for position, ind in enumerate(indices):
                response.append(ind[ind != position][:self.n_neighbors])
            response = np.array(response) - 1
        return response

    def forward(self, x, train=False):
        indices = self._get_loo_nn(x, self.model.dimension_weights.detach().cpu().numpy(), train)
        x_neighbors = torch.tensor(np.take(self.training_set, indices.astype(int), axis=0).astype(np.float32)).to(
            self.device)
        y_neighbors = torch.tensor(np.take(self.target_set, indices, axis=0)).to(self.device)
        mass_tensor, domain_mass = self._get_masses(torch.tensor(x.astype(np.float32)).to(self.device), x_neighbors)
        min_y = y_neighbors.min(dim=1)[0]
        max_y = y_neighbors.max(dim=1)[0]
        prediction = torch.sum(mass_tensor * y_neighbors, dim=1) + domain_mass * (max_y + min_y) / 2

        uncertainty = domain_mass * torch.abs((max_y - min_y) / 2)
        return prediction, uncertainty

    def get_explanation(self, x):
        x = self.input_scaler.transform(x)
        indices = self._get_loo_nn(x, self.model.dimension_weights.detach().cpu().numpy(), train=False)
        x_neighbors = torch.tensor(np.take(self.training_set, indices.astype(int), axis=0).astype(np.float32)).to(
            self.device)
        y_neighbors = torch.tensor(np.take(self.target_set, indices, axis=0)).to(self.device)
        mass_tensor, domain_mass = self._get_masses(torch.tensor(x.astype(np.float32)).to(self.device), x_neighbors)
        min_y = y_neighbors.min(dim=1)[0]
        max_y = y_neighbors.max(dim=1)[0]
        prediction = torch.sum(mass_tensor * y_neighbors, dim=1) + domain_mass * (max_y + min_y) / 2
        upper = torch.sum(mass_tensor * y_neighbors, dim=1) + domain_mass * max_y

        weights = self.model.dimension_weights.detach().cpu().numpy()
        sorted_weight_indices = np.abs(weights).argsort()[::-1][:5]
        x_transformed = self.input_scaler.inverse_transform(x_neighbors[0].detach().cpu().numpy())
        min_neigh_values = x_transformed.min(axis=0)
        max_neigh_values = x_transformed.max(axis=0)
        rules = []
        for i in sorted_weight_indices:
            if min_neigh_values[i] == max_neigh_values[i]:
                rule = "{} = {}".format(self.columns[i].upper(), "%.2f" % min_neigh_values[i])
            else:
                rule = "{} <= {} <= {}".format("%.2f" % min_neigh_values[i], self.columns[i].upper(),
                                               "%.2f" % max_neigh_values[i])
            rules.append([rule, np.abs(weights[i])])
        x_transformed = x_transformed[:, sorted_weight_indices]
        y_neighbors = self.target_scaler.inverse_transform(y_neighbors.detach().cpu().numpy())
        prediction = self.target_scaler.inverse_transform(prediction.detach().cpu().numpy().reshape(-1, 1))[:, 0]
        upper = self.target_scaler.inverse_transform(upper.detach().cpu().numpy().reshape(-1, 1))[:, 0]
        uncertainty = upper - prediction
        columns = list(self.columns[sorted_weight_indices])
        columns.append("target")
        response = {
            "input": self.input_scaler.inverse_transform(x),
            "output": prediction[0],
            "min_output": self.target_scaler.inverse_transform(min_y.cpu().numpy().reshape(-1, 1))[:, 0][0],
            "max_output": self.target_scaler.inverse_transform(max_y.cpu().numpy().reshape(-1, 1))[:, 0][0],
            "uncertainty": uncertainty[0],
            "rules": rules,
            "columns": columns,
            "neighbors": np.hstack((x_transformed, y_neighbors.reshape(-1, 1))),
            "neighbors_masses": mass_tensor.detach().cpu().numpy(),
            "target_mass": domain_mass.detach().cpu().numpy()[0]
        }
        return response

    def _gradient_descent(self, save, create_rules, chunk_size, batch_size=4096):
        self.model.train()
        min_loss = sys.maxsize
        weights = self.model.dimension_weights.clone()
        phi = self.model.phi.clone()
        weights_plot = []
        local_vars = ["Temperature_Exterior_Sensor", "Humedad_Exterior_Sensor", "Datetime", "Weather_Temperature",
                      "Day_Of_Week", "other"]
        for c in local_vars:
            weights_plot.append([1.0])
        x_axis = [0]
        for i in range(self.max_iter):
            acc_loss = 0
            x_axis.append(i + 1)
            other = 0
            for j in range(0, self.training_set.shape[0], batch_size):
                self.optimizer.zero_grad()
                predicted_y, _ = self.forward(self.training_set[j: (j + 1) * batch_size], train=True)
                loss = self.loss(torch.tensor(self.target_set[j: (j + 1) * batch_size])
                                 .to(self.device), predicted_y)
                loss.backward()
                acc_loss += loss.clone()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            if min_loss > float(acc_loss):
                min_loss = float(acc_loss)
                weights = self.model.dimension_weights.clone()
                phi = self.model.phi.clone()

        self.model.dimension_weights = torch.nn.Parameter(weights)
        self.model.phi = torch.nn.Parameter(phi)

        if save:
            torch.save(self.model, 'model.torch')

    def predict(self, x, return_uncertainty=False, chunk_size=None):
        self.model.eval()
        if self.scale:
            x = self.input_scaler.transform(x)
        with torch.no_grad():
            predictions, uncertainty = self.forward(x)
            y, uncertainty = predictions.cpu().numpy().reshape(-1, 1), \
                             uncertainty.cpu().numpy().reshape(-1, 1)
            if return_uncertainty:
                if self.scale:
                    return self.target_scaler.inverse_transform(y)[:, 0], \
                           self.target_scaler.inverse_transform(uncertainty)[:, 0]
                return y, uncertainty
            if self.scale:
                return self.target_scaler.inverse_transform(y)[:, 0]
            return y

    def _get_masses(self, x, k_nearest_x):
        x = x.unsqueeze(1)
        value_x = self.model(x, k_nearest_x)
        K = (1 - value_x.double()).prod(dim=1) + (value_x.double() * utils.get_prod(value_x).double()).sum(dim=1)
        mass_tensor = value_x.double() * utils.get_prod(value_x).double() / K.unsqueeze(1)
        domain_mass = torch.prod(1 - value_x.double(), dim=1) / K
        # Normalize massess
        total = (mass_tensor.sum(dim=1) + domain_mass)
        mass_tensor = mass_tensor / total.unsqueeze(1)
        domain_mass = domain_mass / total
        return mass_tensor.float(), domain_mass.float()

    def add_vectors(self, x, pred):
        self.neighbors_index.add(np.ascontiguousarray(x.astype(np.float32)))
        self.training_set = np.concatenate((self.training_set, x), axis=0)
        self.target_set = np.concatenate((self.target_set, pred), axis=0)

    def load(self, file_name, x_train, y_train):
        x = self.input_scaler.fit_transform(x_train.astype(np.float32))
        self.training_set = torch.tensor(x).to(self.device)
        self.target_set = torch.tensor(y_train.astype(np.float32)).to(self.device)
        self.model = torch.load('{}.torch'.format(file_name))
        self.input_scaler = load('{}.scaler'.format(file_name))
        self.nearest_neighbors = faiss.read_index('{}.faiss'.format(file_name))

    def save(self, file_name):
        torch.save(self.model, '{}.torch'.format(file_name))
        dump(self.input_scaler, '{}.scaler'.format(file_name))
        faiss.write_index(self.nearest_neighbors, '{}.faiss'.format(file_name))
