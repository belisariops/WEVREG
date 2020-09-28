import sys

import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, learning_rate=0.01, max_iter=50,
                 num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)

        self.linear = nn.Linear(hidden_layer_size, output_size)


        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.max_iter = max_iter
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self = self.to(self.device)
        self.hidden_cell = (nn.Parameter(torch.zeros(num_layers, 1, self.hidden_layer_size)).to(self.device),
                            nn.Parameter(torch.zeros(num_layers, 1, self.hidden_layer_size)).to(self.device))
        self.best_params = {}
        self.min_loss = sys.maxsize

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = torch.tanh(self.linear(lstm_out.view(len(input_seq), -1)))
        return predictions

    def fit(self, x, y):
        x_tensor = torch.tensor(x).float().to(self.device)
        y_tensor = torch.tensor(y).float().to(self.device).unsqueeze(1)

        for epoch in range(self.max_iter):
            self.optimizer.zero_grad()
            y_pred = self.forward(x_tensor)
            loss = self.criterion(y_pred, y_tensor)
            loss.backward()
            self.optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x).float().to(self.device)
            resp = self.forward(x_tensor).squeeze(1).cpu().numpy()
            return resp