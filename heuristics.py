import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# class BaseHeuristic:
#     def __init__(self, n=11, k=4):
#         self._n = n
#         self._k = k
#
#     def get_h_values(self, states):
#         states_as_list = [state.get_state_as_list() for state in states]
#         gaps = []
#
#         for state_as_list in states_as_list:
#             gap = 0
#             if state_as_list[0] != 1:
#                 gap = 1
#
#             for i in range(len(state_as_list) - 1):
#                 if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
#                     gap += 1
#
#             gaps.append(gap)
#
#         return gaps


class HeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(HeuristicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class LearnedHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k
        self._model = HeuristicModel(n)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states)
        with torch.no_grad():
            predictions = self._model(states_tensor).numpy()
        return predictions.flatten()

    def train_model(self, input_data, output_labels, epochs=100):
        input_as_list = [state.get_state_as_list() for state in input_data]
        inputs = np.array(input_as_list, dtype=np.float32)
        outputs = np.array(output_labels, dtype=np.float32)

        inputs_tensor = torch.tensor(inputs)
        outputs_tensor = torch.tensor(outputs).unsqueeze(1)  # Adding a dimension for the output

        for epoch in range(epochs):
            self._model.train()
            self._optimizer.zero_grad()

            predictions = self._model(inputs_tensor)
            loss = self._criterion(predictions, outputs_tensor)
            loss.backward()
            self._optimizer.step()

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()


class BootstrappingHeuristic(LearnedHeuristic):
    def __init__(self, n=11, k=4):
        super().__init__(n, k)

    def save_model(self):
        super().save_model('bootstrapping_heuristic.pth')

    def load_model(self):
        super().load_model('bootstrapping_heuristic.pth')
