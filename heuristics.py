import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class BaseHeuristic:
    def __init__(self, player_piece, ai_piece, rows, columns, empty, window_length):
        self.player_piece = player_piece
        self.ai_piece = ai_piece
        self.rows = rows
        self.columns = columns
        self.empty = empty
        self.window_length = window_length

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.player_piece
        if piece == self.player_piece:
            opp_piece = self.ai_piece

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(self.empty) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(self.empty) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(self.empty) == 1:
            score -= 4

        return score

    def score_position(self, state):
        score = 0

        # Score center column
        center_array = [int(i) for i in list(state.board[:, self.columns // 2])]
        center_count = center_array.count(state.player)
        score += center_count * 3

        # Score Horizontal
        for r in range(self.rows):
            row_array = [int(i) for i in list(state.board[r, :])]
            for c in range(self.columns - 3):
                window = row_array[c:c + self.window_length]
                score += self.evaluate_window(window, state.player)

        # Score Vertical
        for c in range(self.columns):
            col_array = [int(i) for i in list(state.board[:, c])]
            for r in range(self.rows - 3):
                window = col_array[r:r + self.window_length]
                score += self.evaluate_window(window, state.player)

        # Score positive sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                window = [state.board[r + i][c + i] for i in range(self.window_length)]
                score += self.evaluate_window(window, state.player)

        # Score negative sloped diagonal
        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                window = [state.board[r + 3 - i][c + i] for i in range(self.window_length)]
                score += self.evaluate_window(window, state.player)

        return score


class ConnectFourHeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(ConnectFourHeuristicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ConnectFourHeuristic:
    def __init__(self, rows=5, columns=5):
        self._rows = rows
        self._columns = columns
        self._input_dim = rows * columns
        self._model = ConnectFourHeuristicModel(self._input_dim)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def get_h_values(self, states):
        states_as_list = [state.get_state_as_list() for state in states]
        states = np.array(states_as_list, dtype=np.float32)
        states_tensor = torch.tensor(states)
        with torch.no_grad():
            predictions = self._model(states_tensor).numpy()
        return predictions.flatten()

    def score_position(self, state):
        return self.get_h_values([state])

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


class BootstrappingConnectFourHeuristic(ConnectFourHeuristic):
    def __init__(self, rows=5, columns=5):
        super().__init__(rows, columns)

    def save_model(self):
        super().save_model('bootstrapping_connect_four_heuristic.pth')

    def load_model(self):
        super().load_model('bootstrapping_connect_four_heuristic.pth')


# class HeuristicModel(nn.Module):
#     def __init__(self, input_dim):
#         super(HeuristicModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.dropout1 = nn.Dropout(0.25)
#         self.fc2 = nn.Linear(64, 32)
#         self.dropout2 = nn.Dropout(0.25)
#         self.fc3 = nn.Linear(32, 16)
#         self.fc4 = nn.Linear(16, 1)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.dropout1(x)
#         x = torch.relu(self.fc2(x))
#         x = self.dropout2(x)
#         x = torch.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
#
#
# class LearnedHeuristic:
#     def __init__(self, n=11, k=4):
#         self._n = n
#         self._k = k
#         self._model = HeuristicModel(n)
#         self._criterion = nn.MSELoss()
#         self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)
#
#     def get_h_values(self, states):
#         states_as_list = [state.get_state_as_list() for state in states]
#         states = np.array(states_as_list, dtype=np.float32)
#         states_tensor = torch.tensor(states)
#         with torch.no_grad():
#             predictions = self._model(states_tensor).numpy()
#         return predictions.flatten()
#
#     def train_model(self, input_data, output_labels, epochs=100):
#         input_as_list = [state.get_state_as_list() for state in input_data]
#         inputs = np.array(input_as_list, dtype=np.float32)
#         outputs = np.array(output_labels, dtype=np.float32)
#
#         inputs_tensor = torch.tensor(inputs)
#         outputs_tensor = torch.tensor(outputs).unsqueeze(1)  # Adding a dimension for the output
#
#         for epoch in range(epochs):
#             self._model.train()
#             self._optimizer.zero_grad()
#
#             predictions = self._model(inputs_tensor)
#             loss = self._criterion(predictions, outputs_tensor)
#             loss.backward()
#             self._optimizer.step()
#
#     def save_model(self, path):
#         torch.save(self._model.state_dict(), path)
#
#     def load_model(self, path):
#         self._model.load_state_dict(torch.load(path))
#         self._model.eval()
#
#
# class BootstrappingHeuristic(LearnedHeuristic):
#     def __init__(self, n=11, k=4):
#         super().__init__(n, k)
#
#     def save_model(self):
#         super().save_model('bootstrapping_heuristic.pth')
#
#     def load_model(self):
#         super().load_model('bootstrapping_heuristic.pth')
