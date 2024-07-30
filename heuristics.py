import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from connect_four_game import ROWS, COLUMNS


class RandomHeuristic:
    def __init__(self, player_piece, ai_piece, rows, columns, empty, window_length):
        self.player_piece = player_piece
        self.ai_piece = ai_piece
        self.rows = rows
        self.columns = columns
        self.empty = empty
        self.window_length = window_length

    def score_position(self, state):
        return random.uniform(-1, 1)


class BaseHeuristicNorm:
    def __init__(self, player_piece, ai_piece, rows, columns, empty, window_length):
        self.player_piece = player_piece
        self.ai_piece = ai_piece
        self.rows = rows
        self.columns = columns
        self.empty = empty
        self.window_length = window_length

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.player_piece if piece == self.ai_piece else self.ai_piece

        if window.count(piece) == 4:
            score += 1
        elif window.count(piece) == 3 and window.count(self.empty) == 1:
            score += 0.1
        elif window.count(piece) == 2 and window.count(self.empty) == 2:
            score += 0.05

        if window.count(opp_piece) == 3 and window.count(self.empty) == 1:
            score -= 0.1

        return score

    def score_position(self, state):
        score = 0
        center_array = [int(i) for i in list(state.board[:, self.columns // 2])]
        center_count = center_array.count(self.ai_piece)
        score += center_count * 0.15

        for r in range(self.rows):
            row_array = [int(i) for i in list(state.board[r, :])]
            for c in range(self.columns - 3):
                window = row_array[c:c + self.window_length]
                score += self.evaluate_window(window, self.ai_piece)

        for c in range(self.columns):
            col_array = [int(i) for i in list(state.board[:, c])]
            for r in range(self.rows - 3):
                window = col_array[r:r + self.window_length]
                score += self.evaluate_window(window, self.ai_piece)

        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                window = [state.board[r + i][c + i] for i in range(self.window_length)]
                score += self.evaluate_window(window, self.ai_piece)

        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                window = [state.board[r + 3 - i][c + i] for i in range(self.window_length)]
                score += self.evaluate_window(window, self.ai_piece)

        return score


class ConnectFourHeuristicModel(nn.Module):
    def __init__(self, input_dim):
        super(ConnectFourHeuristicModel, self).__init__()
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


class ConnectFourHeuristic:
    def __init__(self, rows=ROWS, columns=COLUMNS):
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

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def load_model(self, path):
        self._model.load_state_dict(torch.load(path))
        self._model.eval()

    def update_target_network(self, target_network):
        target_network._model.load_state_dict(self._model.state_dict())


class BootstrappingConnectFourHeuristic(ConnectFourHeuristic):
    def __init__(self, rows=ROWS, columns=COLUMNS):
        super().__init__(rows, columns)

    def save_model(self):
        super().save_model('bootstrapping_connect_four_heuristic.pth')

    def load_model(self):
        super().load_model('bootstrapping_connect_four_heuristic.pth')
