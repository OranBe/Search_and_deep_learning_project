import random

import numpy as np

ROWS = 5
COLUMNS = 5
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4


def drop_piece(board, row, col, piece):
    board[row][col] = piece


class ConnectFourState:
    def __init__(self, board, player):
        if board is None:
            self.board = np.zeros((ROWS, COLUMNS))
        else:
            self.board = board
        if player is None:
            self.player = PLAYER_PIECE
        else:
            self.player = player
        self.minimax_value = None

    def __eq__(self, other):
        if isinstance(other, ConnectFourState):
            return np.array_equal(self.board, other.board) and self.player == other.player
        return False

    def __hash__(self):
        return hash((tuple(map(tuple, self.board)), self.player))

    def get_valid_locations(self):
        valid_locations = []
        for col in range(COLUMNS):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations

    def get_neighbors(self):
        neighbors = []
        valid_locations = self.get_valid_locations()
        for col in valid_locations:
            tmp_board = np.copy(self.board)
            row = self.get_next_open_row(col)
            drop_piece(tmp_board, row, col, self.player)
            next_player = 1 if self.player == 0 else 0
            neighbors.append(ConnectFourState(tmp_board, next_player))
        return neighbors

    def is_valid_location(self, col):
        return self.board[ROWS - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(ROWS):
            if self.board[r][col] == 0:
                return r

    def its_a_draw(self):
        for col in range(COLUMNS):
            if self.is_valid_location(col):
                return False
        return True

    def winning_move(self, piece):
        # Check horizontal locations for win
        for c in range(COLUMNS - 3):
            for r in range(ROWS):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and \
                        self.board[r][
                            c + 3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(COLUMNS):
            for r in range(ROWS - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and \
                        self.board[r + 3][
                            c] == piece:
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMNS - 3):
            for r in range(ROWS - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][
                    c + 2] == piece and \
                        self.board[r + 3][
                            c + 3] == piece:
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMNS - 3):
            for r in range(3, ROWS):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][
                    c + 2] == piece and \
                        self.board[r - 3][
                            c + 3] == piece:
                    return True
        return False

    def print_board(self):
        print(np.flip(self.board, 0))

    def is_terminal_node(self):
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or len(self.get_valid_locations()) == 0

    def get_state_as_list(self):
        return self.board.flatten().tolist()


def create_state_with_n_moves(n, starting_player):
    state = ConnectFourState(None, starting_player)
    for _ in range(n):
        valid_locations = state.get_valid_locations()
        if not valid_locations:
            break
        col = random.choice(valid_locations)
        row = state.get_next_open_row(col)
        drop_piece(state.board, row, col, state.player)
        state.player = 1 if state.player == 2 else 2
    return state






def test_connect_four_state():
    # Create initial game state
    board = np.zeros((ROWS, COLUMNS))
    game_state = ConnectFourState(board, PLAYER_PIECE)

    print("Testing get_neighbors:")
    neighbors = game_state.get_neighbors()

    # Test get_valid_locations
    print("Testing get_valid_locations:")
    valid_locations = game_state.get_valid_locations()
    print(f"Valid locations (should be [0, 1, 2, 3]): {valid_locations}")

    # Test drop_piece and get_next_open_row
    print("Testing drop_piece and get_next_open_row:")
    col = 0
    row = game_state.get_next_open_row(col)
    drop_piece(game_state.board, row, col, PLAYER_PIECE)
    game_state.print_board()
    print(f"Dropped piece at row {row}, column {col}")

    # Test winning_move (should be False)
    print("Testing winning_move (initial move, should be False):")
    print(f"Winning move for PLAYER_PIECE: {game_state.winning_move(PLAYER_PIECE)}")

    # Fill the column and test winning_move (should be True)
    print("Filling a column to test winning_move:")
    for i in range(COLUMNS):
        row = game_state.get_next_open_row(col)
        drop_piece(game_state.board, row, col, PLAYER_PIECE)
    game_state.print_board()
    print(f"Winning move for PLAYER_PIECE (should be True): {game_state.winning_move(PLAYER_PIECE)}")

    # Test its_a_draw (should be False)
    print("Testing its_a_draw (should be False):")
    print(f"Is it a draw: {game_state.its_a_draw()}")

    # Fill the board and test its_a_draw (should be True)
    print("Filling the board to test its_a_draw:")
    for c in range(1, 4):
        for r in range(4):
            drop_piece(game_state.board, r, c, AI_PIECE if (r + c) % 2 == 0 else PLAYER_PIECE)
    game_state.print_board()
    print(f"Is it a draw (should be True): {game_state.its_a_draw()}")

    # Test get_neighbors
    print("Testing get_neighbors:")
    neighbors = game_state.get_neighbors()
    print(f"Number of neighbors (should be 0 if board is full): {len(neighbors)}")
    if len(neighbors) > 0:
        print("First neighbor board state:")
        neighbors[0].print_board()


if __name__ == "__main__":
    test_connect_four_state()
