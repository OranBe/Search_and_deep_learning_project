import math
import random
from connect_four_game import *


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score


def score_position(board, piece):
    score = 0

    # Score center column
    center_array = [int(i) for i in list(board[:, COLUMNS // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Score Horizontal
    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMNS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score Vertical
    for c in range(COLUMNS):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Score positive sloped diagonal
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Score negative sloped diagonal
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def is_terminal_node(state):
    return state.winning_move(PLAYER_PIECE) or state.winning_move(AI_PIECE) or len(state.get_valid_locations()) == 0


def minimax(state, depth, alpha, beta, maximizingPlayer):
    board = state.board
    valid_locations = state.get_valid_locations()
    is_terminal = is_terminal_node(state)
    if depth == 0 or is_terminal:
        if is_terminal:
            if state.winning_move(board, AI_PIECE):
                return None, 100000000000000
            elif state.winning_move(board, PLAYER_PIECE):
                return None, -10000000000000
            else:  # Game is over, no more valid moves
                return None, 0
        else:  # Depth is zero
            return None, score_position(board, AI_PIECE)
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = state.get_next_open_row(col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_state = ConnectFourState(b_copy, PLAYER_PIECE)
            new_score = minimax(new_state, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = state.get_next_open_row(col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_state = ConnectFourState(b_copy, AI_PIECE)
            new_score = minimax(new_state, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def main():
    initial_state = ConnectFourState(None, PLAYER_PIECE)
    initial_state.print_board()

    # Test minimax function
    col, minimax_score = minimax(initial_state, 4, -math.inf, math.inf, True)
    print(f"Best column: {col}, Minimax score: {minimax_score}")

if __name__ == "__main__":
    main()
