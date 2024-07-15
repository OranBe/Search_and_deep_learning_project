import math
import random
from connect_four_game import *
from heuristics import BaseHeuristic, BaseHeuristicNorm


def minimax(state, depth, alpha, beta, maximizingPlayer, heuristic):
    board = state.board
    valid_locations = state.get_valid_locations()
    is_terminal = state.is_terminal_node()
    if depth == 0 or is_terminal:
        if is_terminal:
            if state.winning_move(AI_PIECE):
                state.minimax_value = 1
                return [state], None, 1
            elif state.winning_move(PLAYER_PIECE):
                state.minimax_value = -1
                return [state], None, -1
            else:  # Game is over, no more valid moves
                state.minimax_value = 0
                return [state], None, 0
        else:  # Depth is zero, meaning that reached depth limit
            value = heuristic(state)
            state.minimax_value = value
            return [state], None, value
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        best_path = []
        for col in valid_locations:
            row = state.get_next_open_row(col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_state = ConnectFourState(b_copy, PLAYER_PIECE)
            path, _, new_score = minimax(new_state, depth - 1, alpha, beta, False, heuristic)
            if new_score > value:
                value = new_score
                column = col
                best_path = [state] + path
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_path, column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        best_path = []
        for col in valid_locations:
            row = state.get_next_open_row(col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_state = ConnectFourState(b_copy, AI_PIECE)
            path, _, new_score = minimax(new_state, depth - 1, alpha, beta, True, heuristic)
            if new_score < value:
                value = new_score
                column = col
                best_path = [state] + path
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_path, column, value


def main():
    initial_state = ConnectFourState(None, PLAYER_PIECE)
    # initial_state.print_board()
    heuristic = BaseHeuristicNorm(PLAYER_PIECE, AI_PIECE, ROWS, COLUMNS, EMPTY, WINDOW_LENGTH)

    # Test minimax function
    path, col, minimax_score = minimax(initial_state, 4, -math.inf, math.inf, True, heuristic.score_position)
    print("Best path:")
    for state in path:
        state.print_board()
        print()  # Add a newline for better readability
    print(f"Best column: {col}, Minimax score: {minimax_score}")


if __name__ == "__main__":
    main()
