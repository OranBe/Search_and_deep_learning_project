import math
import random
from connect_four_game import *
from heuristics import BaseHeuristic


def minimax(state, depth, alpha, beta, maximizingPlayer, heuristic):
    board = state.board
    valid_locations = state.get_valid_locations()
    is_terminal = state.is_terminal_node()
    if depth == 0 or is_terminal:
        if is_terminal:
            if state.winning_move(AI_PIECE):
                return None, 100000000000000
            elif state.winning_move(PLAYER_PIECE):
                return None, -10000000000000
            else:  # Game is over, no more valid moves
                return None, 0
        else:  # Depth is zero
            return None, heuristic(state)[0]
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = state.get_next_open_row(col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_state = ConnectFourState(b_copy, PLAYER_PIECE)
            new_score = minimax(new_state, depth - 1, alpha, beta, False, heuristic)[1]
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
            new_score = minimax(new_state, depth - 1, alpha, beta, True, heuristic)[1]
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
    heuristic = BaseHeuristic(PLAYER_PIECE, AI_PIECE, ROWS, COLUMNS, EMPTY, WINDOW_LENGTH)

    # Test minimax function
    col, minimax_score = minimax(initial_state, 4, -math.inf, math.inf, True, heuristic)
    print(f"Best column: {col}, Minimax score: {minimax_score}")


if __name__ == "__main__":
    main()
