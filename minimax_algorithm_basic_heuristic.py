import math
import random
from connect_four_game import *
from heuristics import BaseHeuristicNorm, BootstrappingConnectFourHeuristic


def minimax(state, depth, alpha, beta, maximizingPlayer, heuristic):
    board = state.board
    valid_locations = state.get_valid_locations()
    is_terminal = state.is_terminal_node()
    if depth == 0 or is_terminal:
        if is_terminal:
            if state.winning_move(AI_PIECE):
                return None, 1
            elif state.winning_move(PLAYER_PIECE):
                return None, -1
            else:  # Game is over, no more valid moves
                return None, 0
        else:  # Depth is zero, meaning that reached depth limit
            value = heuristic(state)
            return None, value
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = state.get_next_open_row(col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_state = ConnectFourState(b_copy, PLAYER_PIECE)
            _, new_score = minimax(new_state, depth - 1, alpha, beta, False, heuristic)
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
            _, new_score = minimax(new_state, depth - 1, alpha, beta, True, heuristic)
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def main():
    initial_state = ConnectFourState(None, PLAYER_PIECE)
    # initial_state.print_board()
    heuristic = BaseHeuristicNorm(PLAYER_PIECE, AI_PIECE, ROWS, COLUMNS, EMPTY, WINDOW_LENGTH)

    # Test minimax function
    col, minimax_score = minimax(initial_state, 4, -math.inf, math.inf, True, heuristic.score_position)
    print(f"Best column: {col}, Minimax score: {minimax_score}")


if __name__ == "__main__":
    main()
