import math
import random
from connect_four_game import *
from minimax_algorithm_basic_heuristic import minimax
from heuristics import BaseHeuristic


def main():
    current_state = ConnectFourState(None, PLAYER_PIECE)
    current_state.print_board()
    game_over = False
    turn = random.randint(PLAYER_PIECE, AI_PIECE)
    heuristic = BaseHeuristic(PLAYER_PIECE, AI_PIECE,ROWS, COLUMNS, EMPTY, WINDOW_LENGTH)
    while not game_over:
        # Player's turn
        if turn == PLAYER_PIECE:
            col = random.choice(current_state.get_valid_locations())
            if current_state.is_valid_location(col):
                row = current_state.get_next_open_row(col)
                drop_piece(current_state.board, row, col, PLAYER_PIECE)
                current_state = ConnectFourState(current_state.board, AI_PIECE)  # Update the state
                if current_state.winning_move(PLAYER_PIECE):
                    current_state.print_board()
                    print("PLAYER 1 WINS!")
                    game_over = True

                turn = AI_PIECE
                # turn += 1
                # turn = turn % 2

        # AI's turn
        if turn == AI_PIECE and not game_over:
            col, minimax_score = minimax(current_state, 4, -math.inf, math.inf, True,heuristic)
            if current_state.is_valid_location(col):
                row = current_state.get_next_open_row(col)
                drop_piece(current_state.board, row, col, AI_PIECE)
                current_state = ConnectFourState(current_state.board, AI_PIECE)  # Update the state
                if current_state.winning_move(AI_PIECE):
                    current_state.print_board()
                    print("PLAYER 2 WINS!")
                    game_over = True

                turn = PLAYER_PIECE
                # turn += 1
                # turn = turn % 2

        current_state.print_board()


if __name__ == "__main__":
    main()
