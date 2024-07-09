import math
import random
from connect_four_game import *
from minimax_algorithm_basic_heuristic import minimax

board = create_board()
game_over = False
print_board(board)
turn = random.randint(PLAYER_PIECE, AI_PIECE)

while not game_over and not its_a_draw(board):
    # Player's turn
    if turn == PLAYER_PIECE:
        col = random.choice(get_valid_locations(board))
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER_PIECE)

            if winning_move(board, PLAYER_PIECE):
                print_board(board)
                print("PLAYER 1 WINS!")
                game_over = True

            turn = AI_PIECE
            # turn += 1
            # turn = turn % 2

    # AI's turn
    if turn == AI_PIECE and not game_over:
        col, minimax_score = minimax(board, 4, -math.inf, math.inf, True)
        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI_PIECE)

            if winning_move(board, AI_PIECE):
                print_board(board)
                print("PLAYER 2 WINS!")
                game_over = True

            turn = PLAYER_PIECE
            # turn += 1
            # turn = turn % 2

    print_board(board)

if its_a_draw(board):
    print("IT'S A DRAW!")
else:
    print("GAME OVER")
