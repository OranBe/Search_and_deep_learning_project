import numpy as np

ROWS = 4
COLUMNS = 4
PLAYER_PIECE = 1
AI_PIECE = 2
EMPTY = 0
WINDOW_LENGTH = 4


def create_board():
    board = np.zeros((ROWS, COLUMNS))
    return board


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROWS - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROWS):
        if board[r][col] == 0:
            return r


def print_board(board):
    print(np.flip(board, 0))


def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMNS - 3):
        for r in range(ROWS):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diagonals
    for c in range(COLUMNS - 3):
        for r in range(ROWS - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diagonals
    for c in range(COLUMNS - 3):
        for r in range(3, ROWS):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True
    return False


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def get_all_possible_states(board, piece):
    possible_states = []
    valid_locations = get_valid_locations(board)
    for col in valid_locations:
        temp_board = board.copy()
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, piece)
        possible_states.append(temp_board)
    return possible_states


def its_a_draw(board):
    for col in range(COLUMNS):
        if is_valid_location(board, col):
            return False
    return True


def main():
    board = create_board()
    print_board(board)
    game_over = False
    turn = 0

    while not game_over:
        if turn == 0:
            col = int(input("Player 1 Make your Selection (0-3):"))

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    print("PLAYER 1 WINS!!")
                    game_over = True

        else:
            col = int(input("Player 2 Make your Selection (0-3):"))

            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    print("PLAYER 2 WINS!!")
                    game_over = True

        print_board(board)

        turn += 1
        turn = turn % 2


if __name__ == "__main__":
    main()
