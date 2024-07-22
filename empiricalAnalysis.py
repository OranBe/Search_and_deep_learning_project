import math
from connect_four_game import *
from heuristics import *
from training import *
import time


def run_experiments():
    configurations = [
        ('NN vs Base - Max (NN) starts', 'nn_heuristic', 'base_heuristic', AI_PIECE),
        ('NN vs Base - Min (Base) starts', 'nn_heuristic', 'base_heuristic', PLAYER_PIECE)
    ]

    results = []

    for experiment_name, max_heuristic_function, min_heuristic_function, starting_player in configurations:
        max_heuristic = get_heuristic(max_heuristic_function)
        min_heuristic = get_heuristic(min_heuristic_function)

        print(f"Running experiment: {experiment_name}")

        start_time = time.time()
        current_state = ConnectFourState(None, starting_player)
        current_state.player = starting_player  # Set the starting player
        moves = 0
        path_length = 0

        while not current_state.is_terminal_node():
            if current_state.player == AI_PIECE:
                column, _ = minimax(current_state, 4, -math.inf, math.inf, True, max_heuristic)
            else:
                column, _ = minimax(current_state, 4, -math.inf, math.inf, False, min_heuristic)

            # perform best action
            row = current_state.get_next_open_row(column)
            b_copy = current_state.board.copy()
            drop_piece(b_copy, row, column, AI_PIECE if current_state.player == AI_PIECE else PLAYER_PIECE)
            current_state = ConnectFourState(b_copy, PLAYER_PIECE if current_state.player == AI_PIECE else AI_PIECE)
            moves += 1
            path_length += 1

        end_time = time.time()
        runtime = end_time - start_time

        if current_state.winning_move(AI_PIECE):
            win_count = 1
        else:
            win_count = 0

        print(f"Experiment completed. Runtime: {runtime:.6f}s, Moves: {moves}, Path length: {path_length}, Result: {'Win' if current_state.winning_move(AI_PIECE) else 'Loss or Tie'}")

        results.append((experiment_name, runtime, moves, win_count, path_length))

    print_results(results)

def get_heuristic(name):
    if name == 'base_heuristic':
        return BaseHeuristicNorm(PLAYER_PIECE, AI_PIECE, ROWS, COLUMNS, EMPTY, WINDOW_LENGTH).score_position
    elif name == 'nn_heuristic':
        nn_heuristic = ConnectFourHeuristic()
        nn_heuristic.load_model('bootstrapping_connect_four_heuristic.pth')
        return nn_heuristic.score_position
    else:
        raise ValueError(f"Unknown heuristic: {name}")

def print_results(results):
    print(f"{'Experiment':<30} {'Runtime':<15} {'Moves/Game':<15} {'Win Rate':<10} {'Path Length':<20}")
    for experiment, runtime, moves_per_game, win_rate, path_length in results:
        print(f"{experiment:<30} {runtime:<15.6f} {moves_per_game:<15.2f} {win_rate:<10.2f} {path_length:<20.2f}")

# Example usage:
if __name__ == "__main__":
    run_experiments()
