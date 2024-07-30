import math
from connect_four_game import *
from heuristics import *
from training import *
import time


def run_experiments():
    configurations = [
        ('NN vs Base - Max (NN) starts', 'nn_heuristic', 'base_heuristic', AI_PIECE, 2),
        ('NN vs Base - Min (Base) starts', 'nn_heuristic', 'base_heuristic', PLAYER_PIECE, 2),
        ('NN vs Random - Max (NN) starts', 'nn_heuristic', 'random_heuristic', AI_PIECE, 2),
        ('NN vs Random - Min (Random) starts', 'nn_heuristic', 'random_heuristic', PLAYER_PIECE, 2),
        ('NN vs Base - Max (NN) starts', 'nn_heuristic', 'base_heuristic', AI_PIECE, 3),
        ('NN vs Base - Min (Base) starts', 'nn_heuristic', 'base_heuristic', PLAYER_PIECE, 3),
        ('NN vs Random - Max (NN) starts', 'nn_heuristic', 'random_heuristic', AI_PIECE, 3),
        ('NN vs Random - Min (Random) starts', 'nn_heuristic', 'random_heuristic', PLAYER_PIECE, 3),
        ('NN vs Base - Max (NN) starts', 'nn_heuristic', 'base_heuristic', AI_PIECE, 4),
        ('NN vs Base - Min (Base) starts', 'nn_heuristic', 'base_heuristic', PLAYER_PIECE, 4),
        ('NN vs Random - Max (NN) starts', 'nn_heuristic', 'random_heuristic', AI_PIECE, 4),
        ('NN vs Random - Min (Random) starts', 'nn_heuristic', 'random_heuristic', PLAYER_PIECE, 4),
        ('NN vs Base - Max (NN) starts', 'nn_heuristic', 'base_heuristic', AI_PIECE, 5),
        ('NN vs Base - Min (Base) starts', 'nn_heuristic', 'base_heuristic', PLAYER_PIECE, 5),
        # ('NN vs Random - Max (NN) starts', 'nn_heuristic', 'random_heuristic', AI_PIECE, 5),
        # ('NN vs Random - Min (Random) starts', 'nn_heuristic', 'random_heuristic', PLAYER_PIECE, 5)
    ]

    results = []

    for experiment_name, max_heuristic_function, min_heuristic_function, starting_player, depth in configurations:
        max_heuristic = get_heuristic(max_heuristic_function)
        min_heuristic = get_heuristic(min_heuristic_function)

        print(f"Running experiment: {experiment_name}")

        # Run multiple games if heuristic is random
        num_games = 50 if 'random_heuristic' in [max_heuristic_function, min_heuristic_function] else 1
        total_runtime = 0
        total_moves = 0
        total_wins = 0
        total_ties = 0
        total_path_length = 0

        for _ in range(num_games):
            start_time = time.time()
            current_state = ConnectFourState(None, starting_player)
            current_state.player = starting_player  # Set the starting player
            moves = 0
            path_length = 0

            while not current_state.is_terminal_node():
                if current_state.player == AI_PIECE:
                    column, _ = minimax(current_state, depth, -math.inf, math.inf, True, max_heuristic)
                else:
                    column, _ = minimax(current_state, depth, -math.inf, math.inf, False, min_heuristic)

                # perform best action
                row = current_state.get_next_open_row(column)
                b_copy = current_state.board.copy()
                drop_piece(b_copy, row, column, AI_PIECE if current_state.player == AI_PIECE else PLAYER_PIECE)
                current_state = ConnectFourState(b_copy, PLAYER_PIECE if current_state.player == AI_PIECE else AI_PIECE)
                moves += 1
                path_length += 1

            end_time = time.time()
            runtime = end_time - start_time

            total_runtime += runtime
            total_moves += moves
            total_path_length += path_length

            if current_state.winning_move(AI_PIECE):
                total_wins += 1
            elif current_state.winning_move(PLAYER_PIECE):
                total_wins += 0
            else:
                total_ties += 1

        avg_runtime = total_runtime / num_games
        avg_moves = total_moves / num_games
        win_rate = total_wins / num_games
        tie_rate = total_ties / num_games
        avg_path_length = total_path_length / num_games

        print(f"Experiment completed. Avg Runtime: {avg_runtime:.6f}s, Avg Moves: {avg_moves}, Win Rate: {win_rate:.2f}, Tie Rate: {tie_rate:.2f}, Avg Path length: {avg_path_length}, Depth: {depth}")

        results.append((experiment_name, depth, avg_runtime, avg_moves, win_rate, tie_rate, avg_path_length))

    print_results(results)


def get_heuristic(name):
    if name == 'base_heuristic':
        return BaseHeuristicNorm(PLAYER_PIECE, AI_PIECE, ROWS, COLUMNS, EMPTY, WINDOW_LENGTH).score_position
    elif name == 'random_heuristic':
        return RandomHeuristic(PLAYER_PIECE, AI_PIECE, ROWS, COLUMNS, EMPTY, WINDOW_LENGTH).score_position
    elif name == 'nn_heuristic':
        nn_heuristic = ConnectFourHeuristic()
        nn_heuristic.load_model('bootstrapping_connect_four_heuristic.pth')
        return nn_heuristic.score_position
    else:
        raise ValueError(f"Unknown heuristic: {name}")


def print_results(results):
    header = f"{'Experiment':<35} {'Depth':<10} {'Runtime':<15} {'Moves/Game':<15} {'Win Rate':<10} {'Tie Rate':<10} {'Path Length':<20}"
    print(header)
    print('-' * len(header))
    for experiment, depth, runtime, moves_per_game, win_rate, tie_rate, path_length in results:
        print(f"{experiment:<35} {depth:<10} {runtime:<15.6f} {moves_per_game:<15.2f} {win_rate:<10.2f} {tie_rate:<10.2f} {path_length:<20.2f}")


# Example usage:
if __name__ == "__main__":
    run_experiments()
