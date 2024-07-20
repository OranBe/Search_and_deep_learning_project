import math

from connect_four_game import *
from heuristics import *
from training import *
import time


def run_experiments():
    configurations = [
        ('BaseHeuristic', 'base_heuristic'),
        ('NNHeuristic', 'nn_heuristic')
    ]

    results = []
    random_states = generate_minibatch_of_random_states(100, 20)

    for heuristic_name, heuristic_function in configurations:
        heuristic = get_heuristic(heuristic_function)

        runtimes = []
        move_counts = []
        win_counts = []
        path_lengths = []
        node_expansions = []

        print(f"Running experiments for Heuristic={heuristic_name}")

        for iteration, state in enumerate(random_states):
            start_time = time.time()
            current_state = state
            moves = 0
            path_length = 0
            nodes_expanded = 0
            while not current_state.is_terminal_node():
                if current_state.player == AI_PIECE:
                    column, _ = minimax(current_state, 4, -math.inf, math.inf, True, heuristic)
                else:
                    column, _ = minimax(current_state, 4, -math.inf, math.inf, False, heuristic)

                # perform best action
                row = current_state.get_next_open_row(column)
                b_copy = current_state.board.copy()
                drop_piece(b_copy, row, column, AI_PIECE if current_state.player == AI_PIECE else PLAYER_PIECE)
                current_state = ConnectFourState(b_copy, PLAYER_PIECE if current_state.player == AI_PIECE else AI_PIECE)
                moves += 1
                path_length += 1
                nodes_expanded += len(current_state.get_valid_locations())

            end_time = time.time()
            runtime = end_time - start_time
            runtimes.append(runtime)
            move_counts.append(moves)
            path_lengths.append(path_length)
            node_expansions.append(nodes_expanded)

            if current_state.winning_move(AI_PIECE):
                win_counts.append(1)
            else:
                win_counts.append(0)

            print(f"Iteration {iteration + 1}/100 completed. Runtime: {runtime:.6f}s, Moves: {moves}, Path length: {path_length}, Expansions: {nodes_expanded}, Result: {'Win' if current_state.winning_move(AI_PIECE) else 'Loss or Tie'}")

        avg_runtime = sum(runtimes) / len(runtimes)
        avg_moves_per_game = sum(move_counts) / len(move_counts)
        win_rate = sum(win_counts) / len(win_counts)
        avg_path_length = sum(path_lengths) / len(path_lengths)
        avg_expansions = sum(node_expansions) / len(node_expansions)

        results.append((heuristic_name, avg_runtime, avg_moves_per_game, win_rate, avg_path_length, avg_expansions))

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
    print(f"{'Heuristic':<20} {'Avg Runtime':<15} {'Avg Moves/Game':<15} {'Win Rate':<10} {'Avg Path Length':<20} {'Avg Expansions':<15}")
    for heuristic, avg_runtime, avg_moves_per_game, win_rate, avg_path_length, avg_expansions in results:
        print(f"{heuristic:<20} {avg_runtime:<15.6f} {avg_moves_per_game:<15.2f} {win_rate:<10.2f} {avg_path_length:<20.2f} {avg_expansions:<15.2f}")



# Example usage:
if __name__ == "__main__":
    run_experiments()
