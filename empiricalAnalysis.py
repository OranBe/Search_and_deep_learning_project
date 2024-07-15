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
        path_lengths = []
        node_expansions = []

        print(f"Running experiments for Heuristic={heuristic_name}")

        for iteration, state in enumerate(random_states):
            start_time = time.time()
            path, col, minimax_score = minimax(state, 4, -math.inf, math.inf, True, heuristic)
            end_time = time.time()

            runtime = end_time - start_time
            runtimes.append(runtime)
            path_lengths.append(len(path) if path is not None else 0)
            node_expansions.append(len(path))

            # Print minimax values for each state in the path
            minimax_values = [s.minimax_value for s in path]
            print(
                f"Iteration {iteration + 1}/100 completed. Runtime: {runtime:.6f}s, Path length: {len(path) if path else 'N/A'}, Expansions: {len(path)}, Minimax values: {minimax_values}, Best column: {col}, Minimax score: {minimax_score}")
            print("Best path:")
            for state in path:
                state.print_board()
                print()

            print(
                f"Iteration {iteration + 1}/100 completed. Runtime: {runtime:.6f}s, Path length: {len(path) if path else 'N/A'}")

        avg_runtime = sum(runtimes) / len(runtimes)
        avg_path_length = sum(path_lengths) / len(path_lengths)
        avg_expansions = sum(node_expansions) / len(node_expansions)

        results.append((heuristic_name, avg_runtime, avg_path_length, avg_expansions))

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
    print(f"{'Heuristic':<20} {'Avg Runtime':<15} {'Avg Path Length':<20} {'Avg Expansions':<15}")
    for heuristic, avg_runtime, avg_path_length, avg_expansions in results:
        print(f"{heuristic:<20} {avg_runtime:<15.6f} {avg_path_length:<20.2f} {avg_expansions:<15.2f}")


# Example usage:
if __name__ == "__main__":
    run_experiments()
