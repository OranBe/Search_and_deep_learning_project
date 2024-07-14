import math
import random
from BWAS import BWAS
from connect_four_game import *
from heuristics import BootstrappingConnectFourHeuristic
from minimax_algorithm_basic_heuristic import minimax


def bootstrappingTraining(BootstrappingConnectFourHeuristic):
    # BootstrappingConnectFourHeuristic.load_model()
    num_iterations = 1000  # Number of bootstrapping iterations
    batch_size = 50  # Number of states to generate in each batch
    max_moves = 20
    depth = 4
    heuristic = BootstrappingConnectFourHeuristic.score_position

    for _ in range(num_iterations):
        print("num of iteration:", _)
        random_states = generate_minibatch_of_random_states(batch_size, max_moves)
        input_data = []
        output_labels = []
        for state in random_states:
            _, score = minimax(state, depth, -math.inf, math.inf, True, heuristic)

            path = [state]
            current_state = state
            while not current_state.is_terminal_node():
                column, _ = minimax(current_state, depth, -math.inf, math.inf, True, heuristic)
                row = current_state.get_next_open_row(column)
                b_copy = current_state.board.copy()
                drop_piece(b_copy, row, column, AI_PIECE if current_state.player == PLAYER_PIECE else PLAYER_PIECE)
                current_state = ConnectFourState(b_copy, PLAYER_PIECE if current_state.player == AI_PIECE else AI_PIECE)
                path.append(current_state)

            if path and current_state.is_terminal_node() and current_state.winning_move(AI_PIECE):
                len_path = len(path)
                for i, path_board in enumerate(path):
                    new_state = ConnectFourState(path_board.board, 1 if i % 2 == 0 else 0)
                    input_data.append(new_state)
                    output_labels.append(len_path - 1 - i)

        if input_data and output_labels:
            BootstrappingConnectFourHeuristic.train_model(input_data, output_labels, 5)

    BootstrappingConnectFourHeuristic.save_model()
    print("Model training complete and saved using Bootstrapping heuristic")


def generate_minibatch_of_random_states(number_of_random_states, max_moves):
    random_states = []
    for i in range(number_of_random_states):
        state = create_state_with_n_moves(random.randint(1, max_moves))
        random_states.append(state)
    return random_states


def main():
    # Initialize the heuristic
    heuristic = BootstrappingConnectFourHeuristic()

    # Debug information for initialization
    print("Initialized BootstrappingConnectFourHeuristic")

    # Load the model if needed
    # heuristic.load_model()

    # Run the bootstrapping training
    print("Starting bootstrapping training")
    bootstrappingTraining(heuristic)
    print("Completed bootstrapping training")

    # Save the model after training
    heuristic.save_model()
    print("Model saved after training")


if __name__ == "__main__":
    main()
