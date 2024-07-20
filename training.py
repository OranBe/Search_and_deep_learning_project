import math
import random
from BWAS import BWAS
from connect_four_game import *
from heuristics import BootstrappingConnectFourHeuristic, BaseHeuristic
from minimax_algorithm_basic_heuristic import minimax


def bootstrappingTraining(BootstrappingConnectFourHeuristic):
    BootstrappingConnectFourHeuristic.load_model()
    num_iterations = 500  # Number of bootstrapping iterations
    batch_size = 50  # Number of states to generate in each batch
    max_moves = 15
    depth = 4
    heuristic = BootstrappingConnectFourHeuristic.score_position
    for _ in range(num_iterations):
        print("num of iteration:", _)
        random_states = generate_minibatch_of_random_states(batch_size, max_moves)
        input_data = []
        output_labels = []
        win = 0
        lose = 0
        for state in random_states:
            _, estimated_value = minimax(state, depth, -math.inf, math.inf, True, heuristic)
            path = [state]
            current_state = state
            terminal_value = None
            while not current_state.is_terminal_node():
                if current_state.player == AI_PIECE:
                    column, _ = minimax(current_state, depth, -math.inf, math.inf, True, heuristic)
                else:
                    column, _ = minimax(current_state, depth, -math.inf, math.inf, False, heuristic)

                # perform best action
                row = current_state.get_next_open_row(column)
                b_copy = current_state.board.copy()
                drop_piece(b_copy, row, column, AI_PIECE if current_state.player == PLAYER_PIECE else PLAYER_PIECE)
                current_state = ConnectFourState(b_copy, PLAYER_PIECE if current_state.player == AI_PIECE else AI_PIECE)
                path.append(current_state)

            if current_state.is_terminal_node():
                if current_state.winning_move(AI_PIECE):
                    terminal_value = 1
                    win += 1
                    for s in path:
                        input_data.append(s)
                        output_labels.append(terminal_value)
                elif current_state.winning_move(PLAYER_PIECE):
                    terminal_value = -1
                    lose += 1
                    for s in path:
                        input_data.append(s)
                        output_labels.append(terminal_value)
                else:
                    terminal_value = 0
                    for s in path:
                        input_data.append(s)
                        output_labels.append(terminal_value)

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
