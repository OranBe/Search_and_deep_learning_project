import math
import random
from connect_four_game import *
from heuristics import BootstrappingConnectFourHeuristic
from minimax_algorithm_basic_heuristic import minimax


def bootstrappingTraining(BootstrappingConnectFourHeuristic, target_network):
    BootstrappingConnectFourHeuristic.load_model()
    target_network.load_model()

    num_iterations = 100  # Number of bootstrapping iterations
    batch_size = 1000  # Number of states to generate in each batch
    max_moves = 10
    max_depth =3
    heuristic = BootstrappingConnectFourHeuristic.score_position
    target_heuristic = target_network.score_position
    original_training_frequency = 1  # Train the model every 1 iteration
    update_target_frequency = 5  # Update target network every 5 iterations

    input_data = []
    output_labels = []

    for iteration in range(num_iterations):
        print("num of iteration:", iteration)
        random_states = generate_minibatch_of_random_states(batch_size, max_moves)
        win = 0
        lose = 0
        depth = random.randint(1, max_depth)
        for state, num_random_moves in random_states:
            _, estimated_value = minimax(state, depth, -math.inf, math.inf, True, heuristic)
            path = [state]
            current_state = state
            terminal_value = None
            while not current_state.is_terminal_node():
                if current_state.player == AI_PIECE:
                    column, _ = minimax(current_state, depth, -math.inf, math.inf, True, heuristic)
                else:
                    column, _ = minimax(current_state, depth, -math.inf, math.inf, False, target_heuristic)

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
                    for s in path[num_random_moves:]:
                        input_data.append(s)
                        output_labels.append(terminal_value)

        # if input_data and output_labels:
        #     BootstrappingConnectFourHeuristic.train_model(input_data, output_labels, 5)

        # Train the model every `original_training_frequency` iterations
        if iteration % original_training_frequency == 0 and input_data and output_labels:
            BootstrappingConnectFourHeuristic.train_model(input_data, output_labels, 5)
            input_data = []
            output_labels = []
            print("Original network weights updated")

        if iteration % update_target_frequency == 0 and input_data and output_labels:
            BootstrappingConnectFourHeuristic.update_target_network(target_network)
            print("Target network weights updated")

    # Perform a final training step if there is remaining data
    if input_data and output_labels:
        BootstrappingConnectFourHeuristic.train_model(input_data, output_labels, 5)
        print("Final training step for original network")

    BootstrappingConnectFourHeuristic.save_model()
    print("Model training complete and saved using Bootstrapping heuristic")


def generate_minibatch_of_random_states(number_of_random_states, max_moves):
    random_states = []
    for i in range(number_of_random_states):
        num_random_moves = random.randint(0, max_moves)
        state = create_state_with_n_moves(num_random_moves, random.randint(1, 2))
        random_states.append([state, num_random_moves])
    return random_states


def main():
    # Initialize the heuristic
    heuristic = BootstrappingConnectFourHeuristic()
    target_network = BootstrappingConnectFourHeuristic()

    # Debug information for initialization
    print("Initialized BootstrappingConnectFourHeuristic")

    # Load the model if needed
    # heuristic.load_model()
    # target_network.load_model()

    # Run the bootstrapping training
    print("Starting bootstrapping training")
    bootstrappingTraining(heuristic, target_network)
    print("Completed bootstrapping training")

    # Save the model after training
    heuristic.save_model()
    print("Model saved after training")


if __name__ == "__main__":
    main()
