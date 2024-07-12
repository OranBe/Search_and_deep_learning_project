import random
from BWAS import BWAS
from connect_four_game import ConnectFourState
from connect_four_game import create_state_with_n_moves


def bootstrappingTraining(BootstrappingConnectFourHeuristic):
    BootstrappingConnectFourHeuristic.load_model()
    num_iterations = 1000  # Number of bootstrapping iterations
    batch_size = 50  # Number of states to generate in each batch
    T = 1000000  # Maximum number of expansions
    W = 5  # Weight factor for the heuristic
    B = 10  # Number of expansions in each batch
    max_moves = 20

    for _ in range(num_iterations):
        print("num of iteration:", _)
        random_states = generate_minibatch_of_random_states(batch_size, max_moves)
        input_data = []
        output_labels = []
        counter = 0
        for state in random_states:
            print(counter)
            counter += 1
            path, expansions = BWAS(state, W, B, BootstrappingConnectFourHeuristic.get_h_values, T)
            while path is None:
                T *= 2
                path, expansions = BWAS(state, W, B, BootstrappingConnectFourHeuristic.get_h_values, T)
            if path:
                len_path = len(path)
                for i, path_board in enumerate(path):
                    # TODO - Check if this is correct "1 if i % 2 == 0 else 0"
                    new_state = ConnectFourState(path_board, 1 if i % 2 == 0 else 0)
                    input_data.append(new_state)
                    output_labels.append(len_path - 1 - i)
        if input_data and output_labels:
            BootstrappingConnectFourHeuristic.train_model(input_data, output_labels, 1000)

    BootstrappingConnectFourHeuristic.save_model()
    print("Model training complete and saved using Bootstrapping heuristic")


def generate_minibatch_of_random_states(number_of_random_states, max_moves):
    random_states = []
    for i in range(number_of_random_states):
        state = create_state_with_n_moves(random.randint(1, max_moves))
        random_states.append(state)
    return random_states
