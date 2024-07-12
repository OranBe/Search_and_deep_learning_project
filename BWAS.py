from queue import PriorityQueue

class Node:
    def __init__(self, state, g=0, f=0, parent=None):
        self.state = state
        self.g = g
        self.f = f
        self.parent = parent

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        else:
            return self.g < other.g

    def __eq__(self, other):
        return self.f == other.f and self.g == other.g and self.state == other.state

    def get_state_as_list(self):
        return self.state.get_state_as_list()


def BWAS(start, W, B, heuristic_function, T):
    open = PriorityQueue()  # priority queue based on minimal f
    closed = {}  # maps states to their shortest discovered path costs
    UB = float('inf')
    n_UB = None
    LB = 0
    expansions = 0

    initial_heuristic = heuristic_function([start])[0]
    N_start = Node(start, 0, initial_heuristic, None)
    # heapq.heappush(open, N_start)
    open.put(N_start)

    while not open.empty() and expansions <= T:
        generated = []
        batch_expansions = 0
        while not open.empty() and batch_expansions < B and expansions <= T:
            # n = heapq.heappop(open)
            n = open.get()
            s, g, f, parent = n.state, n.g, n.f, n.parent
            expansions += 1
            batch_expansions += 1

            if len(generated) == 0:
                LB = max(f, LB)
            if s.is_goal():
                if UB > g:
                    UB = g
                    n_UB = n
                continue

            for successor, cost in s.get_neighbors():
                successor_g = g + cost
                if successor not in closed or successor_g < closed[successor]:
                    closed[successor] = successor_g
                    generated.append((successor, successor_g, n))

        if LB >= UB:
            return path_to_goal(n_UB), expansions

        generated_states = [state for state, _, _ in generated]

        if not generated_states:
            continue

        heuristics = heuristic_function(generated_states)

        for i in range(len(generated)):
            s, g, parent = generated[i]
            h = heuristics[i]
            new_node = Node(s, g, g + h * W, parent)
            # heapq.heappush(open, new_node)
            open.put(new_node)
    return path_to_goal(n_UB), expansions


def path_to_goal(n):
    path = []
    while n:
        path.append(n.get_state_as_list())
        n = n.parent
    if not path:
        return None
    return list(reversed(path))

