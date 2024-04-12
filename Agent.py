from Snort import Snort
from Node import Node, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np, numpy
from os.path import exists
from sys import argv
import time
import threading
import random
import configparser

import copy

progress = {"progress": 0, "thread_running": True}


class MCTSAgent:
    def __init__(self, game: Snort, starting_player) -> None:
        self.game = game
        self.starting_player = starting_player

    def best_move(self, num_iterations=30000):
        root = Node(self.game.clone())

        C = 0.8

        for _ in range(num_iterations):
            # selection
            node = self.select(root, self.select_child_UCT, C)

            # expansion
            if node.game.outcome() == Snort.ONGOING:
                new_node = self.expand(node)
            else:
                new_node = node

            # simulation
            outcome = self.simulate(new_node)

            # back propagation
            self.update_backwards(new_node, outcome)

        # return best child (ratio between wins and visits)
        best_child = max(
            root.children, key=lambda c: c.Q / c.visits if c.visits > 0 else 0
        )

        return best_child.game.move_history[-1]

    def select(self, root: Node, method, *args):
        result_node = root
        # while the nodes on the way are already dunzo, but the position is ongoing
        # (explored all moves = dunzo)
        while (
            len(np.argwhere(result_node.unexplored_moves == True)) == 0
            and result_node.game.outcome() == Snort.ONGOING
        ):
            node = method(*args, node=result_node)
            if node == None:
                break

            result_node = node

        return result_node

    def expand(self, node: Node):
        # set unexplored
        new_move = self._get_next_unexplored_move(node)
        node.unexplored_moves[new_move] = False

        # create game clone
        game_clone = node.game.clone()
        game_clone.make_move(new_move)

        # create child and add to parent (node)
        new_node = Node(game_clone, node)
        node.children.append(new_node)
        return new_node

    def _get_next_unexplored_move(self, node: Node):
        legal_mat = node.unexplored_moves
        legal_indices = np.argwhere(legal_mat == True)
        indices = len(np.argwhere(legal_mat == True))
        random_index = np.random.choice(indices)
        x, y = legal_indices[random_index]
        return (x, y)

    # i looked forwared in time, i saw 14,000,605 futures.
    def simulate(self, node: Node):
        # simulate making random moves until the game concludes
        game_clone = node.game.clone()
        outcome = game_clone.outcome()
        while outcome == Snort.ONGOING:
            moves = game_clone.get_legal_moves(game_clone.current_player)

            # make a random move
            true_indices = np.transpose(np.where(moves == True))

            random_move = tuple(true_indices[np.random.randint(0, len(true_indices))])
            game_clone.make_move(random_move)

            outcome = game_clone.outcome()

        return outcome

    def update_backwards(self, node: Node, outcome):
        # update visits
        node.visits += 1

        # update Q
        if outcome == node.game.other_player(node.game.current_player):
            node.Q += 1
        elif outcome == Snort.DRAW:
            node.Q += 0.5

        # propagate
        if node.parent:
            self.update_backwards(node.parent, outcome)

    def select_child_UCT(self, c, node: Node):

        def calculate_UCT(parent: Node, child: Node, c):
            return (child.Q / child.visits) + c * numpy.sqrt(
                numpy.log(parent.visits) / child.visits
            )

        best_child = None
        best_UCT = 0
        for child in node.children:
            current_uct = calculate_UCT(node, child, c)
            if current_uct > best_UCT:
                best_UCT = current_uct
                best_child = child

        return best_child


class Agent:
    def __init__(
        self,
        game: Snort,
        starting_player,
        model: Network,
        encode_type=ENCODE_LEGAL,
    ) -> None:

        self.encode_type = encode_type
        self.starting_player = starting_player
        self.game = game
        self.model = model
        self.init_model()

    def init_model(self):
        # train model initially
        # TODO: maybe initially train on MCTS rather than random values
        x_train = np.random.random((1, self.model.input_size))

        y_probabilities_train = np.random.random(
            (1, self.game.board_size * self.game.board_size)
        )
        y_value_train = np.random.random((1, 1))

        if not exists(f"models/model{self.encode_type}.keras"):
            self.model.network.fit(
                x_train,
                [y_probabilities_train, y_value_train],
                epochs=1,
                verbose=0,
                use_multiprocessing=True,
            )
        else:
            self.model.load_model(f"models/model{self.encode_type}.keras")
            self.model.compile_model()

    def best_move(self, num_iterations=200, num_epochs=10):
        root: Node = Node(self.game.clone())

        C = 0.8

        for _ in range(num_iterations):
            # selection
            node = self.select(root, self._select_child_PUCT, C)

            # get policy, value
            policy, value = node.decode_state(
                self.model.predict(node.encode_state(self.encode_type))
            )
            value = value[0]

            # expansion
            self.expand(node, policy)

            # back propagation
            self.back_propagation(node, value)
            node = root

        return self.best_move_to_do(root)

    def select(self, root: Node, method, *args):
        result_node = root
        # while the nodes on the way are already dunzo, but the position is ongoing
        # (explored all moves = dunzo)
        while (
            len(np.argwhere(result_node.unexplored_moves == True)) == 0
            and result_node.game.outcome() == Snort.ONGOING
        ):
            node = method(*args, node=result_node)
            if node == None:
                break

            result_node = node

        return result_node

    def expand(self, node: Node, policy):
        # expand all possible moves since there
        # is no need for a simulation anymore
        node.unexplored_moves[:] = False  # explore everything
        for move, probability in numpy.ndenumerate(policy):
            if probability > 0:

                # create game clone
                game_clone = node.game.clone()
                game_clone.make_move(move)

                # create child and add to parent (node)
                new_node = Node(game_clone, node)
                new_node.P = probability
                node.children.append(new_node)

    def back_propagation(self, node: Node, value):
        node.Q += value
        node.visits += 1

        # propagate
        if node.parent:
            self.back_propagation(node.parent, value)

    def _select_child_PUCT(self, c, node: Node):
        # TODO: maybe need to change this formula a bit
        def calculate_PUCT(parent: "Node", child: "Node", c):
            return child.Q + c * child.P * (np.sqrt(parent.visits) / (1 + child.visits))

        best_child = None
        best_PUCT = 0
        for child in node.children:
            current_puct = calculate_PUCT(node, child, c)
            if current_puct > best_PUCT:
                best_PUCT = current_puct
                best_child = child

        return best_child

    def best_move_to_do(self, root: "Node"):
        # most visits = best child (its what its)
        probabilities = np.zeros((self.game.board_size, self.game.board_size))

        for child in root.children:
            probabilities[child.game.move_history[-1]] = child.visits
        probabilities /= np.sum(probabilities)

        max_index = numpy.argmax(probabilities)
        move = np.unravel_index(max_index, probabilities.shape)
        return move

    # TODO: remove this function from the Agent class (WTF)
    def train(self, log_progress={}, num_iterations=1000, num_epochs=10):
        game = Snort(self.starting_player)
        turn = np.random.choice([Snort.RED, Snort.BLUE])
        while game.outcome() == Snort.ONGOING:
            self.best_move(turn, copy.deepcopy(game), num_iterations, num_epochs)
            move = self.best_move_to_do(game, turn)
            game.make_move(turn, move)
            turn = game.switch_player(turn)

            log_progress["progress"] += 1

        self.winner = game.switch_player(turn)
        self.model.save_model(f"models/model{self.encode_type}.keras")


def timer():
    start_time = time.time()
    while progress["thread_running"]:
        elapsed_time = int(time.time() - start_time)
        print(
            f"\rtraining...{elapsed_time} | Moves made: {progress['progress']}",
            end="",
            flush=True,
        )
        time.sleep(1)


def main():
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # read board size from config file
    config.read("config.ini")

    board_size = config.get("Snort", "board_size")
    encode_type = ENCODE_LEGAL

    if len(argv) == 2:
        encode_type = int(argv[1])
    model = Network(encode_type, board_size)

    if exists(f"models/model{encode_type}.keras"):
        model.load_model(f"models/model{encode_type}.keras")

    agent = Agent(model, encode_type)

    timer_thread = threading.Thread(target=timer)
    timer_thread.start()

    agent.train(log_progress=progress)

    progress["thread_running"] = False
    timer_thread.join()  # Wait for the timer thread to finish
    print("\ndone")


if __name__ == "__main__":
    main()
