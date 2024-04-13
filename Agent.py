from Snort import Snort
from Node import Node, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np, numpy
from os.path import exists
from sys import argv
import time
import threading
import random
import tensorflow as tf
from Config import Config
from tqdm import trange

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

        # return probabilities to choose each child based on visits
        probabilities = np.zeros((self.game.board_size, self.game.board_size))

        for child in root.children:
            probabilities[child.game.move_history[-1]] = child.visits
        probabilities /= np.sum(probabilities)

        return probabilities

    def select(self, root: Node, method, *args):
        result_node = root
        # while the nodes on the way are already dunzo, but the position is ongoing
        # (explored all moves = dunzo)
        while (
            result_node.fully_explored and result_node.game.outcome() == Snort.ONGOING
        ):
            node = method(*args, node=result_node)
            if node == None:
                break

            result_node = node

        return result_node

    def expand(self, node: Node, policy):
        # expand all possible moves since there
        # is no need for a simulation anymore
        node.fully_explored = True  # explore everything
        for move, probability in numpy.ndenumerate(policy):
            if probability > 0:

                # create game clone
                game_clone = node.game.clone()
                game_clone.make_move(move)

                # create child and add to parent (node)
                new_node = Node(game_clone, node)
                new_node.P = probability
                new_node.visits += 1  # TODO: dont know about this, it doesn't work without it maybe need to change PUCT
                node.children.append(new_node)

    def back_propagation(self, node: Node, value):
        node.Q += value
        node.visits += 1

        # propagate
        if node.parent:
            self.back_propagation(node.parent, value)

    def _select_child_PUCT(self, c, node: Node):
        def calculate_PUCT(parent: "Node", child: "Node", c):
            if child.visits == 0:
                q_value = 0
            else:
                q_value = 1 - ((child.Q / child.visits) + 1) / 2
            return q_value + c * child.P * (np.sqrt(parent.visits) / (1 + child.visits))

        best_child = None
        best_PUCT = -numpy.inf
        for child in node.children:
            current_puct = calculate_PUCT(node, child, c)
            if current_puct > best_PUCT:
                best_PUCT = current_puct
                best_child = child

        return best_child

    def best_move_to_do(self, probabilities):
        # most visits = best child (its what its)
        max_index = numpy.argmax(probabilities)
        move = np.unravel_index(max_index, probabilities.shape)
        return move

    def learn(
        self,
        encode_type,
        log_progress={},
        num_games=100,
        num_sessions=3,
        num_iterations=50,
        num_epochs=10,
        batch_size=10,
    ):
        # play num_games games
        for session in range(num_sessions):
            games_history = []

            # play games against self
            for game_index in trange(num_games):
                games_history.append(
                    self.play_against_self(
                        self.game.board_size,
                        self.game.num_black_squares,
                        num_iterations,
                        num_epochs,
                        encode_type,
                    )
                )

            # train on the games played
            for epoch in trange(num_epochs):
                self.train(games_history, batch_size, encode_type)

            # save model
            self.model.save_model(f"models/model{self.encode_type}.keras")

    # play a game against yourself, and return a tuple (encoded_state, probabilities, outcome)
    # of the last move in the game, who won and what were the probability distributions
    def play_against_self(
        self, board_size, num_black_squares, num_iterations, num_epochs, encode_type
    ):
        # initialize a game with a random starting player
        random_player = np.random.choice([Snort.RED, Snort.BLUE])
        game = Snort(random_player, board_size, num_black_squares)

        history = []
        outcome = game.outcome()
        while outcome == Snort.ONGOING:
            # get "random move" from probabilities distributions
            probabilities = self.best_move(num_iterations, num_epochs)
            flat_probabilities = probabilities.flatten()

            # record this game state in history
            history.append((game.clone(), flat_probabilities, game.current_player))

            # choose a random move but with probability
            # distributions from the MCTS best_move function
            random_index = np.random.choice(
                len(flat_probabilities), p=flat_probabilities
            )
            random_move = np.unravel_index(random_index, probabilities.shape)

            game.make_move(random_move)
            outcome = game.outcome()
            if outcome != Snort.ONGOING:
                result = []
                # arbitrarily choose that RED = -1 and BLUE = 1
                # TODO: maybe change this arbitrary decision? idk
                outcome_map = {Snort.RED: -1, Snort.BLUE: 1}
                for previous_game, probabilities, current_player in history:
                    history_outcome = (
                        outcome_map[outcome]
                        if outcome == game.current_player
                        else outcome_map[game.other_player(outcome)]
                    )
                    result.append(
                        (
                            previous_game.encode_state(encode_type),
                            probabilities,
                            history_outcome,
                        )
                    )

                return result

    def train(self, games_history, batch_size):
        random.shuffle(games_history)
        for batchIdx in range(0, len(games_history), batch_size):
            # get batch sample
            sample = games_history[
                batchIdx : min(len(games_history) - 1, batchIdx + batch_size)
            ]

            encoded_states, probabilities, winning_players = zip(*sample)

            encoded_states, probabilities, winning_players = (
                np.array(encoded_states),
                np.array(probabilities),
                np.array(winning_players).reshape(-1, 1),
            )

            # convert data to TensorFlow tensors
            encoded_states_tensor = tf.convert_to_tensor(
                encoded_states, dtype=tf.float32
            )
            probabilities_tensor = tf.convert_to_tensor(probabilities, dtype=tf.float32)
            winning_players_tensor = tf.convert_to_tensor(
                winning_players, dtype=tf.float32
            )

            # train the model
            self.model.fit(
                encoded_states_tensor,
                [probabilities_tensor, winning_players_tensor],
                batch_size=batch_size,
            )


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

    config = Config.get_config()

    # init variables
    board_size = int(config.get("Snort", "board_size"))
    starting_player = int(config.get("GameUI", "starting_player_color"))
    num_black_squares = int(config.get("Snort", "num_black_squares"))
    encode_type = ENCODE_LEGAL

    if len(argv) == 2:
        encode_type = int(argv[1])
    model = Network(encode_type, board_size)

    if exists(f"models/model{encode_type}.keras"):
        model.load_model(f"models/model{encode_type}.keras")

    game = Snort(starting_player, board_size, num_black_squares)
    agent = Agent(game, starting_player, model, encode_type)

    # timer_thread = threading.Thread(target=timer)
    # timer_thread.start()

    agent.learn(encode_type, log_progress=progress)

    # progress["thread_running"] = False
    # timer_thread.join()  # Wait for the timer thread to finish
    # print("\ndone")


if __name__ == "__main__":
    main()
