from Snort import Snort
from Node import Node, back_propagation, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np, numpy
from os.path import exists
from sys import argv
import time
import threading
import random

import copy

progress = {"progress": 0, "thread_running": True}


class MCTSAgent:
    def __init__(self, game: Snort, starting_player) -> None:
        self.game = game
        self.starting_player = starting_player

    def best_move_to_do(self, num_iterations=1000):
        root = Node(copy.deepcopy(self.game))

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
            root.childs, key=lambda c: c.Q / c.visits if c.visits > 0 else 0
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
        node.childs.append(new_node)
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
        # update
        node.visits += 1
        if outcome == node.game.other_player(self.game.current_player):
            node.Q += 1

        # propagate
        if node.parent:
            self.update_backwards(node.parent, outcome)

    def select_child_PUCT(self, c, node: Node):

        def calculate_PUCT(parent: "Node", child_index: int, c):
            return parent.childs[child_index].Q + c * parent.P[child_index][1] * (
                np.sqrt(parent.visits) / parent.childs[child_index].visits
            )

        best_child = None
        best_PUCT = 0
        for child_index in range(len(node.childs)):
            current_puct = calculate_PUCT(node, child_index, c)
            if current_puct > best_PUCT:
                best_PUCT = current_puct
                best_child = node.childs[child_index]

        return best_child

    def select_child_UCT(self, c, node: Node):

        def calculate_UCT(parent: Node, child: Node, c):
            return (child.Q / child.visits) + c * numpy.sqrt(
                numpy.log(parent.visits) / child.visits
            )

        best_child = None
        best_UCT = 0
        for child in node.childs:
            current_uct = calculate_UCT(node, child, c)
            if current_uct > best_UCT:
                best_UCT = current_uct
                best_child = child

        return best_child


class Agent:
    def __init__(
        self,
        starting_player,
        model: Network = Network(ENCODE_LEGAL),
        encode_type=ENCODE_LEGAL,
    ) -> None:

        self.model = model
        self.encode_type = encode_type
        self.starting_player = starting_player
        self.init_model()

    def init_model(self):
        # train model initially
        x_train = np.random.random((1, self.model.input_size))
        y_train200 = np.random.random((1, 200))
        y_train1 = np.random.random((1, 1))
        if not exists(f"models/model{self.encode_type}.keras"):
            self.model.network.fit(
                x_train,
                [y_train200, y_train1],
                epochs=1,
                verbose=0,
                use_multiprocessing=True,
            )
        else:
            self.model.load_model(f"models/model{self.encode_type}.keras")
            self.model.compile_model()

    def best_move(self, turn, game: Snort, num_iterations, last_epoch, num_epochs):
        firstBorad = np.copy(game.board)
        root: Node = Node(game)
        node = root

        C = 0.8

        for _ in range(num_iterations):
            while not node.is_leaf():
                node = root.select_child_PUCT(C)
                if not node:
                    return
                if not np.array_equal(firstBorad, game.board):
                    print("but why??")
            new_node = node.add_random_child(self.encode_type, self.model)

            back_propagation(new_node, new_node.Q)

            real_Q = np.array([game.reward()]).reshape(1, 1)
            train_P = node.calculate_P()
            self.model.train(
                node.encode_state(self.encode_type),
                [train_P.reshape(1, 200), real_Q],
                last_epoch,
                num_epochs,
            )
            last_epoch += num_epochs
            node = root

    def best_move_to_do(self, game: Snort, turn):
        # TODO: add parent here maybe
        node = Node(game)
        red_moves_p, blue_moves_p, Q = node.decode_state(
            self.model.predict(node.encode_state(self.encode_type))
        )
        red_moves_p, blue_moves_p = red_moves_p.reshape(
            (game.board_size, game.board_size)
        ), blue_moves_p.reshape((game.board_size, game.board_size))
        return (
            np.unravel_index(np.argmax(red_moves_p), red_moves_p.shape)
            if turn == Snort.RED
            else np.unravel_index(np.argmax(blue_moves_p), blue_moves_p.shape)
        )

    # TODO: remove this function from the Agent class (WTF)
    def train(self, log_progress={}, num_iterations=1000, num_epochs=10):
        last_epoch = 1
        game = Snort(self.starting_player)
        turn = np.random.choice([Snort.RED, Snort.BLUE])
        while game.outcome() == Snort.ONGOING:
            self.best_move(
                turn, copy.deepcopy(game), num_iterations, last_epoch, num_epochs
            )
            last_epoch += num_epochs
            move = self.best_move_to_do(game, turn)
            game.make_move(turn, move)
            turn = game.switch_player(turn)

            log_progress["progress"] += 1

        self.winner = game.switch_player(turn)
        self.model.save_model(f"models/model{self.encode_type}.keras")


def main():
    encode_type = ENCODE_LEGAL
    if len(argv) == 2:
        encode_type = int(argv[1])
    model = Network(encode_type)
    if exists(f"models/model{encode_type}.keras"):
        model.load_model(f"models/model{encode_type}.keras")
    r = Agent(model, encode_type)

    timer_thread = threading.Thread(target=timer)
    timer_thread.start()

    r.train(log_progress=progress)

    progress["thread_running"] = False
    timer_thread.join()  # Wait for the timer thread to finish
    print("\ndone")


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


if __name__ == "__main__":
    main()
