from Board import Board, EMPTY, BLACK, RED, BLUE
from Node import Node, back_propagation, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np, numpy
from os.path import exists
from sys import argv
import os
import time
import threading
import tensorflow as tf

import copy

progress = {"progress": 0, "thread_running": True}


class MCTSAgent:
    def __init__(self, game: Board, args, parent=None, turn=None) -> None:
        self.game = game
        self.args = args

        self.parent = parent
        self.turn = turn

    def best_move(self, state, num_iterations):
        root = Node(self.game, self.turn)

        C = 0.8

        for _ in num_iterations:
            # selection
            node = self.select(
                root,
            )
            # expansion
            self.expand()
            # simulation
            self.simulation()
            # back propagation
            self.back_propagation()

    def select(self, root: Node, method, *args):
        result_node = root
        # while the nodes on the way are already dunzo, but the position is ongoing
        # (explored all moves = dunzo)
        while len(result_node.unexpolred_moves) == 0 and not result_node.state.end():
            node = method(*args, node=result_node)
            if node == None:
                break

            result_node = node

        return result_node

    def expand(self, node: Node):
        new_move = node.unexpolred_moves.pop()
        game_clone = copy.deepcopy(node.state)
        game_clone.make_move(new_move)
        new_node = Node(game_clone, node)
        node.children.append(new_node)
        return new_node

    def simulation(self):
        pass

    def back_propagation(self):
        pass

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
        for child in node.children:
            current_uct = calculate_UCT(node, child, c)
            if current_uct > best_UCT:
                best_UCT = current_uct
                best_child = child

        return best_child


class Agent:
    def __init__(
        self, model: Network = Network(ENCODE_LEGAL), encode_type=ENCODE_LEGAL
    ) -> None:
        self.model = model
        self.encode_type = encode_type

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

    def best_move(self, turn, state: Board, num_iterations, last_epoch, num_epochs):
        firstBorad = np.copy(state.board)
        root: Node = Node(state, turn)
        node = root

        C = 0.8

        for _ in range(num_iterations):
            while not node.is_leaf():
                node = root.select_child_PUCT(C)
                if not node:
                    return
                if not np.array_equal(firstBorad, state.board):
                    print("but why??")
            new_node = node.add_random_child(self.encode_type, self.model)

            back_propagation(new_node, new_node.Q)

            real_Q = np.array([state.reward()]).reshape(1, 1)
            train_P = node.calculate_P()
            self.model.train(
                node.encode_state(self.encode_type),
                [train_P.reshape(1, 200), real_Q],
                last_epoch,
                num_epochs,
            )
            last_epoch += num_epochs
            node = root

    def best_move_to_do(self, state: Board, turn):
        node = Node(state, turn)
        red_moves_p, blue_moves_p, Q = node.decode_state(
            self.model.predict(node.encode_state(self.encode_type))
        )
        red_moves_p, blue_moves_p = red_moves_p.reshape((10, 10)), blue_moves_p.reshape(
            (10, 10)
        )
        return (
            np.unravel_index(np.argmax(red_moves_p), red_moves_p.shape)
            if turn == state.RED
            else np.unravel_index(np.argmax(blue_moves_p), blue_moves_p.shape)
        )

    def train(self, log_progress={}, num_iterations=1000, num_epochs=10):
        last_epoch = 1
        game = Board()
        turn = np.random.choice([game.RED, game.BLUE])
        while not game.end(turn):
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
            f"\rtraining...{elapsed_time} | Progress: {progress['progress']}",
            end="",
            flush=True,
        )
        time.sleep(1)


if __name__ == "__main__":
    # don't show tensorflow output
    main()
