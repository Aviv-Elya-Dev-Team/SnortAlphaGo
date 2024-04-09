from Board import Board, EMPTY, BLACK, RED, BLUE
from Node import Node, back_propagation, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np
from os.path import exists
from sys import argv
import os
import time
import threading
import tensorflow as tf

import copy

progress = {"progress": 0, "thread_running": True}


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
