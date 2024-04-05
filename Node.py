import numpy as np
from scipy.special import softmax
from Board import Board, EMPTY, BLACK, RED, BLUE
from typing import List
import copy

ENCODE_LEGAL, ENCODE_BOARD, ENCODE_BOTH = 0, 1, 2


class Node:
    def __init__(self, state: Board, turn) -> None:
        self.state = state
        self.Q = 0
        self.P = []
        self.parent = None
        self.childs: List[Node] = []
        self.visits = 0
        self.turn = turn
        self.unexpolred_moves = (
            state.red_legal_moves if turn == RED else state.blue_legal_moves
        )

    def calculate_P(self):
        result = np.zeros((10, 10))
        all_visits = sum([child.visits for child in self.childs])
        for p, child in zip(self.P, self.childs):
            loc = p[0]
            result[loc] = child.visits / all_visits
        flattened = result.flatten()
        return (
            np.concatenate([np.zeros_like(flattened), flattened])
            if self.turn == BLUE
            else np.concatenate([flattened, np.zeros_like(flattened)])
        )

    def add_random_child(self, encode_type, model):
        # add child
        legal_mat = self.unexpolred_moves
        legal_indices = np.argwhere(legal_mat == True)
        random_index = np.random.choice(len(np.argwhere(legal_mat == True)))
        x, y = legal_indices[random_index]
        self.unexpolred_moves[x, y] = False

        # insert to self.childs
        self.state.make_move(self.turn, (x, y))
        self.turn = self.state.switch_player(self.turn)
        child = Node(copy.deepcopy(self.state), self.turn)
        self.childs.append(child)
        self.state.unmake_last_move()

        self.turn = self.state.switch_player(self.turn)

        red_moves_p, blue_moves_p, Q = self.decode_state(
            model.predict(self.encode_state(encode_type))
        )
        P = np.concatenate((red_moves_p.flatten(), blue_moves_p.flatten()))
        self.Q = Q[0]
        moves_p = red_moves_p if self.turn == RED else blue_moves_p
        self.P.append([(x, y), moves_p[x, y]])

        return child

    def encode_state(self, encode_type=ENCODE_BOTH):
        if encode_type == ENCODE_LEGAL:
            return np.concatenate(
                (
                    self.state.red_legal_moves.flatten().astype(int),
                    self.state.blue_legal_moves.flatten().astype(int),
                    [1, 0] if self.turn == self.state.RED else [0, 1],
                )
            ).reshape(-1, 202)
        if encode_type == ENCODE_BOARD:
            grid = self.state.board
            red_board, blue_board, black_board = (
                np.copy(grid),
                np.copy(grid),
                np.copy(grid),
            )
            (
                red_board[grid == self.state.RED],
                blue_board[grid == self.state.BLUE],
                black_board[grid == self.state.BLACK],
            ) = (1, 1, 1)
            (
                red_board[grid != self.state.RED],
                blue_board[grid != self.state.BLUE],
                black_board[grid != self.state.BLACK],
            ) = (0, 0, 0)
            return np.concatenate(
                (
                    red_board.flatten(),
                    black_board.flatten(),
                    black_board.flatten(),
                    [1, 0] if self.turn == RED else [0, 1],
                )
            ).reshape(-1, 302)
        if encode_type == ENCODE_BOTH:
            return np.concatenate(
                (
                    self.encode_state(ENCODE_BOARD)[:, :-2],
                    self.encode_state(ENCODE_LEGAL),
                ),
                axis=1,
            ).reshape(-1, 502)

    def decode_state(self, vector):
        P, Q = vector[0][0], vector[1][0]
        P, Q = np.array(P), np.array(Q)
        red_moves, blue_moves = P[:100].reshape(10, 10), P[100:].reshape(10, 10)
        red_moves, blue_moves = softmax(red_moves.flatten()).reshape(10, 10), softmax(
            red_moves.flatten()
        ).reshape(10, 10)
        (
            red_moves[self.state.red_legal_moves == False],
            blue_moves[self.state.blue_legal_moves == False],
        ) = (0, 0)
        return red_moves, blue_moves, Q

    def init_P_childs(self, moves_p):
        for child in self.childs:
            x, y = moves_p[np.argwhere(self.state.board != child.state.board)][0]
            self.P.append(moves_p[x, y])

    def select_best_child(self) -> "Node":
        if len(self.childs) == 0 or len(self.P) == 0:
            return None
        return self.childs[
            (
                np.argmax([p * c.Q for p, c in zip(self.P, self.childs)])
                if self.turn == self.state.RED
                else np.argmin([p * c.Q for p, c in zip(self.P, self.childs)])
            )
        ]

    def select_child_PUCT(self, c):

        def calculate_PUCT(node: "Node", child_index: int, c):
            return node.childs[child_index].Q + c * node.P[child_index][1] * (
                np.sqrt(node.visits) / node.childs[child_index].visits
            )

        best_child = None
        best_PUCT = 0
        for child_index in range(len(self.childs)):
            current_puct = calculate_PUCT(self, child_index, c)
            if current_puct > best_PUCT:
                best_PUCT = current_puct
                best_child = self.childs[child_index]

        return best_child

    def is_leaf(self):
        return np.any(self.unexpolred_moves == True)


def back_propagation(node: Node, value):
    # TODO: maybe negative Q is bad
    node.visits += 1
    node.Q += value
    if node.parent:
        back_propagation(node.parent, value)
