import numpy as np, numpy
from scipy.special import softmax
from Snort import Snort
from typing import List
import copy

ENCODE_LEGAL, ENCODE_BOARD, ENCODE_BOTH = 0, 1, 2


class Node:
    def __init__(self, game: Snort, parent: "Node" = None) -> None:
        self.game = game
        self.Q = 0
        self.P = []
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.unexplored_moves = numpy.copy(
            self.game.get_legal_moves(self.game.current_player)
        )

    def calculate_P(self):
        result = np.zeros((self.game.board_size, self.game.board_size))
        all_visits = sum([child.visits for child in self.children])
        for p, child in zip(self.P, self.children):
            loc = p[0]
            result[loc] = child.visits / all_visits
        flattened = result.flatten()
        return (
            np.concatenate([np.zeros_like(flattened), flattened])
            if self.game.current_player == Snort.BLUE
            else np.concatenate([flattened, np.zeros_like(flattened)])
        )

    def add_random_child(self, encode_type, model):
        # add child
        legal_mat = self.unexplored_moves
        legal_indices = np.argwhere(legal_mat == True)
        random_index = np.random.choice(len(np.argwhere(legal_mat == True)))
        row, column = legal_indices[random_index]
        self.unexplored_moves[row, column] = False

        # insert to self.childs
        self.game.make_move((row, column))
        child = Node(copy.deepcopy(self.game), self)
        self.children.append(child)
        self.game.unmake_last_move()

        red_moves_p, blue_moves_p, Q = self.decode_state(
            model.predict(self.encode_state(encode_type))
        )
        P = np.concatenate((red_moves_p.flatten(), blue_moves_p.flatten()))
        self.Q = Q[0]
        moves_p = red_moves_p if self.game.current_player == Snort.RED else blue_moves_p
        self.P.append([(row, column), moves_p[row, column]])

        return child

    def encode_state(self, encode_type=ENCODE_BOTH):
        if encode_type == ENCODE_LEGAL:
            return np.concatenate(
                (
                    self.game.red_legal_moves.flatten().astype(int),
                    self.game.blue_legal_moves.flatten().astype(int),
                    [1, 0] if self.game.current_player == Snort.RED else [0, 1],
                )
            ).reshape(-1, (self.game.board_size * self.game.board_size * 2) + 2)
        if encode_type == ENCODE_BOARD:
            grid = self.game.board
            red_board, blue_board, black_board = (
                np.copy(grid),
                np.copy(grid),
                np.copy(grid),
            )
            (
                red_board[grid == Snort.RED],
                blue_board[grid == Snort.BLUE],
                black_board[grid == Snort.BLACK],
            ) = (1, 1, 1)
            (
                red_board[grid != Snort.RED],
                blue_board[grid != Snort.BLUE],
                black_board[grid != Snort.BLACK],
            ) = (0, 0, 0)
            return np.concatenate(
                (
                    red_board.flatten(),
                    blue_board.flatten(),
                    black_board.flatten(),
                    [1, 0] if self.game.current_player == Snort.RED else [0, 1],
                )
            ).reshape(-1, ((self.game.board_size * self.game.board_size) * 3) + 2)
        if encode_type == ENCODE_BOTH:
            encode_board = self.encode_state(ENCODE_BOARD)[:, :-2]
            encode_legal = self.encode_state(ENCODE_LEGAL)
            return np.concatenate(
                (
                    encode_board,
                    encode_legal,
                ),
                axis=1,
            ).reshape(-1, max(encode_board.shape) + max(encode_legal.shape))

    def decode_state(self, vector):
        P, Q = np.array(vector[0][0]), np.array(vector[1][0])
        red_moves, blue_moves = (
            P[:100].reshape(self.game.board_size, self.game.board_size),
            P[100:].reshape(self.game.board_size, self.game.board_size),
        )

        red_moves, blue_moves = (
            softmax(red_moves.flatten()).reshape(
                self.game.board_size, self.game.board_size
            ),
            softmax(red_moves.flatten()).reshape(
                self.game.board_size, self.game.board_size
            ),
        )

        (
            red_moves[self.game.red_legal_moves == False],
            blue_moves[self.game.blue_legal_moves == False],
        ) = (0, 0)
        return red_moves, blue_moves, Q

    def init_P_childs(self, moves_p):
        for child in self.children:
            x, y = moves_p[np.argwhere(self.game.board != child.game.board)][0]
            self.P.append(moves_p[x, y])

    def select_best_child(self) -> "Node":
        if len(self.children) == 0 or len(self.P) == 0:
            return None
        return self.children[
            (
                np.argmax([p * c.Q for p, c in zip(self.P, self.children)])
                if self.game.current_player == Snort.RED
                else np.argmin([p * c.Q for p, c in zip(self.P, self.children)])
            )
        ]

    def select_child_PUCT(self, c):

        def calculate_PUCT(node: "Node", child_index: int, c):
            return node.children[child_index].Q + c * node.P[child_index][1] * (
                np.sqrt(node.visits) / node.children[child_index].visits
            )

        best_child = None
        best_PUCT = 0
        for child_index in range(len(self.children)):
            current_puct = calculate_PUCT(self, child_index, c)
            if current_puct > best_PUCT:
                best_PUCT = current_puct
                best_child = self.children[child_index]

        return best_child

    def is_leaf(self):
        return np.any(self.unexplored_moves == True)
