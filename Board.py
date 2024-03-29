import numpy as np


class Board:
    EMPTY, BLACK, RED, BLUE = 0, 1, 2, 3

    def __init__(self) -> None:
        self.board = np.zeros((10, 10), np.uint8)
        self.red_legal_moves = np.ones((10, 10), dtype=np.bool_)
        self.blue_legal_moves = np.ones((10, 10), dtype=np.bool_)
        i = 0
        while i < 3:
            x, y = np.random.randint(0, 9), np.random.randint(0, 9)
            if self.board[x, y] == self.EMPTY:
                self.board[x, y] = self.BLACK
                self.red_legal_moves[x, y], self.blue_legal_moves[x, y] = False, False
                if x > 0:
                    self.red_legal_moves[x - 1, y], self.blue_legal_moves[x - 1, y] = (
                        False,
                        False,
                    )
                if x < 9:
                    self.red_legal_moves[x + 1, y], self.blue_legal_moves[x + 1, y] = (
                        False,
                        False,
                    )
                if y > 0:
                    self.red_legal_moves[x, y - 1], self.blue_legal_moves[x, y - 1] = (
                        False,
                        False,
                    )
                if y < 9:
                    self.red_legal_moves[x, y + 1], self.blue_legal_moves[x, y + 1] = (
                        False,
                        False,
                    )
                i += 1
        self.moves = []

    def legal_move(self, player, position):
        x, y = position
        if x not in range(10) or y not in range(10) or (len(self.moves)>0 and self.moves[-1][0] == player):
            return False
        return (
            self.blue_legal_moves[x, y]
            if player == self.BLUE
            else self.red_legal_moves[x, y]
        )

    def make_move(self, player, position):
        x, y = position
        if self.legal_move(player, position):
            self.board[x, y] = player
            self.moves.append([player, position])
            self.red_legal_moves[x, y] = False
            self.blue_legal_moves[x, y] = False
            if player == self.BLUE:
                if x > 0:
                    self.red_legal_moves[x - 1, y] = False
                if x < 9:
                    self.red_legal_moves[x + 1, y] = False
                if y > 0:
                    self.red_legal_moves[x, y - 1] = False
                if y < 9:
                    self.red_legal_moves[x, y + 1] = False
            else:
                if x > 0:
                    self.blue_legal_moves[x - 1, y] = False
                if x < 9:
                    self.blue_legal_moves[x + 1, y] = False
                if y > 0:
                    self.blue_legal_moves[x, y - 1] = False
                if y < 9:
                    self.blue_legal_moves[x, y + 1] = False
            return True
        return False

    def unmake_last_move(self):
        if len(self.moves) == 0:
            return
        player, position = self.moves.pop()
        x, y = position
        self.board[x, y] = self.EMPTY
        positions_to_check = [[x, y], [x + 1, y], [x - 1, y], [x, y - 1], [x, y + 1]]
        for i, j in positions_to_check:
            if not self.is_in_the_borad(i, j):
                continue
            for player in [self.RED, self.BLUE]:
                if not self.have_non_same_neib(player, (i, j)):
                    if player == self.RED:
                        self.red_legal_moves[i, j] = True
                    else:
                        self.blue_legal_moves[i, j] = True

    def is_in_the_borad(self, x, y):
        return x in range(10) and y in range(10)

    def have_non_same_neib(self, player, position):
        x, y = position
        return not (
            (
                not self.is_in_the_borad(x + 1, y) or
                self.board[x + 1, y] in [player, self.EMPTY]
            )
            and (
                not self.is_in_the_borad(x - 1, y) or
                self.board[x - 1, y] in [player, self.EMPTY]
            )
            and (
                not self.is_in_the_borad(x, y - 1) or
                self.board[x, y - 1] in [player, self.EMPTY]
            )
            and (
                not self.is_in_the_borad(x, y + 1) or
                self.board[x, y + 1] in [player, self.EMPTY]
            )
        )

    def end(self, player):
        return (
            np.all(self.blue_legal_moves == False)
            if player == self.RED
            else np.all(self.red_legal_moves == False)
        )

    def switch_player(self, player):
        return self.RED if player == self.BLUE else self.BLUE
