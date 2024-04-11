import numpy


class Snort:
    EMPTY, BLACK, RED, BLUE, ONGOING = 0, 1, 2, 3, 4

    def __init__(self, starting_player, board_size=10, num_black_squares=3) -> None:
        self.board_size = board_size
        self.num_black_squares = num_black_squares

        self.current_player = starting_player
        self.board = numpy.zeros((board_size, board_size))

        # setup legal moves for each player
        self.red_legal_moves = numpy.ones(
            (self.board_size, self.board_size), dtype=numpy.bool_
        )
        self.blue_legal_moves = numpy.ones(
            (self.board_size, self.board_size), dtype=numpy.bool_
        )
        self.move_history = []

        self.init_black_squares(num_black_squares)

    def init_black_squares(self, num_black_squares):
        black_squares_filled = 0
        while black_squares_filled < num_black_squares:
            # get random move
            random_move = (
                numpy.random.randint(0, self.board_size),
                numpy.random.randint(0, self.board_size),
            )

            # insert random move to the board
            if self.board[random_move] == self.EMPTY:

                self.board[random_move] = self.BLACK

                # update legal moves
                self.red_legal_moves[random_move] = False
                self.blue_legal_moves[random_move] = False

                # take care of neighbors
                neighbors = self._get_cell_neighbors(random_move)
                for neighbor in neighbors:
                    self.red_legal_moves[neighbor] = False
                    self.blue_legal_moves[neighbor] = False

                black_squares_filled += 1

    # returns locations of all the neighbors
    # of the given cell in the board matrix
    def _get_cell_neighbors(self, cell: tuple):
        # returns all the neighbors of a cell, regardless of which player is in these squares
        #
        row, column = cell
        neighbors = []
        # up
        if row - 1 >= 0:
            neighbors.append((row - 1, column))

        # down
        if row + 1 < self.board_size:
            neighbors.append((row + 1, column))

        # left
        if column - 1 >= 0:
            neighbors.append((row, column - 1))

        # right
        if column + 1 < self.board_size:
            neighbors.append((row, column + 1))

        return neighbors

    # switches the player from RED to BLUE or
    # from BLUE to RED depends on who's playing right now
    def _switch_player(self):
        self.current_player = Snort.other_player(self.current_player)

    # makes the move on the board and
    # returns True if the move was made
    # and False otherwise
    def make_move(self, move: tuple):
        if self.is_legal_move(move):
            self.board[move] = self.current_player
            self._update_legal_moves(move)
            self.move_history.append(move)
            self._switch_player()
            return True

        return False

    # gets the move and updates red or blue legal moves
    def _update_legal_moves(self, move: tuple):
        neighbors = self._get_cell_neighbors(move)

        # get the correct legal moves array
        current_player_legal_moves = self.get_legal_moves(self.current_player)
        other_player_legal_moves = self.get_legal_moves(
            Snort.other_player(self.current_player)
        )

        # update legal moves
        current_player_legal_moves[move] = False
        other_player_legal_moves[move] = False
        for neighbor in neighbors:
            other_player_legal_moves[neighbor] = False

    # clones this game and returns the new object
    def clone(self):
        result = Snort(self.current_player, self.board_size, self.num_black_squares)
        result.red_legal_moves = numpy.copy(self.red_legal_moves)
        result.blue_legal_moves = numpy.copy(self.blue_legal_moves)
        result.board = numpy.copy(self.board)
        result.move_history = [move for move in self.move_history]
        return result

    # returns either RED, BLUE or ONGOING to tell either
    # who won or if the game is still ongoing
    def outcome(self):
        if numpy.all(self.blue_legal_moves == False):
            return self.RED

        elif numpy.all(self.red_legal_moves == False):
            return self.BLUE
        else:
            return self.ONGOING

    # returns the last move that was made
    def last_move(self):
        if len(self.move_history) == 0:
            return None
        return self.move_history[-1]

    # returns a list of available moves for the current player
    def get_legal_moves(self, player):
        return self.red_legal_moves if player == self.RED else self.blue_legal_moves

    # checks if the given move is a move the current player can play
    def is_legal_move(self, move: tuple):
        row, column = move
        if (
            # not an empty slot
            self.board[move] != self.EMPTY
            # out of bounds
            or (row < 0 or row >= self.board_size)
            or (column < 0 or column >= self.board_size)
        ):
            return False

        legal_moves = self.get_legal_moves(self.current_player)
        return legal_moves[move]

    # returns the player opposite of the one given as a parameter
    @staticmethod
    def other_player(player):
        if player == Snort.RED:
            return Snort.BLUE
        else:
            return Snort.RED
