import numpy as np
EMPTY, BLACK, RED, BLUE = 0, 1, 2, 3
class Board:
    def __init__(self) -> None:
        self.board = np.zeros((10,10), np.uint8)
        self.red_legal_moves = np.ones((10,10), dtype=np.bool)
        self.blue_legal_moves = np.ones((10,10), dtype=np.bool)
        i = 0
        while i<3:
            x, y = np.random.randint(0,9), np.random.randint(0,9)
            if self.board[x, y] == EMPTY:
                self.board[x, y] = BLACK
                self.red_legal_moves[x, y], self.blue_legal_moves[x,y] = False, False
                if x>0:
                    self.red_legal_moves[x-1, y], self.blue_legal_moves[x-1,y] = False, False
                if x<9:
                    self.red_legal_moves[x+1, y], self.blue_legal_moves[x+1,y] = False, False
                if y>0:
                    self.red_legal_moves[x, y-1], self.blue_legal_moves[x,y-1] = False, False
                if y<9:
                    self.red_legal_moves[x, y+1], self.blue_legal_moves[x,y+1] = False, False
                i += 1
        self.moves = []
    
    
    def legal_move(self, player, position):
        x, y = position
        if x not in range(10) or y not in range(10):
            return False
        return self.blue_legal_moves[x, y] if player==BLUE else self.red_legal_moves[x, y]
        
    
    def make_move(self, player, position):
        x, y = position
        if self.legal_move(player, position):
            self.board[x, y] = player
            self.moves.append([player, position])
            self.red_legal_moves[x, y] =  False
            self.blue_legal_moves[x, y] =  False
            if player == BLUE:    
                if x>0:
                    self.red_legal_moves[x-1, y] = False
                if x<9:
                    self.red_legal_moves[x+1, y] = False
                if y>0:
                    self.red_legal_moves[x, y-1] = False
                if y<9:
                    self.red_legal_moves[x, y+1] = False
            else:
                if x>0:
                    self.blue_legal_moves[x-1, y] = False
                if x<9:
                    self.blue_legal_moves[x+1, y] = False
                if y>0:
                    self.blue_legal_moves[x, y-1] = False
                if y<9:
                    self.blue_legal_moves[x, y+1] = False
            return True
        return False
    
    
    def unmake_last_move(self):
        player, position = self.moves.pop()
        x, y = position
        self.board[x, y] = EMPTY
        positions_to_check = [[x, y], [x+1, y], [x-1, y], [x, y-1], [x, y+1]]
        for i, j in positions_to_check:
            pass