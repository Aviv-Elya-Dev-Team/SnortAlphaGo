import numpy as np
from scipy.special import softmax
from Board import Board, BLUE, RED, BLACK

ENCODE_LEGAL, ENCODE_BOARD, ENCODE_BOTH = 0, 1, 2

class Node:
    def __init__(self, state: Board, turn) -> None:
        self.state = state
        self.Q = 0
        self.P = []
        self.parent = None
        self.childs = []
        self.visits = 0
        self.turn = turn
        self.is_leaf = False
    
    
    def increase_visits(self):
        self.visits+=1
    
    
    def add_random_child(self):
        legal_mat = self.state.red_legal_moves if self.turn == RED else self.state.blue_legal_moves
        legal_indices = np.argwhere(legal_mat==True) 
        if len(legal_indices) == 0:
            self.is_leaf = True
            return None 
        random_index = np.random.choice(len(np.argwhere(legal_mat==True)))
        x, y = legal_indices[random_index]
        self.state.make_move(self.turn, (x, y))
        self.turn = self.state.switch_player(self.turn)
        child = Node(self.state, self.turn)
        if child.state.board not in [c.state.board for c in self.childs]:
            self.childs.append(child)
            child.parent = self
        else:
            child = None
        self.state.unmake_last_move()
        self.turn = self.state.switch_player(self.turn)
        if not child:
            return self.add_random_child()
        return child
        
        
    def encode_state(self, encode_type = ENCODE_BOTH):
        if encode_type==ENCODE_LEGAL:
            return np.concatenate((self.state.red_legal_moves.flatten().astype(int), self.state.blue_legal_moves.flatten().astype(int), [1, 0] if self.turn == RED else [0, 1]))
        if encode_type==ENCODE_BOARD:
            grid = self.state.board
            red_board, blue_board, black_board = np.copy(grid), np.copy(grid), np.copy(grid)
            red_board[grid==RED], blue_board[grid==BLUE], black_board[grid==BLACK] = 1, 1, 1 
            red_board[grid!=RED], blue_board[grid!=BLUE], black_board[grid!=BLACK] = 0, 0, 0 
            return np.concatenate((red_board.flatten(), black_board.flatten(), black_board.flatten(),[1, 0] if self.turn == RED else [0, 1]))
        if encode_type==ENCODE_BOTH:
            return np.concatenate((self.encode_state(ENCODE_BOARD)[:-2], self.encode_state(ENCODE_LEGAL)))
            
    
    def decode_state(self, vector):
        P, Q = vector[:200], vector[200:]
        red_moves, blue_moves = P[:100].reshape(10, 10), P[100:].reshape(10, 10)
        red_moves[self.state.red_legal_moves == False], blue_moves[self.state.blue_legal_moves == False] = 0, 0
        red_moves, blue_moves = softmax(red_moves.flatten()).reshape(10, 10), softmax(red_moves.flatten()).reshape(10, 10)
        return red_moves, blue_moves, Q
    
    
    def init_P_childs(self, moves_p):
        for child in self.childs:
            x, y = moves_p[np.argwhere(self.state.board != child.state.board)][0]
            self.P.append(moves_p[x, y])
    
    
    def set_P_childs(self, moves_p):
        for p, child in zip(self.P, self.childs):
            x, y = moves_p[np.argwhere(self.state.board != child.state.board)][0]
            p = moves_p[x, y]
    
    
    def select_best_child(self):
        return self.childs[np.argmax([p*c.Q for p, c in zip(self.P, self.childs)]) if self.turn == RED\
                           else np.argmin([p*c.Q for p, c in zip(self.P, self.childs)])]
        
        
    
def back_propagation(node: Node, value):
    if node.turn == BLUE:
        node.Q += value
    else:
        node.Q -= value
    if node.parent:
        back_propagation(node.parent, value)