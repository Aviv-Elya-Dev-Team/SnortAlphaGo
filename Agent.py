from Board import Board, EMPTY, BLACK, RED, BLUE
from Node import Node, back_propagation
from Network import Network
import numpy as np
class Agent:
    def __init__(self, model: Network, encode_type) -> None:
        self.model = model
        self.encode_type = encode_type
        self.model.train(np.zeros((self.model.input_size,1)),np.zeros((201,1)), 0, 1)
    
    
    def best_move(self, state: Board, num_iterations):
        root = Node(state, state.RED)
        # call the network here
        last_epoch = 1
        num_epoch = 10
        node = root
        for _ in range(num_iterations):
            while not node.is_leaf:
                node = node.select_best_child()
                node = node.add_random_child()
                if not node:
                    break
                red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
                node.Q = Q
                node.visits = 1
                node.init_P_childs(red_moves_p if node.turn == state.RED else blue_moves_p)
                back_propagation(node, Q)
                self.model.train(node.encode_state(self.encode_type), self.Q, last_epoch, num_epoch)
                last_epoch += num_epoch
    
    
    def best_move_to_do(self, state: Board, turn):
        node = Node(state, turn)
        red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
        red_moves_p, blue_moves_p = red_moves_p.reshape((10, 10)), blue_moves_p.reshape((10, 10))  
        return np.unravel_index(np.argmax(red_moves_p), red_moves_p.shape) if turn == state.RED else np.unravel_index(np.argmax(blue_moves_p), blue_moves_p.shape)
    
    
    def train(self):
        pass   
    
                