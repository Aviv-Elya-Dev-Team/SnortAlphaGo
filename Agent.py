from Board import Board
from Node import Node
from Network import Network
import numpy as np
class Agent:
    def __init__(self, model: Network, encode_type) -> None:
        self.model = model
        self.encode_type = encode_type
    
    
    def best_move(self, state: Board, num_iterations):
        root = Node(state, state.RED)
        # call the network here
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
                Node.back_propagation(node, Q)
    
    
    def best_move_to_do(self, state: Board, turn):
        node = Node(state, turn)
        red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
        red_moves_p, blue_moves_p = red_moves_p.reshape((10, 10)), blue_moves_p.reshape((10, 10))  
        return np.unravel_index(np.argmax(red_moves_p), red_moves_p.shape) if turn == state.RED else np.unravel_index(np.argmax(blue_moves_p), blue_moves_p.shape)
         
                
    
                
                