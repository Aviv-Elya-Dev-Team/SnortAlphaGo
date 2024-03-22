import Board
import Node
class Agent:
    def __init__(self, model, encode_type) -> None:
        self.model = model
        self.encode_type = encode_type
    
    
    def best_move(self, state: Board, num_iterations):
        root = Node(state, Board.RED)
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
                node.init_P_childs(red_moves_p if node.turn == Board.RED else blue_moves_p)
                Node.back_propagation(node, Q)
    
    
    def best_move_to_de(self, state: Board, turn):
        node = Node(state, turn)
        red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
        return np.argmax(red_moves_p) if turn == Board.RED else np.argmax(blue_moves_p)
         
                
    
                
                