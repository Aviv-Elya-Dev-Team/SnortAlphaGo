from Board import Board, EMPTY, BLACK, RED, BLUE
from Node import Node, back_propagation, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np
from os.path import exists 
from sys import argv
class Agent:
    def __init__(self, model: Network=Network(ENCODE_LEGAL), encode_type=ENCODE_LEGAL) -> None:
        self.model = model
        self.encode_type = encode_type
        x_train = np.random.random((1, self.model.input_size))
        y_train200 = np.random.random((1, 200))
        y_train1 = np.random.random((1, 1))
        if not exists(f'model{self.encode_type}.keras'):
            self.model.network.fit(x_train, [y_train200, y_train1], epochs=1)
        else:
            self.model.load_model(f'model{self.encode_type}.keras')
    
    
    def best_move(self, turn, state: Board, num_iterations, last_epoch, num_epochs):
        firstBorad = np.copy(state.board)
        node = Node(state, turn)
        for _ in range(num_iterations):
            while not node.is_leaf():
                node = node.select_best_child()
                if not node:
                    return   
                if not np.array_equal(firstBorad, state.board):
                    print('but why??')
            node = node.add_random_child()
            
            red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
            P = np.concatenate((red_moves_p.flatten(), blue_moves_p.flatten()))
            node.Q = Q[0]
            real_Q = np.array([state.reward()]).reshape(1, 1)
            node.visits += 1
            node.init_P_childs(red_moves_p if node.turn == state.RED else blue_moves_p)
            back_propagation(node, node.Q)
            self.model.train(node.encode_state(self.encode_type), [P.reshape(1, 200), real_Q], last_epoch, num_epochs)
            last_epoch += num_epochs
        
    
    
    def best_move_to_do(self, state: Board, turn):
        node = Node(state, turn)
        red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
        red_moves_p, blue_moves_p = red_moves_p.reshape((10, 10)), blue_moves_p.reshape((10, 10))  
        return np.unravel_index(np.argmax(red_moves_p), red_moves_p.shape) if turn == state.RED else np.unravel_index(np.argmax(blue_moves_p), blue_moves_p.shape)
    
    
    def train(self, num_iterations=10, num_epochs=10):
        last_epoch = 1
        game = Board()
        counter = 0
        turn = np.random.choice([game.RED, game.BLUE])
        while not game.end(turn):
            self.best_move(turn, game, num_iterations, last_epoch, num_epochs)
            last_epoch += num_epochs
            move = self.best_move_to_do(game, turn)
            game.make_move(turn, move)
            turn = game.switch_player(turn)
            
        self.winner = game.switch_player(turn)
        self.model.save_model(f'model{self.encode_type}.keras')

def main():
    encode_type = ENCODE_LEGAL
    if len(argv)==2:
        encode_type = int(argv[1])
    model = Network(encode_type)
    if exists(f'model{encode_type}.keras'):
        model.load_model(f'model{encode_type}.keras') 
    r = Agent(model, encode_type)
    r.train()
    


if __name__=='__main__':
    main()
    
                