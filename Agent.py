from Board import Board, EMPTY, BLACK, RED, BLUE
from Node import Node, back_propagation, ENCODE_BOTH, ENCODE_LEGAL, ENCODE_BOARD
from Network import Network
import numpy as np
from os.path import exists 
class Agent:
    def __init__(self, model: Network=Network(ENCODE_LEGAL), encode_type=ENCODE_LEGAL) -> None:
        self.model = model
        self.encode_type = encode_type
        x_train = np.random.random((1, self.model.input_size))
        y_train200 = np.random.random((1, 200))
        y_train1 = np.random.random((1, 1))
        if not exists('model.keras'):
            self.model.network.fit(x_train, [y_train200, y_train1], epochs=1)
    
    
    def best_move(self, turn, state: Board, num_iterations, last_epoch, num_epochs):
        node = Node(state, turn)
        for _ in range(num_iterations):
            if not node:
                print('fuck')
                
            while not node.is_leaf():
                node = node.select_best_child()       
            node = node.add_random_child()
            red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
            P = np.concatenate((red_moves_p.flatten(), blue_moves_p.flatten()))
            node.Q = Q[0]
            node.visits = 1
            node.init_P_childs(red_moves_p if node.turn == state.RED else blue_moves_p)
            back_propagation(node, node.Q)
            self.model.train(node.encode_state(self.encode_type), [P.reshape(1, 200), Q.reshape(1, 1)], last_epoch, num_epochs)
            last_epoch += num_epochs
    
    
    def best_move_to_do(self, state: Board, turn):
        node = Node(state, turn)
        red_moves_p, blue_moves_p, Q = node.decode_state(self.model.predict(node.encode_state(self.encode_type)))
        red_moves_p, blue_moves_p = red_moves_p.reshape((10, 10)), blue_moves_p.reshape((10, 10))  
        return np.unravel_index(np.argmax(red_moves_p), red_moves_p.shape) if turn == state.RED else np.unravel_index(np.argmax(blue_moves_p), blue_moves_p.shape)
    
    
    def train(self, num_iterations=10, num_epochs=10):
        last_epoch = 1
        game = Board()
        turn = np.random.choice([game.RED, game.BLUE])
        while not game.end(turn):
            self.best_move(turn, game, num_iterations, last_epoch, num_epochs)
            last_epoch += num_epochs
            move = self.best_move_to_do(game, turn)
            while not game.legal_move(turn, move):
                move = self.agents[turn].best_move_to_do(game, turn)
            game.make_move(turn, move)
            turn = game.switch_player(turn)
        self.winner = game.switch_player(turn)
        self.model.save_model('model.keras')

def main():
    model = Network(ENCODE_LEGAL)
    if exists('model.keras'):
        model.load_model('model.keras') 
    r = Agent(model, ENCODE_LEGAL)
    r.train()
    


if __name__=='__main__':
    main()
    
                