from Board import Board
from Agent import Agent
from Network import Network
from Node import ENCODE_LEGAL
import numpy as np
class BotGames:
    def __init__(self, red_agent: Agent, blue_agent: Agent) -> None:
        self.agents = {2:red_agent, 3:blue_agent}
        self.winner = 0
    
    
    def run_game(self):
        game = Board()
        turn = np.random.choice([game.RED, game.BLUE])
        while not game.end(turn):
            move = self.agents[turn].best_move_to_do(game, turn)
            while not game.legal_move(turn, move):
                move = self.agents[turn].best_move_to_do(game, turn)
            game.make_move(turn, move)
            turn = game.switch_player(turn)
        self.winner = turn
        return game.moves
    

def main():
    r = Agent(Network(ENCODE_LEGAL), ENCODE_LEGAL)
    bg = BotGames(r, r)
    print(bg.run_game())    


if __name__=='__main__':
    main()
        
        