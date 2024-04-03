import pygame
import sys
from Board import Board
from Agent import Agent
from Network import Network
import numpy as np, numpy
from sys import argv
import time

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# screen constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
CELL_SIZE = 60


class SnortGameVisualizer:
    PVP = 0
    CPU_VS_CPU = 1
    PLAYER_VS_CPU = 2
    def __init__(self, board, player_type, model_type):
        self.board: Board = board
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snort Game")
        self.clock = pygame.time.Clock()

        self.turn = Board.RED

        self.font = pygame.font.SysFont(None, 36)

        self.winner = -1
        
        self.player_type = player_type
        
        self.agents = {Board.RED: None, Board.BLUE: None}
        
        if (self.player_type == self.PLAYER_VS_CPU):
            self.agents[Board.RED] = Agent()
            self.agents[Board.BLUE] = "Player"
        elif self.player_type == self.CPU_VS_CPU:
            self.agents[Board.RED] = Agent(Network(model_type))
            self.agents[Board.BLUE] = Agent()

    def draw_board(self):
        self.screen.fill(WHITE)
        rows = self.board.board.shape[0]
        cols = self.board.board.shape[1]
        for row in range(rows):
            for col in range(cols):
                piece = self.board.board[row, col]

                if piece == Board.RED:
                    color = RED
                elif piece == Board.BLUE:
                    color = BLUE
                elif piece == Board.BLACK:
                    color = BLACK
                else:
                    color = WHITE

                # players' nodes on the board
                pygame.draw.rect(
                    self.screen,
                    color,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    0,
                )

                # grid
                pygame.draw.rect(
                    self.screen,
                    BLACK,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1,
                )

    def draw_legal_moves(self, player):
        legal_move_list = []
        if player == Board.RED:
            legal_move_list = self.board.red_legal_moves
        else:
            legal_move_list = self.board.blue_legal_moves

        rows, cols = numpy.where(legal_move_list == True)
        for row, col in zip(rows, cols):
            if self.board.board[row, col] == Board.EMPTY:

                pygame.draw.circle(
                    self.screen,
                    GREEN,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        row * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 8,
                )

    def run(self):
        running = True
        while running:
            if (self.player_type == self.PVP):
                running = self.handle_events()
            elif self.player_type == self.PLAYER_VS_CPU:
                if self.turn == Board.BLUE:
                    running = self.handle_events()
                else:
                    if self.winner == -1:
                        move = self.agents[Board.RED].best_move_to_do(self.board, self.turn)
                        self.handle_click(move, True)
                    
            elif self.player_type == self.CPU_VS_CPU:
                if self.winner == -1:
                    move = self.agents[self.turn].best_move_to_do(self.board, self.turn)
                    self.handle_click(move, True)
                    
            self.draw_board()
            self.draw_legal_moves(self.turn)
            if self.winner != -1:
                if self.winner == Board.RED:
                    color = RED
                    winner = "red"
                else:
                    color = BLUE
                    winner = "blue"
                self.draw_text(
                    f"{winner} player wins!",
                    color,
                    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
                )
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def draw_text(self, text, color, pos):
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)

        # Add a box behind the text
        box_width = text_rect.width + 20
        box_height = text_rect.height + 20
        box_rect = pygame.Rect(
            (pos[0] - box_width // 2, pos[1] - box_height // 2), (box_width, box_height)
        )
        pygame.draw.rect(self.screen, GRAY, box_rect)
        self.screen.blit(text_surface, text_rect)

    def handle_click(self, pos, is_CPU=False):
        if (is_CPU):
            row = pos[0]
            col = pos[1]
        else:
            col = pos[0] // CELL_SIZE
            row = pos[1] // CELL_SIZE

        if self.board.legal_move(self.turn, (row, col)):
            # make move
            self.board.make_move(self.turn, (row, col))

            # check for winner
            if self.board.end(self.turn):
                self.winner = self.turn

            # update turn
            self.turn = self.board.switch_player(self.turn)

        else:
            # display error message for 1 second then continue
            self.draw_text(
                "Invalid move! Try again", RED, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )
            pygame.display.flip()
            pygame.time.wait(10)

            # redraw board without the error message
            self.draw_board()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:  # backspace for a hacky unmake move
                    self.board.unmake_last_move()
                    self.turn = self.board.switch_player(self.turn)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # left mouse button
                    self.handle_click(pygame.mouse.get_pos())
        return True


def main():
    # create a board (example board, use real board later)
    board = Board()

    visualizer = SnortGameVisualizer(board,SnortGameVisualizer.PLAYER_VS_CPU, argv[1] if len(argv)==2 else 0)
    visualizer.run()


# Example
if __name__ == "__main__":
    main()