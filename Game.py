import pygame
import sys
from Board import Board
import numpy as np, numpy

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

RED_LEGAL_MOVE = (255, 0, 0, 125)

# screen constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
CELL_SIZE = 60


class SnortGameVisualizer:
    def __init__(self, board):
        self.board: Board = board
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snort Game")
        self.clock = pygame.time.Clock()

        self.turn = Board.RED

        self.font = pygame.font.SysFont(None, 36)

    def draw_board(self):
        self.screen.fill(WHITE)
        rows = self.board.board.shape[0]
        cols = self.board.board.shape[1]
        for row in range(rows):
            for col in range(cols):
                pygame.draw.rect(
                    self.screen,
                    BLACK,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1,
                )
                piece = self.board.board[row, col]

                if piece == Board.RED:
                    color = RED
                elif piece == Board.BLUE:
                    color = BLUE
                elif piece == Board.BLACK:
                    color = BLACK
                else:
                    color = WHITE

                pygame.draw.circle(
                    self.screen,
                    color,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        row * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 3,
                )

    def handle_click(self, pos):
        col = pos[0] // CELL_SIZE
        row = pos[1] // CELL_SIZE

        if self.board.legal_move(self.turn, (row, col)):
            # make move
            self.board.make_move(self.turn, (row, col))

            # check for winner
            if self.board.end(self.turn):
                self.draw_text(
                    "Winner winner chicken dinner!",
                    GREEN,
                    (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
                )

            # update turn
            self.turn = self.board.switch_player(self.turn)

        else:
            # display error message for 1 second then continue
            self.draw_text(
                "Invalid move! Try again", RED, (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )
            pygame.display.flip()
            pygame.time.wait(1000)

            # redraw board without the error message
            self.draw_board()

    def draw_illegal_moves(self, player):
        legal_move_list = []
        if player == Board.RED:
            legal_move_list = self.board.red_legal_moves
        else:
            legal_move_list = self.board.blue_legal_moves

        rows, cols = numpy.where(legal_move_list == False)
        for row, col in zip(rows, cols):
            if self.board.board[row, col] == Board.EMPTY:
                pygame.draw.rect(
                    self.screen,
                    RED_LEGAL_MOVE,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    0,
                )

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pressed()[0]:  # left mouse button
                        self.handle_click(pygame.mouse.get_pos())
            self.draw_board()
            self.draw_illegal_moves(self.turn)
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


def main():
    # create a board (example board, use real board later)
    board = Board()

    visualizer = SnortGameVisualizer(board)
    visualizer.run()


# Example
if __name__ == "__main__":
    main()
