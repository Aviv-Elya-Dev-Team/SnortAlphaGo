import pygame
import sys
from Board import Board

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# screen constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
CELL_SIZE = 100


class SnortGameVisualizer:
    def __init__(self, board):
        self.board: Board = board
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snort Game")
        self.clock = pygame.time.Clock()

        self.turn = Board.RED

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
        print("Clicked at row:", row, "col:", col)

        if self.board.legal_move(self.turn, (row, col)):
            # make move
            self.board.make_move(self.turn, (row, col))

            # update turn
            if self.turn == Board.RED:
                self.turn == Board.BLUE
            else:
                self.turn == Board.RED

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
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


def main():
    # create a board (example board, use real board later)
    board = Board()

    visualizer = SnortGameVisualizer(board)
    visualizer.run()


# Example
if __name__ == "__main__":
    main()
