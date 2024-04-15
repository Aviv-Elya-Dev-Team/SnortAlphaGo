import pygame
from Snort import Snort
from Agent import Agent
from Network import Network
import numpy as np, numpy
from sys import argv
from Config import Config

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# retrieve values from the config file
config = Config.get_config()

# board and cell sizes
CELL_SIZE = int(config.get("GameUI", "cell_size"))
BOARD_SIZE = int(config.get("Snort", "board_size"))

# screen constants
SCREEN_WIDTH = int(config.get("GameUI", "screen_width"))
SCREEN_HEIGHT = int(config.get("GameUI", "screen_height"))

# staring player and colors
STARTING_PLAYER_PVC = config.get("player_vs_cpu", "starting_player")
STARTING_PLAYER_COLOR = int(config.get("GameUI", "starting_player_color"))
SECOND_PLAYER_COLOR = Snort.other_player(STARTING_PLAYER_COLOR)


class SnortGameVisualizer:
    PVP = 0
    CPU_VS_CPU = 1
    PLAYER_VS_CPU = 2

    def __init__(self, game, player_type, model_type_cpu1=0, model_type_cpu2=0):
        self.game: Snort = game
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snort Game")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 36)

        self.winner = -1

        self.player_type = player_type

        self.agents = {STARTING_PLAYER_COLOR: None, SECOND_PLAYER_COLOR: None}

        if self.player_type == self.PLAYER_VS_CPU:
            if STARTING_PLAYER_PVC == "Player":
                self.agents[STARTING_PLAYER_COLOR] = "Player"

                # Agent
                self.agents[SECOND_PLAYER_COLOR] = Agent(
                    self.game,
                    self.game.current_player,
                    Network(model_type_cpu1, self.game.board_size),
                    model_type_cpu1,
                )

                # MCTS Agent
                # self.agents[STARTING_PLAYER] = MCTSAgent(
                #     self.game, self.game.current_player
                # )
            else:
                self.agents[STARTING_PLAYER_COLOR] = Agent(
                    self.game,
                    self.game.current_player,
                    Network(model_type_cpu1, self.game.board_size),
                    model_type_cpu1,
                )
                self.agents[SECOND_PLAYER_COLOR] = "Player"

                # MCTS Agent
                # self.agents[STARTING_PLAYER] = MCTSAgent(
                #     self.game, self.game.current_player
                # )

        elif self.player_type == self.CPU_VS_CPU:
            self.agents[STARTING_PLAYER_COLOR] = Agent(
                Network(model_type_cpu1, BOARD_SIZE), model_type_cpu1
            )
            self.agents[SECOND_PLAYER_COLOR] = Agent(
                Network(model_type_cpu2, BOARD_SIZE), model_type_cpu2
            )

    def draw_board(self):
        self.screen.fill(WHITE)
        rows = self.game.board.shape[0]
        cols = self.game.board.shape[1]
        for row in range(rows):
            for col in range(cols):
                piece = self.game.board[row, col]

                if piece == Snort.RED:
                    color = RED
                elif piece == Snort.BLUE:
                    color = BLUE
                elif piece == Snort.BLACK:
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

    def draw_legal_moves(self):
        legal_move_list = self.game.get_legal_moves(self.game.current_player)

        rows, cols = numpy.where(legal_move_list == True)
        for row, col in zip(rows, cols):
            if self.game.board[row, col] == Snort.EMPTY:

                pygame.draw.circle(
                    self.screen,
                    RED if self.game.current_player == Snort.RED else BLUE,
                    (
                        col * CELL_SIZE + CELL_SIZE // 2,
                        row * CELL_SIZE + CELL_SIZE // 2,
                    ),
                    CELL_SIZE // 8,
                )

    def run(self):
        running = True
        while running:
            if self.player_type == self.PVP:
                running = self.handle_events()
            elif self.player_type == self.PLAYER_VS_CPU:
                if self.game.current_player == STARTING_PLAYER_COLOR:
                    running = self.handle_events()
                else:
                    if self.winner == -1:
                        probabilities = self.agents[SECOND_PLAYER_COLOR].best_move()
                        move = self.agents[SECOND_PLAYER_COLOR].best_move_to_do(
                            probabilities
                        )
                        self.handle_click(move, True)

            elif self.player_type == self.CPU_VS_CPU:
                if self.winner == -1:
                    probabilities = self.agents[self.game.current_player].best_move()
                    move = self.agents[self.game.current_player].best_move_to_do(
                        probabilities
                    )
                    self.handle_click(move, True)

            self.draw_board()
            self.draw_legal_moves()
            if self.winner != -1:
                if self.winner == Snort.RED:
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
        if is_CPU:
            col = pos[1]
            row = pos[0]
        else:
            col = pos[0] // CELL_SIZE
            row = pos[1] // CELL_SIZE

        if self.game.is_legal_move((row, col)):
            # make move
            self.game.make_move((row, col))

            # check for winner
            outcome = self.game.outcome()
            if self.game.outcome() != Snort.ONGOING:
                self.winner = outcome

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
                    self.game.unmake_last_move()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:  # left mouse button
                    self.handle_click(pygame.mouse.get_pos())
        return True


def main():
    # create a board (example board, use real board later)
    game = Snort(STARTING_PLAYER_COLOR, board_size=BOARD_SIZE)
    if len(argv) == 2:
        visualizer = SnortGameVisualizer(game, int(argv[1]))
    if len(argv) == 3:
        visualizer = SnortGameVisualizer(game, int(argv[1]), int(argv[2]))
    if len(argv) == 4:
        visualizer = SnortGameVisualizer(game, int(argv[1]), int(argv[2]), int(argv[3]))
    visualizer.run()


# Example
if __name__ == "__main__":
    main()
