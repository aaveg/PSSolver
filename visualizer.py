import pygame
import sys

class Visualizer:
    def __init__(self, width=800, height=600, title="PSS Visualizer"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.running = True

    def draw(self):
        # Override this method to draw custom visuals
        self.screen.fill((255, 255, 255))  # Fill screen with white

    def run(self, fps=60):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.draw()
            pygame.display.flip()
            self.clock.tick(fps)
        pygame.quit()
        sys.exit()
