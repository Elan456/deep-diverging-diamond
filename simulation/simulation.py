import pygame
from .ddi import DDI

class Simulation:
    def __init__(self, render=False):
        self.render = render

        if self.render:
            self.screen = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption("Deep Diverging Diamond")
            self.clock = pygame.time.Clock()
            self.running = True

        self.ddi = DDI()

    def run(self):
        if self.render:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                self.screen.fill((128, 128, 128))
                self.ddi.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(60)
