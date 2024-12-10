import pygame
from .ddi import DDI

pygame.init()
pygame.font.init()

class Simulation:
    def __init__(self, inputFile, render=False):
        self.render = render

        if self.render:
            self.screen = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption("Deep Diverging Diamond")
            self.clock = pygame.time.Clock()
            self.running = True

        if inputFile:
            self.ddi = DDI(inputFile)
        else:
            self.ddi = DDI(None)

    def run(self):
        if self.render:
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

                self.ddi.update()

                self.screen.fill((128, 128, 128))
                self.ddi.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(5)
                if self.ddi.is_done():
                    self.running = False
        self.ddi.final_stats()
