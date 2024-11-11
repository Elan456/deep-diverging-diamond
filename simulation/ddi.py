"""
Diverging Diamond Interchange (DDI) simulation
"""
import pygame 
from .road_segment import RoadSegment, Node
from .lane_defs import road_segments, routes
from .router import Router





class DDI:
    def __init__(self):
        self.router = Router(road_segments, routes)

    def draw(self, screen):
        # Draw the grid of 50x50
        for i in range(0, 13):
            pygame.draw.line(screen, (50, 50, 50), (100, 100 + i * 50), (700, 100 + i * 50), 1)
            pygame.draw.line(screen, (50, 50, 50), (100 + i * 50, 100), (100 + i * 50, 700), 1)


        # for segment in road_segments:
        #     segment.draw(screen)

        self.router.draw_routes(screen)