"""
Diverging Diamond Interchange (DDI) simulation
"""
import pygame 
from .road_segment import RoadSegment, Node
from .lane_defs import road_segments, routes
from . import lane_defs as ld
from .router import Router
from .car import Car





class DDI:
    def __init__(self):
        self.router = Router(road_segments, routes)
        self.cars = [
            Car(self.router, routes[0][0], routes[0][1], 0, self),
            Car(self.router, routes[0][0], routes[0][1], 1, self),
            Car(self.router, routes[0][0], routes[0][1], 1, self),
            Car(self.router, ld.from_south_bound_node, ld.end_east_bound_lane_1, 5, self)
        ]

        self.occupations = {}  # Map of location (x,y) : occupation section
        for os in self.router.all_occupation_sections:
            self.occupations[(os.x, os.y)] = os 

    def draw(self, screen):
        # Draw the grid of 50x50
        for i in range(0, 13):
            pygame.draw.line(screen, (50, 50, 50), (100, 100 + i * 50), (700, 100 + i * 50), 1)
            pygame.draw.line(screen, (50, 50, 50), (100 + i * 50, 100), (100 + i * 50, 700), 1)


        # for segment in road_segments:
        #     segment.draw(screen)

        self.router.draw_routes(screen)

        for car in self.cars:
            car.draw(screen)

    def update(self):
        for car in self.cars:
            car.update()

        # Cull cars that are done
        self.cars = [car for car in self.cars if not car.done]