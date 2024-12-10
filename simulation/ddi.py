"""
Diverging Diamond Interchange (DDI) simulation
"""
import random
import pygame 
from .road_segment import RoadSegment, Node
from .lane_defs import road_segments, routes
from . import lane_defs as ld
from .router import Router
from .car import Car
import csv

class DDI:
    def __init__(self, inputFile):
        self.router = Router(road_segments, routes)
        self.cars = []
        if inputFile:
            with open(inputFile, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    self.cars.append(Car(self.router, routes[int(row[0])][0], routes[int(row[0])][1], int(row[1]), self))
        else:
            self.cars = [
                Car(self.router, routes[0][0], routes[0][1], 0, self),
                Car(self.router, routes[0][0], routes[0][1], 1, self),
                Car(self.router, routes[0][0], routes[0][1], 1, self),
                Car(self.router, ld.from_south_bound_node, ld.end_east_bound_lane_1, 5, self)
            ]
        self.occupations = {}  # Map of location (x,y) : occupation section
        for os in self.router.all_occupation_sections:
            self.occupations[(os.x, os.y)] = os 
        self.all_cars = self.cars.copy()

    def draw(self, screen):
        # Draw the grid of 50x50
        for i in range(0, 13):
            pygame.draw.line(screen, (50, 50, 50), (100, 100 + i * 50), (700, 100 + i * 50), 1)
            pygame.draw.line(screen, (50, 50, 50), (100 + i * 50, 100), (100 + i * 50, 700), 1)

        self.router.draw_routes(screen)

        for car in self.cars:
            car.draw(screen)
    def toggle_lights(self):
        for os in self.router.all_occupation_sections:
            if os.is_light:
                if bool(random.getrandbits(1)):
                    os.toggle_light()

    def update(self):
        for car in self.cars:
            car.update()
        self.toggle_lights()
        # Cull cars that are done
        self.cars = [car for car in self.cars if not car.done]
    
    def is_done(self):
        return len(self.cars) == 0
    
    def final_stats(self):
        print("Simulation finished")
        print("Final stats:")
        car_count = 0
        running_average = 0
        for car in self.all_cars:
            running_average = running_average * .8 + (car.ending_tick - car.spawn_tick) * .2
            car_count += 1
            # print(f"Car {car_count} took {car.ending_tick - car.spawn_tick} ticks to finish")
        print(f"Average time to finish: {running_average}")

        
