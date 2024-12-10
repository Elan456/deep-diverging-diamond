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

block_font = pygame.font.Font(None, 36)

class DDI:
    def __init__(self, scenario):
        print("Scenario: ", scenario)
        self.router = Router(self, road_segments, routes)
        self.cars = []
        for s_car in scenario:  # s_car is a tuple of (route index, start time)
            self.cars.append(Car(self.router, routes[int(s_car[0])][0], routes[int(s_car[0])][1], int(s_car[1]), self))

        self.occupations = {}  # Map of location (x,y) : occupation section
        for os in self.router.all_occupation_sections:
            self.occupations[(os.x, os.y)] = os 
        self.all_cars = self.cars.copy()

        self.crash_just_occurred = False # Updated every tick to see if a crash just occurred

        self.light_states = [0] * 12  # 0 is red, 1 is green

    def draw(self, screen):
        # Draw the grid of 50x50
        for i in range(0, 13):
            pygame.draw.line(screen, (50, 50, 50), (100, 100 + i * 50), (700, 100 + i * 50), 1)
            pygame.draw.line(screen, (50, 50, 50), (100 + i * 50, 100), (100 + i * 50, 700), 1)

        self.router.draw_routes(screen)

        for car in self.cars:
            car.draw(screen)

        # Draw a list of 1 and 0s vertically for the induction plates, and another next to it for the light states
        induction_plate_states = self.get_induction_plate_states()
        light_states = self.light_states
        for i in range(12):
            pygame.draw.rect(screen, (255, 255, 255), (750, 100 + i * 50, 50, 50), 1)
            pygame.draw.rect(screen, (255, 255, 255), (800, 100 + i * 50, 50, 50), 1)
            if induction_plate_states[i] == 1:
                pygame.draw.rect(screen, (0, 0, 0), (751, 101 + i * 50, 48, 48))
            if light_states[i] == 1:
                pygame.draw.rect(screen, (0, 255, 0), (801, 101 + i * 50, 48, 48))
            elif light_states[i] == 0:
                pygame.draw.rect(screen, (255, 0, 0), (801, 101 + i * 50, 48, 48))

            # Draw the number to the right of the blocks
            text = block_font.render(str(i), True, (255, 255, 255))
            screen.blit(text, (870, 100 + i * 50 + 10))

        # Draw labels
        text = block_font.render("Plates", True, (255, 255, 255))
        screen.blit(text, (700, 50))
        text = block_font.render("Light", True, (255, 255, 255))
        screen.blit(text, (850, 50))

    def get_induction_plate_states(self):
        """
        Returns a list of 1 and 0s representing if each induction plate is occupied or not 
        """
        induction_plate_states = [0] * 12

        # Check if a car is waiting at a light, if so, set the induction plate corresponding to that light to 1
        for car in self.cars:
            if car.at_light:
                induction_plate_states[ld.LIGHTS.index((car.next_node.x, car.next_node.y))] = 1

        return induction_plate_states\
        
    def get_crash_just_occurred(self):
        """
        Did a crash occur this tick?
        """
        return self.crash_just_occurred

    def set_light_states(self, light_states):
        """
        Set the light states to the given list of 1s and 0s of length 12
        """
        self.light_states = light_states

    def update(self):
        for car in self.cars:
            car.update()
        
        self.crash_just_occurred = False
        for os in self.router.all_occupation_sections:
            if os.check_collision():
                self.crash_just_occurred = True

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

        
