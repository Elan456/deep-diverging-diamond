import pygame
from typing import List
import csv 
from .ddi import DDI

pygame.init()
pygame.font.init()

class Simulation:

    class State:
        def __init__(self, current_tick, current_induction_plate_states,
                      induction_plate_last_activated,
                        crash_occurred,
                        all_cars_done, clock_signal):

            self.current_tick = current_tick  # The current tick of the simulation
            self.current_induction_plate_states = current_induction_plate_states  # A list of 1s and 0s representing if each induction plate is occupied or not
            self.induction_plate_last_activated = induction_plate_last_activated  # A list of how many ticks ago the induction plates were last activated
            self.crash_occurred = crash_occurred  # Did a crash occur this tick?
            self.all_cars_done = all_cars_done  # Are all the cars done?

            self.induction_plate_last_activated_with_clock_signal = induction_plate_last_activated + [clock_signal]

    @staticmethod
    def read_input_file(inputFile):
        """
        Given the csv file, returns a list of route index and start times. 
        """
        cars = []
        with open(inputFile, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Skip empty rows
                if not row[0] or not row[1]:
                    continue
                cars.append((int(row[0]), int(row[1])))
        return cars

    def __init__(self, render=False, render_frequency=5):
        self.render = render
        self.render_frequency = render_frequency

        if self.render:
            self.screen = pygame.display.set_mode((1200, 800))
            pygame.display.set_caption("Deep Diverging Diamond")
            self.clock = pygame.time.Clock()
            self.running = True

        self.ddi = None
        self.scenario = None # A list of cars and their spawn times
        self.induction_times = [] # A list of times for how many ticks ago the induction plates were triggered, 0 means something is on it, 100 means never triggered
        self.tick = 0 

    def set_scenario(self, cars=None, inputFile=None):
        """
        Sets the scenario for the simulation
        I.e. defines where and when the cars will spawn
        """
        self.scenario = None 
        if cars:
            self.scenario = cars
        elif inputFile:
            self.scenario = self.read_input_file(inputFile)
        else:
            raise ValueError("Must provide either cars or inputFile")

        if cars and inputFile:
            raise ValueError("Cannot provide both cars and inputFile")
        
        self.reset()

    def reset(self):
        """
        Keeping the current scenario, reset the simulation
        """
        self.ddi = DDI(self.scenario)
        self.induction_times = [100] * 12
        self.tick = 0
        self.clock_signal = 0

    def step(self, episode=0):
        """
        Steps the simulation forward one tick

        Pass in the episode number, so the sim knows when to render
        """
        self.tick += 1
        if self.tick % 10 == 0:
            self.clock_signal = 1 - self.clock_signal
        self.ddi.update()
        if self.ddi.is_done():
            self.running = False

        induction_plate_states = self.ddi.get_induction_plate_states()
        for i in range(12):
            if induction_plate_states[i] == 1:
                self.induction_times[i] = 0
            else:
                self.induction_times[i] += 1

        if self.render and episode % self.render_frequency == 0:
            self.screen.fill((128, 128, 128))
            self.ddi.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(15)

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
           

    def get_state(self) -> List[int]:
        """
        Returns the current state vector of the simulation
        
        Represented as a list of 1 and 0 for each induction plate 
        """
        induction_plate_states = self.ddi.get_induction_plate_states()
        crash_occurred = self.ddi.get_crash_just_occurred()
        return self.State(self.tick, 
            induction_plate_states,
              self.induction_times, crash_occurred, self.ddi.is_done(), self.clock_signal)

    def get_average_time(self):
        """
        Returns the average time a car took to finish the simulation
        """
        return self.ddi.get_average_time()
    
    def apply_action(self, action: List[int]):
        """
        Apply the given action to the simulation

        Action is 12 boolean values representing either green or red for each light
        """
        self.ddi.set_light_states(action)


    def run(self):
        if self.render:
            paused = False 

            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        if event.key == pygame.K_RIGHT and paused:
                            self.step()
                    # If they click on a light, toggle it
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = pygame.mouse.get_pos()
                        if x > 800 and x < 850 and y > 100 and y < 700:
                            light_index = (y - 100) // 50
                            self.ddi.light_states[light_index] = 1 if self.ddi.light_states[light_index] == 0 else 0
                        
                if not paused:
                    self.step()
                    print(self.get_average_time())

                self.screen.fill((128, 128, 128))
                self.ddi.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(5)
                if self.ddi.is_done():
                    self.running = False

                
        self.ddi.final_stats()
