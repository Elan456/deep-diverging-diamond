import pygame 
import math 
from .utils import draw_line_as_polygon

class Car:
    def __init__(self, router, start_node, end_node, spawn_tick, ddi):
        self.tick = -1
        self.ddi = ddi
        self.spawn_tick = spawn_tick
        self.router = router
        self.start_node = start_node
        self.end_node = end_node
        self.is_active = False
        self.done = False
        self.facing_direction = [0, 0] # The direction the car is facing
        self.facing_angle = 0  # The angle the car is facing
        self.current_node = start_node

        self.route = self.router.get_route(start_node, end_node)  # List of nodes to drive to
        self.route_index = 0  # The index of the current node in the route we are on.
        self.ending_tick = 0
        self.crashed = False 
        self.other = None  

        self.at_light = False # If I'm one os away from a light, must be updated every .update()

    def crash(self, other=None):
        """
        Other is the other car that I crashed into
        """
        self.crashed = True 
        self.other = other

    def finish(self):
        self.done = True 
        self.ending_tick = self.tick
        # Remove myself from the occupation map
        if self.ddi.occupations[(self.current_node.x, self.current_node.y)].occupant == self:
            self.ddi.occupations[(self.current_node.x, self.current_node.y)].occupant = None
    

    def update(self):
        """
        Call this on each tick
        """

        self.tick += 1
        self.current_node = self.route[self.route_index]

        if self.crashed:
            self.finish()
            return

        if self.route_index == len(self.route) - 1:
            self.finish()
            return

        self.next_node = self.route[self.route_index + 1]

        self.at_light = self.next_node.is_light

        self.facing_direction = [self.next_node.x - self.current_node.x, self.next_node.y - self.current_node.y]
        angle_deg = math.degrees(math.atan2(self.facing_direction[1], self.facing_direction[0]))
        self.facing_angle = (360 - angle_deg) % 360
        # If I'm not active yet, check if I should be
        if not self.is_active:
            if self.tick >= self.spawn_tick:
                self.is_active = True
            return 
        
        # Increase the route index if we can move to the next node
        if self.ddi.occupations[(self.next_node.x, self.next_node.y)].can_enter(self):
            self.route_index += 1
            # set that I am occupying the next node and not the current node
            self.ddi.occupations[(self.next_node.x, self.next_node.y)].occupant = self
            self.ddi.occupations[(self.current_node.x, self.current_node.y)].occupant = None


    def draw(self, surface):
        draw_x = self.current_node.get_render_x()
        draw_y = self.current_node.get_render_y()
        length = 30
        width = 3
        explosion_radius = 20

        if self.is_active:
            # Draw a line pointing in self.direction from draw_x, draw_y
            start_line = (draw_x, draw_y)
            end_line = (draw_x + self.facing_direction[0] * length, draw_y + self.facing_direction[1] * length)
            draw_line_as_polygon(surface, start_line, end_line, width, (0, 0, 0), aa=False)

            # Draw a smaller skinny line not quite as long to show the direction
            pygame.draw.line(surface, (255, 0, 255), (draw_x, draw_y), (draw_x + self.facing_direction[0] * length * 0.5, draw_y + self.facing_direction[1] * length * 0.5), 1)

        if self.crashed:
            # Draw a circle with red on the outside, yellow on the inside
            pygame.draw.circle(surface, (255, 0, 0), (draw_x, draw_y), explosion_radius)
            pygame.draw.circle(surface, (255, 255, 0), (draw_x, draw_y), explosion_radius - 5)

            # Draw a line to my crash partner
            if self.other:
                pygame.draw.line(surface, (255, 0, 0), (draw_x, draw_y), (self.other.current_node.get_render_x(), self.other.current_node.get_render_y()), 5)
