import pygame 
import math 

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

        self.facing_direction = [0, 0]  # The direction the car is facing
        self.current_node = start_node

        

        self.route = self.router.get_route(start_node, end_node)  # List of nodes to drive to
        self.route_index = 0  # The index of the current node in the route we are on.

    def finish(self):
        self.done = True 
        # Remove myself from the occupation map
        if self.ddi.occupations[(self.current_node.x, self.current_node.y)].occupant == self:
            self.ddi.occupations[(self.current_node.x, self.current_node.y)].occupant = None

    def update(self):
        """
        Call this on each tick
        """

        self.tick += 1
        self.current_node = self.route[self.route_index]

        if self.route_index == len(self.route) - 1:
            self.finish()
            return

        self.next_node = self.route[self.route_index + 1]

        self.facing_direction = [self.next_node.x - self.current_node.x, self.next_node.y - self.current_node.y]

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
        length = 40
        width = 20

        if self.is_active:
            # Draw a line pointing in self.direction from draw_x, draw_y
            pygame.draw.line(surface, (0, 0, 0), (draw_x, draw_y), (draw_x + self.facing_direction[0] * length, draw_y + self.facing_direction[1] * length), width)

            # Draw a smaller skinny line not quite as long to show the direction
            pygame.draw.line(surface, (255, 0, 0), (draw_x, draw_y), (draw_x + self.facing_direction[0] * length * 0.8, draw_y + self.facing_direction[1] * length * 0.8), 1)


