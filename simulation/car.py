import pygame 
import math 

class Car:
    def __init__(self, router, start_node, end_node, spawn_tick):
        self.tick = 0
        self.spawn_tick = spawn_tick
        self.router = router
        self.start_node = start_node
        self.end_node = end_node
        self.is_active = False

        self.direction = [0, 0]  # The direction the car is facing
        self.current_node = start_node

        

        self.route = self.router.get_route(start_node, end_node)  # List of nodes to drive to
        self.route_index = 0  # The index of the current node in the route we are on.

    def update(self):
        """
        Call this on each tick
        """
        
        self.tick += 1
        self.next_node = self.route[self.route_index]
        self.current_node = self.route[self.route_index - 1]

        self.direction = [self.next_node.x - self.current_node.x, self.next_node.y - self.current_node.y]

        # If I'm not active yet, check if I should be
        if not self.is_active:
            if self.tick >= self.spawn_tick:
                self.is_active = True
            return 
        
        # Move towards the next node 

    def draw(self, surface):
        pass 

        # draw_x = self.current_node.get_render_x()
        # draw_y = self.current_node.get_render_y()

        # if self.is_active:
        #     print("drawing")
        #     # Draw a line in the direction of the next node
        #     pygame.draw.line(surface, (0, 0, 0), (self.x, self.y), (self.x + self.direction[0] * 50, self.y + self.direction[1] * 50), 25)

        #     # Draw a little circle at the start and end
        #     pygame.draw.circle(surface, (0, 100, 0), (self.x, self.y), 5)

        #     # Draw a little arrow from the middle back to the front
        #     mid_x = self.x + self.direction[0] * 25
        #     mid_y = self.y + self.direction[1] * 25

        #     pygame.draw.line(surface, (100, 0, 0), (self.x, self.y), (mid_x, mid_y), 5)



