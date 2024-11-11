import pygame 
import numpy as np 

class Node:
    def __init__(self, x, y, light_controlled=False):
        self.x = x
        self.y = y
        self.light_controlled = light_controlled

    def get_render_x(self):
        return self.x * 50 + 100
    
    def get_render_y(self):
        return self.y * 50 + 100

class RoadSegment:
    """
    Represents a single-lane road segment
    """
    def __init__(self, start: Node, end: Node):
        self.start = start
        self.end = end

    def draw(self, surface):
        pygame.draw.line(surface, (255, 255, 255),
                        (self.start.get_render_x(),
                        self.start.get_render_y()), 
                        (self.end.get_render_x(), 
                         self.end.get_render_y()), 5)
        
        # Draw a little circle at the start and end
        pygame.draw.circle(surface, (0, 100, 0), 
                           (self.start.get_render_x(),
                            self.start.get_render_y()), 5)
        
        pygame.draw.circle(surface, (100, 0, 0),
                            (self.end.get_render_x(),
                             self.end.get_render_y()), 5)

        # Draw a little arrow in the middle to show direction
        mid_x = (self.start.get_render_x() + self.end.get_render_x()) // 2
        mid_y = (self.start.get_render_y() + self.end.get_render_y()) // 2
        
        
        direction = np.array([self.end.x - self.start.x, self.end.y - self.start.y])
        direction = direction / np.linalg.norm(direction)

        # if the x is positive, then use blue
        if direction[0] > 0:
            color = (0, 125, 255)
        else: # orange
            color = (255, 125, 0)
        
        pygame.draw.circle(surface, color, (mid_x, mid_y), 5)
        pygame.draw.line(surface, color, (mid_x, mid_y),
                            (mid_x + int(15 * direction[0]), mid_y + int(15 * direction[1])), 3)
        
