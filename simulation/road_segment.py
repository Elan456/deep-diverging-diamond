import pygame 
import numpy as np 

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f"Node({self.x}, {self.y})"
    
    # define less than
    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def get_render_x(self):
        return self.x * 50 + 100
    
    def get_render_y(self):
        return self.y * 50 + 100

class OccupationSection:
    """
    An occupation section is a section of the road that exactly one car can occupy (otherwise an accident occurs)
    These occupation sections are evenly spaced by distance, approx one GridUnit in length

    Situation example: (Z is an occupation section)
         C
         |
    A----Z----B
         |
         D

    If a car heading from A to B is in the occupation section, then no other cars heading in that same
    direction can enter the occupation section until the car leaves it.

    If a car heading from A to B is in the occupation section, a car heading from C to D could enter and
    cause an accident. This is because we are assuming that cars don't check for cross-traffic, only same-direction traffic.

    i.e. No rear-end collisions, only side-swipes and t-bones
    
    Rules:
    - Don't enter the section if a car heading in the same direction is already in the section
    - Don't enter the section if it's red lighted
    - Don't check for cars heading in a perpendicular direction (Assue we don't check for cross-traffic)
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.occupant = None
        self.red_light = False
        self.overlaps = []  # List of other occupation sections that overlap with this one

    def add_overlap(self, other: 'OccupationSection'):
        self.overlaps.append(other)

    def can_enter(self, car):
        """
        Returns true or false if the given car can enter this occupation section

        A cross-traffic car won't trigger a false, only a car heading in the same direction
        It's the traffic light's job to prevent cross-traffic collisions 
        """

        # If the light is red, then no one can enter
        if self.red_light:
            return False

        # If the section is already occupied, then no one can enter
        if self.occupant is not None:
            # Check if the occupant and the car's directions are within 60 degrees of each other (merging or same direction)
            if np.dot(self.occupant.facing_direction, car.facing_direction) < 0.5:
                return False
        
        # Check if any of the overlapping sections are occupied
        for overlap in self.overlaps:
            if overlap.occupant is not None and overlap.occupant.direction == car.direction:
                return False
            
        return True

    def get_render_x(self):
        return self.x * 50 + 100
    
    def get_render_y(self):
        return self.y * 50 + 100

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 0, 0), (self.x * 50 + 100, self.y * 50 + 100, 25, 25), 1)



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
        
