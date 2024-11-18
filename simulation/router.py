"""
Helps cars to live-navigate through the simulation.
"""

import pygame
import math 

from simulation.car import Car 
from simulation.road_segment import RoadSegment, Node, OccupationSection

from typing import List, Dict, Tuple

from . import lane_defs

class Router:
    def __init__(self, road_segments, routes: List[Tuple[Node, Node]]):
        """
        :param road_segments: List of RoadSegment objects
        :param routes: List of tuples, where each tuple is a start and end node
        """
        self.routes = routes
        self.road_segments: List[RoadSegment] = road_segments
        self.full_routes: Dict[Tuple: List[OccupationSection]] = {}  # The key is a tuple of the start and end node, the value is a list of nodes representing each occupation section

        self.node_neighbors: Dict[Node, List[Node]] = {}
        self.populate_neighbors()
        self.generate_full_routes()

        self.render_tick = 0

    def get_route(self, start: Node, end: Node):
        """
        Get the route from start to end
        """
        return self.full_routes[(start, end)]

    def populate_neighbors(self):
        """
        For every node, populate the list of its neighbors
        """
        for segment in self.road_segments:
            if segment.start not in self.node_neighbors:
                self.node_neighbors[segment.start] = []

            self.node_neighbors[segment.start].append(segment.end)

    def generate_full_routes(self):
        """
        For every start, end pair, find the list of nodes that make up the route
        """
        for start, end in self.routes:
            route = self.find_route(start, end)
            occupation_sections_route = [
                OccupationSection(route[0].x, route[0].y)
            ]

            # The occupation sections should be evenly spaced at 1 unit apart 

            for i in range(1, len(route)):
                s_start = route[i - 1]
                s_end = route[i]

                dx = s_end.x - s_start.x
                dy = s_end.y - s_start.y

                num_sections = int(math.sqrt(dx ** 2 + dy ** 2) * 2)

                for j in range(num_sections):
                    occupation_sections_route.append(OccupationSection(s_start.x + dx * j / num_sections, s_start.y + dy * j / num_sections))
            occupation_sections_route.append(OccupationSection(route[-1].x, route[-1].y))
            self.full_routes[(start, end)] = occupation_sections_route
            

        # Sort the routes by start_node
        self.full_routes = dict(sorted(self.full_routes.items(), key=lambda x: x[0][0]))

    def find_route(self, start: Node, end: Node) -> List[Node]:
        """
        Find the route from start to end using the neighbors and a BFS algorithm (there should only be one possible route)
        """
        visited = set()
        queue = [[start]]

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node == end:
                return path

            if node in visited:
                continue

            visited.add(node)


            try:
                for neighbor in self.node_neighbors[node]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
            except KeyError:
                pass  # No neighbors

        return []


    def draw_route_labels(self, screen):
        """
        At the beginning of each route, put a text label
        e.g. E-1 == East-bound lane 1
        """

        font = pygame.font.Font(None, 36)

        for i, (start, end) in enumerate(self.routes):
            route = self.full_routes[(start, end)]
            start = route[0]
            end = route[-1]
            
            # Draw a letter label at start
            label = ""
            
            east_start_x = lane_defs.start_east_bound_lane_1.x
            west_start_x = lane_defs.start_west_bound_lane_1.x
            south_start = lane_defs.from_south_bound_node
            north_start = lane_defs.from_north_bound_node

            lane_1 = [lane_defs.start_east_bound_lane_1, lane_defs.start_west_bound_lane_1]
            lane_2 = [lane_defs.start_east_bound_lane_2, lane_defs.start_west_bound_lane_2]
            lane_3 = [lane_defs.start_east_bound_lane_3, lane_defs.start_west_bound_lane_3]

            if start.x == east_start_x:
                label += "E"
            elif start.x == west_start_x:
                label += "W"

            if start in lane_1:
                label += "-1"
            elif start in lane_2:
                label += "-2"
            elif start in lane_3:
                label += "-3"

            if start == south_start:
                label = "S"
            elif start == north_start:
                label = "N"

            text = font.render(label, True, (255, 255, 255))
            screen.blit(text, (start.get_render_x() - 20, start.get_render_y() - 20))






    def draw_routes(self, screen):
        """
        Draw the routes on the screen
        """
        self.render_tick += 1
        routes = list(self.full_routes.keys())

        # Draw all the routes in a faded grey color
        for route in routes:
            nodes = self.full_routes[route]

            for i in range(len(nodes) - 1):
                pygame.draw.line(screen, (50, 50, 50),
                                (nodes[i].get_render_x(), nodes[i].get_render_y()), 
                                (nodes[i + 1].get_render_x(), nodes[i + 1].get_render_y()), 5)
                
                nodes[i].draw(screen)
                


        display_route = routes[int(self.render_tick / 100) % len(routes)]

        nodes = self.full_routes[display_route]

        for i in range(len(nodes) - 1):
            pygame.draw.line(screen, (0, 100, 0),
                            (nodes[i].get_render_x(), nodes[i].get_render_y()), 
                            (nodes[i + 1].get_render_x(), nodes[i + 1].get_render_y()), 5)
            
           
            
        # Draw a green circle at the start and a red circle at the end
        pygame.draw.circle(screen, (0, 255, 0), 
                            (nodes[0].get_render_x(), nodes[0].get_render_y()), 5)
        
        pygame.draw.circle(screen, (255, 0, 0),
                            (nodes[-1].get_render_x(), nodes[-1].get_render_y()), 5)
        
        self.draw_route_labels(screen)
            