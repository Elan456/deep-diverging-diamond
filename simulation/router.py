"""
Helps cars to live-navigate through the simulation.
"""

import pygame

from simulation.car import Car 
from simulation.road_segment import RoadSegment, Node

from typing import List, Dict, Tuple

class Router:
    def __init__(self, road_segments, routes: List[Tuple[Node, Node]]):
        """
        :param road_segments: List of RoadSegment objects
        :param routes: List of tuples, where each tuple is a start and end node
        """
        self.routes = routes
        self.road_segments: List[RoadSegment] = road_segments
        self.full_routes: Dict[Tuple: List[Node]] = {}  # The key is a tuple of the start and end node, the value is a list of nodes 
        self.cars = []

        self.node_neighbors: Dict[Node, List[Node]] = {}
        self.populate_neighbors()
        self.generate_full_routes()

        self.render_tick = 0

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
            self.full_routes[(start, end)] = route

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

    def add_car(self, car: Car):
        self.cars.append(car)

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
            