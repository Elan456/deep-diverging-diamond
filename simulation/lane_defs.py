from .road_segment import Node, RoadSegment

to_south_bound_node = Node(2.5, 9)
from_south_bound_node = Node(2.5, 0)

to_north_bound_node = Node(9.5, 0)
from_north_bound_node = Node(9.5, 9)

start_east_bound_lane_1 = Node(0, 5)
start_east_bound_lane_2 = Node(0, 6)
start_east_bound_lane_3 = Node(0, 7)

end_east_bound_lane_1 = Node(12, 5)
end_east_bound_lane_2 = Node(12, 6)
end_east_bound_lane_3 = Node(12, 7)

start_west_bound_lane_1 = Node(12, 2)
start_west_bound_lane_2 = Node(12, 3)
start_west_bound_lane_3 = Node(12, 4)

end_west_bound_lane_1 = Node(0, 2)
end_west_bound_lane_2 = Node(0, 3)
end_west_bound_lane_3 = Node(0, 4)

east_bound_lane_1_nodes_north_bound = [
    start_east_bound_lane_1,
    Node(1, 5),
    Node(4, 2),
    Node(8, 2),
    to_north_bound_node,
]

east_bound_lane_1_east_bound_nodes = [
    start_east_bound_lane_1,
    Node(1, 5),
    Node(4, 2),
    Node(8, 2),
    Node(11, 5),
    end_east_bound_lane_1,
]

east_bound_lane_2_nodes = [
    start_east_bound_lane_2,
    Node(1, 6),
    Node(4, 3),
    Node(8, 3),
    Node(11, 6),
    end_east_bound_lane_2,
]

east_bound_lane_3_nodes = [
    start_east_bound_lane_3,  # Start
    Node(1, 7),  # Western intersection
    Node(4, 4),  # West-side Bridge 3rd lane 
    Node(8, 4),  # East-side Bridge 3rd lane
    Node(11, 7), # Eastern intersection east side 
    end_east_bound_lane_3, # End
]

east_bound_to_south_bound_lane_3_nodes = [
    start_east_bound_lane_3,  # Start
    Node(1, 7),  # Western intersection
    to_south_bound_node,
]

west_bound_lane_1_nodes = [
    start_west_bound_lane_1,  # Start
    Node(11, 2),  # Eastern intersection
    Node(8, 5),  # East-side Bridge 3rd lane
    Node(4, 5),  # West-side Bridge 3rd lane
    Node(1, 2),
    end_west_bound_lane_1,
]

west_bound_lane_1_to_north_bound_nodes = [
    start_west_bound_lane_1,  # Start
    Node(11, 2),  # Eastern intersection
    to_north_bound_node,
]

west_bound_lane_2_nodes = [
    start_west_bound_lane_2,
    Node(11, 3),
    Node(8, 6),
    Node(4, 6),
    Node(1, 3),
    end_west_bound_lane_2,
]

west_bound_lane_3_south_bound_nodes = [
    start_west_bound_lane_3,
    Node(11, 4),
    Node(8, 7),
    Node(4, 7),
    to_south_bound_node,
]

west_bound_lane_3_west_bound_nodes = [
    start_west_bound_lane_3,
    Node(11, 4),
    Node(8, 7),
    Node(4, 7),
    Node(1, 4),
    end_west_bound_lane_3,
]

from_south_bound_to_west_bound_nodes = [
    from_south_bound_node,
    Node(1, 2),
    end_west_bound_lane_1,
]

from_south_bound_to_east_bound_nodes = [
    from_south_bound_node,
    Node(4, 2),
    Node(8, 2),
    Node(11, 5),
    end_east_bound_lane_1,
]

from_north_bound_to_east_bound_nodes = [
    from_north_bound_node,
    Node(11, 7),
    end_east_bound_lane_3,
]

from_north_bound_to_west_bound_nodes = [
    from_north_bound_node,
    Node(8, 7),
    Node(4, 7),
    Node(1, 4),
    end_west_bound_lane_3,
]

node_series = [
    east_bound_lane_1_nodes_north_bound,
    east_bound_lane_2_nodes,
    east_bound_lane_3_nodes,
    west_bound_lane_1_nodes,
    west_bound_lane_2_nodes,
    west_bound_lane_3_south_bound_nodes,
    east_bound_to_south_bound_lane_3_nodes,
    west_bound_lane_1_to_north_bound_nodes,
    from_south_bound_to_west_bound_nodes,
    from_south_bound_to_east_bound_nodes,
    from_north_bound_to_east_bound_nodes,
    from_north_bound_to_west_bound_nodes,

    east_bound_lane_1_east_bound_nodes,
    west_bound_lane_3_west_bound_nodes,
]

# A route is defined as the starting node and the ending node
routes = [
    # Starting from east-bound (5 of these)
    (start_east_bound_lane_1, to_north_bound_node),  # 0
    (start_east_bound_lane_1, end_east_bound_lane_1), # 1
    (start_east_bound_lane_2, end_east_bound_lane_2), # 2
    (start_east_bound_lane_3, to_south_bound_node), # 3
    (start_east_bound_lane_3, end_east_bound_lane_3), # 4

    # Starting from west-bound (5 of these)
    (start_west_bound_lane_1, to_north_bound_node), # 5
    (start_west_bound_lane_1, end_west_bound_lane_1), # 6
    (start_west_bound_lane_2, end_west_bound_lane_2), # 7
    (start_west_bound_lane_3, to_south_bound_node),  # 8
    (start_west_bound_lane_3, end_west_bound_lane_3),  # 9

    # Starting from south-bound (3 of these)
    (from_south_bound_node, end_west_bound_lane_1), # 10
    (from_south_bound_node, end_east_bound_lane_1),  # 11
    (from_south_bound_node, to_north_bound_node), # 12


    # Starting from north-bound (3 of these)
    (from_north_bound_node, end_west_bound_lane_3), # 13
    (from_north_bound_node, end_east_bound_lane_3), # 14
    (from_north_bound_node, to_south_bound_node), # 15
]


LIGHTS = [
            (1.0, 5.0), (1.0, 6.0), (1.0, 7.0), 
            (11.0, 2.0), (11.0, 3.0), (11.0, 4.0),
            (8.0, 2.0), (8.0, 3.0), (8.0, 4.0),
            (4.0, 5.0), (4.0, 6.0), (4.0, 7.0),
        ]

print("Route count:", len(routes))

road_segments = []
for series in node_series:
    for i in range(len(series) - 1):
        road_segments.append(RoadSegment(series[i], series[i + 1]))
