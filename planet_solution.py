from collections import defaultdict
import numpy as np
import sys


class Graph:

    def __init__(self, t_stations):
        # Using Kruskal's Algorithm to find the number of stations
        self.n = t_stations
        self.graph = []

        # Using Dijkstra's Algorithm to find the safest path
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, node_a, node_b, distance):
        self.graph.append([node_a, node_b, distance])

    def find(self, parent, node):
        # Function to find parent nodes of a node
        if parent[node] == node:
            return node
        return self.find(parent, parent[node])

    def union(self, parent, rank, x, y):
        root_a = self.find(parent, x)
        root_b = self.find(parent, y)

        if rank[root_a] < rank[root_b]:
            parent[root_a] = root_b
        elif rank[root_a] > rank[root_b]:
            parent[root_b] = root_a
        else:
            parent[root_b] = root_a
            rank[root_a] += 1

    def m_s_t(self):
        # Function to create a minimum spanning tree for the graph and
        # returns a list of values[node_a, node_b, distance] that belong to the minimum spanning tree

        # List to store minimum spanning tree
        results = []

        i, j = 0, 0

        # sort graph according to edge size
        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent, rank = [], []

        # create n subsets with single elements
        for node in range(self.n):
            parent.append(node)
            rank.append(0)

        # number of edges to be taken is n-1
        while j < self.n - 1:
            node_a, node_b, distance = self.graph[i]
            i += 1
            x = self.find(parent, node_a)
            y = self.find(parent, node_b)

            if x is not y:
                j += 1
                results.append([node_a, node_b, distance])
                self.union(parent, rank, x, y)

        return results

    def find_safest_path(self, start, end, span_tree):
        # Function to find the safest path using Dijkstra's Algorithm.

        def _add_edge(span_tree):
            for edge in span_tree:
                node_a = edge[0]
                node_b = edge[1]
                distance = edge[2]
                self.edges[node_a].append(node_b)
                self.edges[node_b].append(node_a)
                self.weights[(node_a, node_b)] = distance
                self.weights[(node_b, node_a)] = distance

        # initialise graph for dijkstra implementation
        _add_edge(span_tree)
        safest_paths = {start: (None, 0)}
        current_node = start
        visited = set()
        distance = 0

        while current_node != end:
            visited.add(current_node)
            destinations = self.edges[current_node]
            distance_to_current_node = safest_paths[current_node][1]

            for next_node in destinations:
                distance = self.weights[(current_node, next_node)] + distance_to_current_node
                if next_node not in safest_paths:
                    safest_paths[next_node] = (current_node, distance)
                else:
                    current_shortest_weight = safest_paths[next_node][1]
                    if current_shortest_weight > distance:
                        safest_paths[next_node] = (current_node, distance)

            next_destinations = (
                {node: safest_paths[node] for node in safest_paths if node not in visited}
            )
            if not next_destinations:
                return "route not possible"

            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = safest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]

        return path

    def max_distance(self, path):
        # Function to find the maximum distance between two nodes in a path.
        max_diz = 0.0
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]
            distance = self.weights[(node_a, node_b)]
            if distance > max_diz:
                max_diz = distance

        return max_diz


def data(nodez):
    # Function to get the coordinates and nodes of Zearth and the number of nodes from stdin as a .txt file
    # or user inputs

    # Get number of nodes
    t_node = int(nodez[1])
    
    # Check value of t_node
    if (t_node < 1) or (t_node > 2000):
        raise Exception('Number of stations should be within 1 and 2000')

    earth = [0.0, 0.0, 0.0]
    Zearth = coordz(nodez[0])

    nodes = [earth, Zearth]

    # get coordinates excluding t_nodes and Zearth
    for i in range(t_node):
        coord = coordz(nodez[i + 2])
        nodes.append(coord)

    return nodes, t_node + 1 + 1  # include earth and Zearth


def coordz(nod):
    # Function to extract coordinate values from the string read from the .txt file

    # List to store coordinates
    coordinate = []

    # split string in terms of whitespace
    dat = nod.split()
    # convert string to floats
    for i in dat:
        i = float(i)

        # Check for values
        if (i < -10000.00) | (i > 10000.00):
            raise Exception('Coordinate values must be within -10000.00, 10000.00')

        coordinate.append(i)

    return coordinate


def routez(start, end, span_tree):
    # Function to use a minimal spanning tree to find the route from start to end.
    span_tree = sorted(span_tree, key=lambda item: item[0])


def get_distance(coord1, coord2):
    # Function to calculate the distance between nodes

    # convert to numpy array form as list cannot do subtraction
    coord1 = np.asarray(coord1)
    coord2 = np.asarray(coord2)

    return np.linalg.norm(coord1 - coord2)


if __name__ == "__main__":
    # load file containing data
    nodez = sys.stdin.read().split('\n')

    # Retrieve data
    coordinates, t_nodes = data(nodez)

    # initialise Graph object for constructing minimum spanning tree
    span_tree = Graph(t_nodes)

    # load edges to graph. Assume fully connected
    # node 0 is earth, node 1 is Zearth
    for i, coord1 in enumerate(coordinates):
        for j, coord2 in enumerate(coordinates):
            if (i != j) & (i < j):
                distance = get_distance(coord1, coord2)
                span_tree.add_edge(i, j, distance)

    results = span_tree.m_s_t()

    # Get the safest path from span_tree
    route = span_tree.find_safest_path(0, 1, results)

    # Get the maximum distance of the safest path
    max_dist = span_tree.max_distance(route)

    # print maximum distance
    sys.stdout.write("%.2f" % max_dist)
