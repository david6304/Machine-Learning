import os
from typing import Dict, Set


def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    file = open(filename, 'r')
    lines = file.readlines()
    neighbours = {}
    for line in lines:
        v, w = line.split(' ')
        v, w = int(v), int(w.strip())
        if v in neighbours:
            neighbours[v].add(w)
        else:
            neighbours[v] = set([w])
        if w in neighbours:
            neighbours[w].add(v)
        else:
            neighbours[w] = set([v])

    return neighbours

def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    degrees = {}
    for node in graph:
        degrees[node] = len(graph[node])
    return degrees


def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """

    def bfs(g, s, vertices):
        seen = {}
        for v in vertices:
            seen[v] = False
        # Set start node to seen
        seen[s] = True
        to_explore = [s]
        
        distances = {s: 0}

        # Traverse graph starting from s
        while not to_explore == []:
            v = to_explore[0]
            del to_explore[0]
            dist = distances[v]
            if v in g:
                for w in g[v]:
                    if not seen[w]:
                        to_explore.append(w)
                        distances[w] = dist + 1
                        seen[w] = True
        return distances[list(distances.keys())[-1]]

    vertices = set([key for key in graph.keys()])

    return max([bfs(graph, v, vertices) for v in vertices])
            


def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")


if __name__ == '__main__':
    main()
