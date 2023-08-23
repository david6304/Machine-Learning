import os
from typing import Set, Dict, List, Tuple
from exercises.tick10 import load_graph
from collections import deque


def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """

    return int(sum(len(neighbours) for neighbours in graph.values()) / 2)


def get_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    components = []
    seen = {}
    for vertice in graph.keys():
        seen[vertice] = False

    def visit(v, seen, g, temp):
        seen[v] = True
        temp.add(v)
        for w in g[v]:
            if not seen[w]:
                visit(w, seen, g, temp)
        return temp
    
    for v in graph.keys():
        if not seen[v]:
            temp = set([])
            components.append(visit(v, seen, graph, temp))
        
    return components

def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    queue = deque([])
    stack = []
    betweenness = {(v, w): 0 for v in graph.keys() for w in graph[v]}
    for s in graph.keys():
        # Initialisation
        distances = {}
        predecessors = {}
        n_shortest_paths = {}
        dependency = {}

        for w in graph.keys():
            predecessors[w] = []
            distances[w] = -1
            n_shortest_paths[w] = 0
            dependency[w] = 0

        distances[s] = 0
        n_shortest_paths[s] = 1
        queue.append(s)

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph[v]:
                # Path discovery 
                if distances[w] == -1:
                    distances[w] = distances[v] + 1
                    queue.append(w)

                # Path counting
                if distances[w] == distances[v] + 1:
                    n_shortest_paths[w] += n_shortest_paths[v]
                    predecessors[w].append(v)
            
        # Accumulation
        while not stack == []:
            w = stack.pop()
            for v in predecessors[w]:
                c = n_shortest_paths[v] / n_shortest_paths[w] * (dependency[w] + 1)
                betweenness[(v, w)] += c
                dependency[v] += c
        
    return betweenness


def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    components = get_components(graph)
    while len(components) < min_components and get_number_of_edges(graph) > 0:
        betweenness = get_edge_betweenness(graph)
        max_betweenness = max(betweenness.values())
        max_edges = []
        for edge in betweenness.keys():
            if betweenness[edge] == max_betweenness:
                max_edges.append(edge)
        for u, v in max_edges:
            graph[u].discard(v)
            graph[v].discard(u)
        components = get_components(graph)
    
    return components
        

def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))

    num_edges = get_number_of_edges(graph)
    print(f"Number of edges: {num_edges}")

    components = get_components(graph)
    print(f"Number of components: {len(components)}")

    edge_betweenness = get_edge_betweenness(graph)
    print(f"Edge betweenness: {edge_betweenness}")

    clusters = girvan_newman(graph, min_components=20)
    print(f"Girvan-Newman for 20 clusters: {clusters}")


if __name__ == '__main__':
    main()

# irs38