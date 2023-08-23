import os
from typing import Dict, Set
from exercises.tick10 import load_graph
from collections import deque


def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    total_v = len(graph)
    queue = deque([])
    stack = []
    betweenness = {v: 0 for v in graph.keys()}
    for i, s in enumerate(graph.keys()):
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
                dependency[v] += n_shortest_paths[v] / n_shortest_paths[w] * (dependency[w] + 1)
            if w != s:
                betweenness[w] += dependency[w] / 2 
        
        print(f'Completed: {i/total_v*100}%')
        
    return betweenness




def main():
    # graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))
    graph = {1:{2}, 2:{3}, 3:{4}, 4:{}}
    
    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")


if __name__ == '__main__':
    main()
