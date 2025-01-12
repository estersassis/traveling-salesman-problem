import heapq
import networkx as nx
from math import inf, ceil


class TravelingSalesmanProblem:
    def __init__(self, graph: list):
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(graph)
    
    def branch_and_bound_algorithm(self):
        n = len(self.graph.nodes)
        if n == 0:
            return (0, [])
        
        def calculate_bound(path):
            bound = 0
            for node in self.graph.nodes:
                if node in path:
                    continue  # vértices já visitados
                # sum(duas menores)/2
                edges = sorted([self.graph[node][neighbor]['weight'] for neighbor in self.graph.neighbors(node)])
                if len(edges) >= 2:
                    bound += edges[0] + edges[1]
                elif len(edges) == 1:
                    bound += edges[0]
            return bound / 2 

        pq = []
        start_node = 0
        heapq.heappush(pq, (0, 0, [start_node], 0))  # (bound, current_cost, path, last_node)

        best_cost = inf
        best_path = []

        while pq:
            curr_bound, curr_cost, curr_path, last_node = heapq.heappop(pq)

            # poda
            if curr_bound >= best_cost:
                continue

            if len(curr_path) == n:
                final_cost = curr_cost + self.graph[curr_path[-1]][curr_path[0]]['weight']
                if final_cost < best_cost:
                    best_cost = final_cost
                    best_path = curr_path + [curr_path[0]]
                continue

            for neighbor in self.graph.neighbors(last_node):
                if neighbor not in curr_path:
                    new_path = curr_path + [neighbor]
                    edge_cost = self.graph[last_node][neighbor]['weight']
                    new_cost = curr_cost + edge_cost
                    new_bound = new_cost + calculate_bound(new_path)
                    if new_bound < best_cost:
                        heapq.heappush(pq, (new_bound, new_cost, new_path, neighbor))

        return best_cost, best_path
    
    def twice_arount_the_tree_algorithm(self): # 2-aproximativo
        root = next(iter(self.graph.nodes))
        mst = nx.minimum_spanning_tree(self.graph)
        preorder = list(nx.dfs_preorder_nodes(mst, source=root))
        hamiltonian_cycle = preorder + [preorder[0]]
        
        return hamiltonian_cycle

    def simplify_eulerian(self, circuit):
        simplified_circuit = []
        visited = set()

        for u, v in circuit:
            if v in visited:
                while simplified_circuit and simplified_circuit[-1][0] != v:
                    visited.remove(simplified_circuit.pop()[1])
            simplified_circuit.append((u, v))
            visited.add(v)
        
        return simplified_circuit

    def christofides_algorithm(self): # 1.5-aproximativo
        # Computar a árvore geradora mínima (MST)
        mst = nx.minimum_spanning_tree(self.graph)
        
        # Encontrar os vértices de grau ímpar
        odd_degree_vertices = [node for node, degree in mst.degree() if degree % 2 == 1]

        # Subgrafo induzido pelos vértices de grau ímpar
        induced_subgraph = self.graph.subgraph(odd_degree_vertices)

        # Matching perfeito de peso mínimo
        matching = nx.algorithms.matching.min_weight_matching(induced_subgraph)

        # Formar o G' combinando a MST e o matching
        combined_graph = nx.Graph()
        for u, v in mst.edges():
            weight = self.graph[u][v]['weight']
            combined_graph.add_edge(u, v, weight=weight)
        for u, v in matching:
            weight = self.graph[u][v]['weight']
            combined_graph.add_edge(u, v, weight=weight)

        # Computar o circuito euleriano
        if not nx.is_eulerian(combined_graph):
            raise ValueError("G' não é euleriano.")
        eulerian_circuit = list(nx.eulerian_circuit(combined_graph))

        # Eliminar vértices duplicados
        simplified_circuit = self.simplify_eulerian(eulerian_circuit)
        path = [u for u, v in simplified_circuit]
        path.append(simplified_circuit[-1][1])
        return path

problem = TravelingSalesmanProblem([
    (0, 1, 3),  # Peso 3 entre os vértices 0 (a) e 1 (b)
    (0, 2, 1),  # Peso 1 entre os vértices 0 (a) e 2 (c)
    (0, 3, 5),  # Peso 5 entre os vértices 0 (a) e 3 (d)
    (0, 4, 8),  # Peso 8 entre os vértices 0 (a) e 4 (e)
    (1, 2, 6),  # Peso 6 entre os vértices 1 (b) e 2 (c)
    (1, 3, 7),  # Peso 7 entre os vértices 1 (b) e 3 (d)
    (1, 4, 9),  # Peso 9 entre os vértices 1 (b) e 4 (e)
    (2, 3, 4),  # Peso 4 entre os vértices 2 (c) e 3 (d)
    (2, 4, 2),  # Peso 2 entre os vértices 2 (c) e 4 (e)
    (3, 4, 3)   # Peso 3 entre os vértices 3 (d) e 4 (e)
])
print(problem.twice_arount_the_tree_algorithm())
print(problem.christofides_algorithm())
print(problem.branch_and_bound_algorithm())