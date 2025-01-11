import heapq
import networkx as nx
from math import inf, ceil


class TravelingSalesmanProblem:
    def __init__(self, graph: list):
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(graph)
    
    def bound(self, path):
        total_bound = 0

        for node in self.graph.nodes:
            edges = [
                self.graph[node][neighbor]['weight']
                for neighbor in self.graph.neighbors(node)
            ]
            if node in path:
                if len(edges) > 0:
                    total_bound += min(edges)
            else:
                if len(edges) > 1:
                    total_bound += sum(sorted(edges)[:2])
                elif len(edges) == 1:
                    total_bound += edges[0]

        total_bound = ceil(total_bound / 2)
        return total_bound

    def branch_and_bound_algorithm(self):
        root = (self.bound([]), 0 , 0, [0]) 
        best = inf
        sol = None
        n = self.graph.number_of_nodes()
        # node[0]: bound, node[1]: level, node[2]: custo acumulado, node[3]: caminho atual
        queue = []
        heapq.heappush(queue, root)
        while queue:
            node = heapq.heappop(queue)
            if node[1] > n:
                if best > node[2]: 
                    best = node[2]
                    sol = node[3]
            elif node[0] < best:
                if node[1] < n:
                    for k in range(1, n-1):
                        if (
                            k not in node[3] 
                            and self.graph.get_edge_data(node[3][-1], k)
                            and self.bound(node[3] + [k]) < best
                        ):
                            heapq.heappush(queue,(
                                self.bound(node[3]+[k]),
                                node[1] + 1,
                                node[2] + self.graph.get_edge_data(node[3][-1], k)['weight'],
                                node[3] + [k]
                            ))
                elif (
                    self.graph.get_edge_data(node[3][-1], 0)
                    and self.bound(node[3] + [0]) < best
                    and self.graph.edges in node[3]
                ):
                    heapq.heappush(queue,(
                        self.bound(node[3]+[0]),
                        node[1] + 1,
                        node[2] + self.graph.get_edge_data(node[3][-1], 0)['weight'],
                        node[3] + [0]
                    ))

        return node[3]
    
    def twice_arount_the_tree_algorithm(self):
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

    def christofides_algorithm(self):
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
