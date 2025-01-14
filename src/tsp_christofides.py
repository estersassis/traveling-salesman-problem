import asyncio
import time
import sys
import networkx as nx


class ChristofidesAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.minimum_path = None
        self.minimum_cost = None
        self.execution_time = None
        self.memory_usage = None
        self.relative_error = None
        self.timeout = False

    async def execute(self): # 1.5-aproximativo
        start_time = time.process_time()

        mst = await asyncio.to_thread(nx.minimum_spanning_tree, self.graph)

        odd_degree_vertices = [node for node, degree in mst.degree() if degree % 2 == 1]
        induced_subgraph = self.graph.subgraph(odd_degree_vertices)
        
        matching = set()
        visited = set()

        for u in induced_subgraph.nodes:
            if u in visited:
                continue
            v, weight = await asyncio.to_thread(
                lambda: min(
                    ((neighbor, induced_subgraph[u][neighbor]['weight'])
                    for neighbor in induced_subgraph.neighbors(u) if neighbor not in visited),
                    key=lambda x: x[1]
                )
            )
            matching.add((u, v, weight))
            visited.update({u, v})
            await asyncio.sleep(0)

        eulerian_graph = nx.MultiGraph(mst)
        for u, v, weight in matching:
            eulerian_graph.add_edge(u, v, weight=weight)
            await asyncio.sleep(0)

        is_eulerian = await asyncio.to_thread(nx.is_eulerian, eulerian_graph)
        if not is_eulerian:
            raise nx.NetworkXError("Graph is not Eulerian.")
        
        eulerian_circuit = await asyncio.to_thread(nx.eulerian_circuit, eulerian_graph)
        
        visited = set()
        hamiltonian_path = []
        for u, v in eulerian_circuit:
            if u not in visited:
                visited.add(u)
                hamiltonian_path.append(u)
            await asyncio.sleep(0)
        hamiltonian_path.append(hamiltonian_path[0])
        self.minimum_path = hamiltonian_path

        self.minimum_cost = sum(
            self.graph[self.minimum_path[i]][self.minimum_path[i + 1]]['weight']
            for i in range(len(self.minimum_path) - 1)
        )

        end_time = time.process_time()
        self.execution_time = end_time - start_time
        self.memory_usage = sys.getsizeof(mst)

        return self