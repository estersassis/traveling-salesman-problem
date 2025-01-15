import asyncio
import heapq
import time
import sys
from math import inf


class BranchAndBoundAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.minimum_path = None
        self.minimum_cost = None
        self.execution_time = None
        self.memory_usage = None
        self.relative_error = None
        self.timeout = False

    async def calculate_bound(self, path):
        bound = 0
        for node in self.graph.nodes:
            if node in path:
                continue  # Vértices já visitados
            # sum(duas menores)/2
            edges = sorted([self.graph[node][neighbor]['weight'] for neighbor in self.graph.neighbors(node)])
            if len(edges) >= 2:
                bound += edges[0] + edges[1]
            elif len(edges) == 1:
                bound += edges[0]
            await asyncio.sleep(0)  # Liberar controle após cada nó
        return bound / 2

    async def execute(self):
        n = len(self.graph.nodes)
        if n == 0:
            return (0, [])
        
        start_time = time.process_time()
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
                    new_bound = new_cost + await self.calculate_bound(new_path)
                    if new_bound < best_cost:
                        heapq.heappush(pq, (new_bound, new_cost, new_path, neighbor))
                await asyncio.sleep(0)
            await asyncio.sleep(0)
        
        end_time = time.process_time()
        self.execution_time = end_time - start_time
        self.memory_usage = (
            sys.getsizeof(pq) + 
            sum(sys.getsizeof(item) for item in pq) + 
            sys.getsizeof(best_path)
        )
        self.minimum_cost = best_cost
        self.minimum_path = best_path
        return self
