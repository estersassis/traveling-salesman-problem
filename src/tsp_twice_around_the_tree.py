import asyncio
import time
import sys
import networkx as nx


class TwiceAroundTheTreeAlgorithm:
    def __init__(self, graph):
        self.graph = graph
        self.minimum_path = None
        self.minimum_cost = None
        self.execution_time = None
        self.memory_usage = None
        self.relative_error = None
        self.timeout = False

    async def execute(self): # 2-aproximativo
        start_time = time.process_time()
        root = next(iter(self.graph.nodes))
        mst = await asyncio.to_thread(nx.minimum_spanning_tree, self.graph)
        preorder = list(nx.dfs_preorder_nodes(mst, source=root))
        hamiltonian_cycle = preorder + [preorder[0]]

        total_weight = 0
        for i in range(len(hamiltonian_cycle) - 1):
            u, v = hamiltonian_cycle[i], hamiltonian_cycle[i + 1]
            total_weight += self.graph[u][v]['weight']

            await asyncio.sleep(0)
        
        end_time = time.process_time()
        self.execution_time = end_time - start_time
        self.memory_usage = (
            sys.getsizeof(mst) +
            sys.getsizeof(preorder) +
            sys.getsizeof(hamiltonian_cycle) 
        )
        self.minimum_cost = total_weight
        self.minimum_path = hamiltonian_cycle

        return self
