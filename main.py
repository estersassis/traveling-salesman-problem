import os
import shutil
import asyncio
import heapq
import math
import time
import sys
import argparse
import networkx as nx
from math import inf, ceil

class TravelingSalesmanProblem:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.optimal_values_path = "optimal.txt"
        self.instance = None
        self.timeout = 10.0

        self.graph = self.build_graph()

        self.twice_around_the_tree_algorithm = None
        self.christofides_algorithm = None
        self.branch_and_bound_algorithm = None
    
    def build_graph(self):
        print("Building graph...")
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        node_coords = []
        reading_coords = False
        
        for line in lines:
            line = line.strip()
            if "NAME" in line:
                self.instance = line.split(": ")[1]
                continue
            if "EDGE_WEIGHT_TYPE" in line:
                if "EUC_2D" not in line:
                    raise Exception("Not a 2D Euclidian instance")
            if "DIMENSION" in line:
                dim = line.split(": ")[1]
                if int(dim) > 3000:
                    raise Exception("Not enough memory to compute this graph")
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            if line == "EOF":
                break

            if reading_coords:
                parts = line.split()
                if len(parts) == 3:
                    node_coords.append((float(parts[1]), float(parts[2])))

        graph = nx.Graph()
        n = len(node_coords)

        for i in range(n):
            for k in range(i + 1, n):
                x1, y1 = node_coords[i]
                x2, y2 = node_coords[k]
                weight = round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                graph.add_edge(i, k, weight=weight)
        return graph
    
    async def run_algorithms_concurrently(self):
        async def execute_with_timeout(algorithm_instance, name):
            try:
                print(f"Starting {name}...")
                start_time = time.perf_counter()
                result = await asyncio.wait_for(algorithm_instance.execute(), timeout=self.timeout)
                print(f"{name} completed in {time.perf_counter() - start_time:.2f} seconds")
                return result
            except asyncio.TimeoutError:
                print(f"{name} timeout in {time.perf_counter() - start_time:.2f} seconds")
                algorithm_instance.timeout = True
                return algorithm_instance
            
        self.twice_around_the_tree_algorithm = self.TwiceAroundTheTreeAlgorithm(self.graph)
        self.christofides_algorithm = self.ChristofidesAlgorithm(self.graph)
        self.branch_and_bound_algorithm = self.BranchAndBoundAlgorithm(self.graph)

        print("Running algorithm...")
        await asyncio.gather(
            execute_with_timeout(self.twice_around_the_tree_algorithm, "TwiceAroundTheTreeAlgorithm"),
            execute_with_timeout(self.christofides_algorithm, "ChristofidesAlgorithm"),
            execute_with_timeout(self.branch_and_bound_algorithm, "BranchAndBoundAlgorithm")
        )
    
    def calculate_relative_error(self):
        optimal_values = {}
        with open(self.optimal_values_path, 'r') as file:
            for line in file:
                instance_name, optimal_value = line.strip().split(" : ")
                optimal_values[instance_name] = float(optimal_value)

        if self.instance not in optimal_values:
            return

        optimal_value = optimal_values[self.instance]

        for algorithm in [
            self.branch_and_bound_algorithm,
            self.twice_around_the_tree_algorithm,
            self.christofides_algorithm,
        ]:
            if algorithm.timeout == False:
                algorithm.relative_error = (algorithm.minimum_cost - optimal_value) / optimal_value if optimal_value > 0 else 0

    def generate_result_file(self, file_name, results_folder):
        self.calculate_relative_error()

        with open(f"{results_folder}/{file_name}.result", 'w') as file:
            file.write(f"Instance: {self.instance}\n")
            
            for algorithm_name, algorithm_obj in {
                "Branch and Bound Algorithm": self.branch_and_bound_algorithm,
                "Twice Around the Tree Algorithm": self.twice_around_the_tree_algorithm,
                "Christofides Algorithm": self.christofides_algorithm,
            }.items():
                if algorithm_obj is not None:
                    file.write(f"\n--- {algorithm_name} ---\n")
                    if algorithm_obj.timeout:
                        file.write(f"Timeout\n")
                    else:
                        file.write(f"Minimum Path: {algorithm_obj.minimum_path}\n")
                        file.write(f"Minimum Cost: {algorithm_obj.minimum_cost}\n")
                        file.write(f"Execution Time: {algorithm_obj.execution_time:.4f} seconds\n")
                        file.write(f"Memory Usage: {algorithm_obj.memory_usage} bytes\n")
                        file.write(f"Relative Error: {algorithm_obj.relative_error:.4f}\n")

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
            self.memory_usage = sys.getsizeof(pq) + sum(sys.getsizeof(item) for item in pq)
            self.minimum_cost = best_cost
            self.minimum_path = best_path
            return self

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
            self.memory_usage = sys.getsizeof(mst)
            self.minimum_cost = total_weight
            self.minimum_path = hamiltonian_cycle

            return self
    
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

class TravelingSalesmanProblemRunner:
    def __init__(self, folder_to_process="tsp", processed_folder="processed", results_folder="results"):
        self.folder_to_process = folder_to_process
        self.processed_folder = processed_folder
        self.results_folder = results_folder

        os.makedirs(self.folder_to_process, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)

    async def run_for_all_files(self):
        files = [f for f in os.listdir(self.folder_to_process) if f.endswith(".tsp")]

        for file_name in files:
            file_path = os.path.join(self.folder_to_process, file_name)
            print(f"Processing {file_path}...")

            try:
                problem = TravelingSalesmanProblem(file_path)
                await problem.run_algorithms_concurrently()
                problem.generate_result_file(file_name, self.results_folder)
                shutil.move(file_path, os.path.join(self.processed_folder, file_name))
                print(f"Finished processing {file_name}.")
            except Exception as e:
                shutil.move(file_path, os.path.join(self.processed_folder, file_name))
                print(f"Error processing {file_name}: {e}. Skipping to next file.")
                continue
        
    def move_back_processed_files(self):
        files = os.listdir(self.processed_folder)
        for file_name in files:
            src_path = os.path.join(self.processed_folder, file_name)
            dest_path = os.path.join(self.folder_to_process, file_name)
            shutil.move(src_path, dest_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Process Traveling Salesman Problem files.")
    parser.add_argument(
        "-p", "--process-folder", type=str, default="tsp_instances",
        help="Folder containing files to process (default: tsp_instances)"
    )
    parser.add_argument(
        "-o", "--processed-folder", type=str, default="tsp_processed_instances",
        help="Folder to store processed files (default: tsp_processed_instances)"
    )
    parser.add_argument(
        "-r", "--results-folder", type=str, default="tsp_results",
        help="Folder to store results (default: tsp_results)"
    )
    parser.add_argument(
        "--move-back", action="store_true",
        help="Move all files from processed folder back to process folder after processing"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    runner = TravelingSalesmanProblemRunner(
        folder_to_process=args.process_folder,
        processed_folder=args.processed_folder,
        results_folder=args.results_folder
    )

    asyncio.run(runner.run_for_all_files())

    if args.move_back:
        runner.move_back_processed_files()