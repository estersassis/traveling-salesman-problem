import asyncio
import math
import time
import networkx as nx
from src.tsp_branch_and_bound import BranchAndBoundAlgorithm
from src.tsp_twice_around_the_tree import TwiceAroundTheTreeAlgorithm
from src.tsp_christofides import ChristofidesAlgorithm


class TravelingSalesmanProblem:
    def __init__(self, file_path: str, optimal_file):
        self.file_path = file_path
        self.optimal_file = optimal_file
        self.instance = None
        self.timeout = 300.0

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
            
        self.twice_around_the_tree_algorithm = TwiceAroundTheTreeAlgorithm(self.graph)
        self.christofides_algorithm = ChristofidesAlgorithm(self.graph)
        self.branch_and_bound_algorithm = BranchAndBoundAlgorithm(self.graph)

        print("Running algorithm...")
        await asyncio.gather(
            execute_with_timeout(self.twice_around_the_tree_algorithm, "TwiceAroundTheTreeAlgorithm"),
            execute_with_timeout(self.christofides_algorithm, "ChristofidesAlgorithm"),
            execute_with_timeout(self.branch_and_bound_algorithm, "BranchAndBoundAlgorithm")
        )
    
    def calculate_relative_error(self):
        optimal_values = {}
        try:
            with open(self.optimal_file, 'r') as file:
                for line in file:
                    instance_name, optimal_value = line.strip().split(" : ")
                    optimal_values[instance_name] = float(optimal_value)
        except FileNotFoundError:
            for algorithm in [
                self.branch_and_bound_algorithm,
                self.twice_around_the_tree_algorithm,
                self.christofides_algorithm,
            ]:
                if algorithm.timeout == False:
                    algorithm.relative_error = -1

        if self.instance not in optimal_values:
            for algorithm in [
                self.branch_and_bound_algorithm,
                self.twice_around_the_tree_algorithm,
                self.christofides_algorithm,
            ]:
                if algorithm.timeout == False:
                    algorithm.relative_error = -1
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
                        file.write(f"Minimum Path: NA\n")
                        file.write(f"Minimum Cost: NA\n")
                        file.write(f"Execution Time: NA\n")
                        file.write(f"Memory Usage: NA\n")
                        file.write(f"Relative Error: NA\n")
                    else:
                        file.write(f"Minimum Path: {algorithm_obj.minimum_path}\n")
                        file.write(f"Minimum Cost: {algorithm_obj.minimum_cost}\n")
                        file.write(f"Execution Time: {algorithm_obj.execution_time:.4f} seconds\n")
                        file.write(f"Memory Usage: {algorithm_obj.memory_usage} bytes\n")
                        file.write(f"Relative Error: {algorithm_obj.relative_error:.4f}\n")