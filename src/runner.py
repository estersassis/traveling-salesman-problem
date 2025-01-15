import os
import shutil
import asyncio
from src.tsp import TravelingSalesmanProblem


class TravelingSalesmanProblemRunner:
    def __init__(self, folder_to_process="tsp", processed_folder="processed", results_folder="results", optimal_file="optimal.txt"):
        self.folder_to_process = folder_to_process
        self.processed_folder = processed_folder
        self.results_folder = results_folder
        self.optimal_file = optimal_file

        os.makedirs(self.folder_to_process, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)
    
    async def process_file(self, file_name, semaphore):
        async with semaphore:
            file_path = os.path.join(self.folder_to_process, file_name)
            print(f"Processing {file_path}...")

            try:
                problem = TravelingSalesmanProblem(file_path, self.optimal_file)
                await problem.run_algorithms_concurrently()
                problem.generate_result_file(file_name, self.results_folder)
                shutil.move(file_path, os.path.join(self.processed_folder, file_name))
                print(f"Finished processing {file_name}.")
            except Exception as e:
                shutil.move(file_path, os.path.join(self.processed_folder, file_name))
                print(f"Error processing {file_name}: {e}. Skipping to next file.")

    async def run_for_all_files(self):
        files = [f for f in os.listdir(self.folder_to_process) if f.endswith(".tsp")]

        semaphore = asyncio.Semaphore(10)
        tasks = [self.process_file(file_name, semaphore) for file_name in files]

        await asyncio.gather(*tasks)
        
    def move_back_processed_files(self):
        files = os.listdir(self.processed_folder)
        for file_name in files:
            src_path = os.path.join(self.processed_folder, file_name)
            dest_path = os.path.join(self.folder_to_process, file_name)
            shutil.move(src_path, dest_path)