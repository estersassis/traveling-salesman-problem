import argparse
import asyncio
from src.runner import TravelingSalesmanProblemRunner


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
        "-opt", "--optimal-file", type=str, default="optimal.txt",
        help="Path to the file containing optimal values (default: optimal.txt)"
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
        results_folder=args.results_folder,
        optimal_file=args.optimal_file
    )

    asyncio.run(runner.run_for_all_files())

    if args.move_back:
        runner.move_back_processed_files()