import os

def process_multiple_tsp_files_to_latex(input_folder, output_file):
    latex_rows = []

    for input_file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, input_file)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()

            instance = ""
            branch_and_bound = {"Execution Time": "NA", "Memory Usage": "NA", "Relative Error": "NA"}
            twice_around = {"Execution Time": "NA", "Memory Usage": "NA", "Relative Error": "NA"}
            christofides = {"Execution Time": "NA", "Memory Usage": "NA", "Relative Error": "NA"}

            current_algorithm = None

            for line in lines:
                line = line.strip()

                if line.startswith("Instance:"):
                    instance = line.split(":")[1].strip()
                elif "Branch and Bound Algorithm" in line:
                    current_algorithm = branch_and_bound
                elif "Twice Around the Tree Algorithm" in line:
                    current_algorithm = twice_around
                elif "Christofides Algorithm" in line:
                    current_algorithm = christofides
                elif line.startswith("Execution Time:"):
                    current_algorithm["Execution Time"] = line.split(":")[1].strip().replace("seconds", "").strip()
                elif line.startswith("Memory Usage:"):
                    current_algorithm["Memory Usage"] = line.split(":")[1].strip().replace("bytes", "").strip()
                elif line.startswith("Relative Error:"):
                    current_algorithm["Relative Error"] = line.split(":")[1].strip()

            # Formatar o resultado em LaTeX
            latex_row = (
                f"\\textbf{{{instance}}} & {branch_and_bound['Execution Time']} & {branch_and_bound['Memory Usage']} & {branch_and_bound['Relative Error']} & "
                f"{twice_around['Execution Time']} & {twice_around['Memory Usage']} & {twice_around['Relative Error']} & "
                f"{christofides['Execution Time']} & {christofides['Memory Usage']} & {christofides['Relative Error']} \\\\ \\hline"
            )

            latex_rows.append((instance, latex_row))

    # Ordenar as linhas por dimensão extraída de "instance"
    def extract_dimension(instance):
        return int(''.join(filter(str.isdigit, instance)))

    latex_rows.sort(key=lambda x: extract_dimension(x[0]))

    # Escrever as linhas em um arquivo de saída
    with open(output_file, 'w') as out_file:
        out_file.write("\n".join(row[1] for row in latex_rows))

# Exemplo de uso
input_folder = "./tsp_results"
output_file = "output_latex_rows.tex"
process_multiple_tsp_files_to_latex(input_folder, output_file)
print(f"Arquivo de saída gerado: {output_file}")
