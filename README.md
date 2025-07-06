# Traveling Salesman Problem

Este trabalho investiga o desempenho de algoritmos para o Problema do Caixeiro Viajante em sua versão euclidiana (Problema do Caixeiro Viajante Euclidiano) para a matéria de Algoritmos II do curso de Ciência da Computação da Universidade Federal de Minas Gerais (UFMG). Os algoritmos usados são: Branch and Bound, Twice Around the Tree (2-aproximativo) e Christofides (1.5-aproximativo).

Observação: Esse projeto já contém instâncias retiradas do site http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ e instâncias extras criadas pelas autoras para maior abordagem de testes. Os resultados do processamento das mesmas também já estão presentes para fins de análise. 

---

## **Instruções de Execução**

### **Execução com Valores Padrão**
Para executar o programa utilizando os valores padrão para as pastas e arquivos:
```bash
$ python3 main.py
```

---

### **Execução Personalizada**
Para controlar o caminho das pastas e arquivos, você pode usar as seguintes tags:
```bash
$ python3 main.py -p tsp -o processed_folder -r results_folder -opt optimal_file.txt
```

**Explicação das Tags:**
- `-p` ou `--path`: Define o caminho da pasta que contém as instâncias TSP.
- `-o` ou `--output`: Define a pasta onde os arquivos processados serão armazenados.
- `-r` ou `--results`: Define a pasta onde os resultados serão salvos.
- `-opt` ou `--optimal`: Define o arquivo que contém os valores ótimos.
---

### **Tag para Retornar os Arquivos à Pasta de Origem**
Se desejar garantir que os arquivos voltem para a pasta de origem após o processamento, utilize a seguinte tag:
```bash
$ python3 main.py --move-back
```

---

### **Ajuda e Informações Adicionais**
Para mais informações sobre as tags e opções disponíveis, utilize:
```bash
$ python3 main.py -h
```

---

## **Requisitos**
- Python 3.x instalado no sistema.
- Dependências listadas no arquivo `requirements.txt`.

### **Instalação das Dependências**
```bash
$ pip install -r requirements.txt
```

## **Autores**

- Ester Sara Assis Silva - estersarasilva@gmail.com
- Julia Paes de Viterbo -juliapaesv@gmail.com
