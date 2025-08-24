import os
import random
from time import time
import matplotlib.pyplot as plt
import numpy as np

# Loads graph from file (unweighted for degree calculation)
def load_graph_data(file_path, is_directed=False):
    graph_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                node1, node2 = map(int, line.strip().split())
                if node1 not in graph_dict:
                    graph_dict[node1] = {}
                if node2 not in graph_dict:
                    graph_dict[node2] = {}
                graph_dict[node1][node2] = 1
                if not is_directed:
                    graph_dict[node2][node1] = 1
    except FileNotFoundError:
        print(f"Oops! File {file_path} not found")
        return {}
    return graph_dict

# Measures execution time
def measure_time(func, *args):
    start = time()
    result = func(*args)
    end = time()
    return result, end - start

# Saves execution time to file
def save_execution_time(algo_name, time_taken):
    os.makedirs('results', exist_ok=True)
    with open('results/execution_times.txt', 'a') as f:
        f.write(f"{algo_name}: {time_taken:.6f} seconds\n")

# Prints graph statistics
def display_graph_stats(graph_dict):
    nodes = len(graph_dict)
    edges = sum(len(graph_dict[node]) for node in graph_dict) // 2
    avg_degree = edges / nodes if nodes > 0 else 0
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Average degree: {avg_degree:.2f}")

# Samples a smaller connected graph using BFS
def create_sampled_graph(graph, target_nodes):
    if target_nodes < 1 or not graph:
        print("Invalid target size or empty graph")
        return {}
    
    sampled_graph = {}
    max_tries = 5
    for _ in range(max_tries):
        start = random.choice(list(graph.keys()))
        visited = {start}
        queue = [start]
        sampled_graph[start] = {}
        
        while queue and len(visited) < target_nodes:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    if neighbor not in sampled_graph:
                        sampled_graph[neighbor] = {}
                    weight = graph[current][neighbor]
                    sampled_graph[current][neighbor] = weight
                    sampled_graph[neighbor][current] = weight
                    if len(visited) < target_nodes:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        if len(visited) >= target_nodes:
            break
        sampled_graph = {}
    
    if len(visited) < target_nodes:
        print(f"Warning: Sampled only {len(visited)} nodes")
    
    return sampled_graph

# Plots execution times as a line graph
def plot_line_graph(sizes, times, labels, x_label, y_label, title, filename):
    plt.figure(figsize=(10, 6))
    for label in labels:
        plt.plot(sizes, times[label], marker='o', label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()

# Plots degree distribution as a histogram
def plot_degree_histogram(degree_counts, title, filename):
    plt.figure(figsize=(12, 8))
    degrees = sorted(degree_counts.keys())
    counts = [degree_counts[degree] for degree in degrees]
    plt.bar(degrees, counts, width=0.8, alpha=0.7)
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title(title)
    plt.grid(True, axis='y')
    if len(degrees) > 20:
        plt.xticks(np.arange(min(degrees), max(degrees)+1, step=max(1, (max(degrees)-min(degrees))//10)))
    plt.savefig(f"plots/{filename}")
    plt.close()

# Calculates average degree and related stats
def calculate_average_degree(graph_dict):
    if not graph_dict:
        print("Graph is empty!")
        return 0, {}, 0, 0, 0
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/AverageDegree_trace.txt', 'w')
    trace.write("Average Degree Calculation Trace\n")
    trace.write("=" * 50 + "\n\n")
    
    degrees = {}
    for node in graph_dict:
        degree = len(graph_dict[node])
        degrees[node] = degree
        trace.write(f"Node {node}: Degree = {degree}\n")
    
    trace.write("\nCalculating degree distribution:\n")
    degree_counts = {}
    for degree in degrees.values():
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
        trace.write(f"Degree {degree}: Count = {degree_counts[degree]}\n")
    
    degree_list = list(degrees.values())
    node_count = len(degree_list)
    
    total_degree = sum(degree_list)
    avg_degree = total_degree / node_count if node_count > 0 else 0
    
    min_degree = min(degree_list) if degree_list else 0
    max_degree = max(degree_list) if degree_list else 0
    
    sorted_degrees = sorted(degree_list)
    median_degree = (sorted_degrees[node_count//2 - 1] + sorted_degrees[node_count//2]) / 2 if node_count % 2 == 0 else sorted_degrees[node_count//2]
    
    trace.write("\nFinal Statistics:\n")
    trace.write(f"Total nodes: {node_count}\n")
    trace.write(f"Total degree sum: {total_degree}\n")
    trace.write(f"Average degree: {avg_degree:.2f}\n")
    trace.write(f"Minimum degree: {min_degree}\n")
    trace.write(f"Maximum degree: {max_degree}\n")
    trace.write(f"Median degree: {median_degree}\n")
    trace.write(f"Degree distribution: {degree_counts}\n")
    
    trace.close()
    return avg_degree, degree_counts, min_degree, max_degree, median_degree

# Runs average degree calculation
def run_average_degree_calculation(graph_file_path='datasets/roadNet-TX.txt', sample_size=1000, save_output=True):
    full_graph = load_graph_data(graph_file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return 0, {}, 0, 0, 0, 0.0
    
    if len(full_graph) > sample_size:
        print(f"Sampling to {sample_size} nodes")
        graph = create_sampled_graph(full_graph, sample_size)
    else:
        graph = full_graph
    
    if not graph:
        print("Sampled graph is empty")
        return 0, {}, 0, 0, 0, 0.0
    
    print("\nRunning Average Degree:")
    display_graph_stats(graph)
    
    (avg_degree, degree_counts, min_degree, max_degree, median_degree), exec_time = measure_time(calculate_average_degree, graph)
    
    print(f"Average Degree time: {exec_time:.6f} seconds")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Minimum degree: {min_degree}")
    print(f"Maximum degree: {max_degree}")
    print(f"Median degree: {median_degree}")
    
    save_execution_time("Average Degree", exec_time)
    
    if save_output:
        os.makedirs('results', exist_ok=True)
        with open('results/AverageDegree_results.txt', 'w') as f:
            f.write("Average Degree Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total nodes: {len(graph)}\n")
            f.write(f"Total edges: {sum(len(graph[node]) for node in graph) // 2}\n")
            f.write(f"Average degree: {avg_degree:.2f}\n")
            f.write(f"Minimum degree: {min_degree}\n")
            f.write(f"Maximum degree: {max_degree}\n")
            f.write(f"Median degree: {median_degree}\n")
            f.write(f"Degree distribution: {degree_counts}\n")
    
    plot_degree_histogram(degree_counts, "Degree Distribution", "degree_distribution.png")
    
    return avg_degree, degree_counts, min_degree, max_degree, median_degree, exec_time

# Analyzes average degree performance
def analyze_average_degree_performance():
    full_graph = load_graph_data('datasets/roadNet-TX.txt', is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"Average Degree": []}
    
    for size in sizes:
        print(f"\nTesting Average Degree with {size} nodes")
        sampled = create_sampled_graph(full_graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["Average Degree"].append(0.0)
            continue
        (_, _, _, _, _), exec_time = measure_time(calculate_average_degree, sampled)
        print(f"Average Degree time: {exec_time:.6f} seconds")
        times["Average Degree"].append(exec_time)
    
    # Plot for nodes
    plot_line_graph(sizes, times, ["Average Degree"], "Number of Nodes", "Time (seconds)", 
                    "Average Degree: Time vs Nodes", "average_degree_nodes.png")
    # Plot for vertices
    plot_line_graph(sizes, times, ["Average Degree"], "Vertices", "Time (seconds)", 
                    "Average Degree: Time vs Vertices", "average_degree_vertices.png")

if __name__ == "__main__":
    run_average_degree_calculation(sample_size=1000)
    analyze_average_degree_performance()