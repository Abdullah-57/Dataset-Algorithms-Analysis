import os
import sys
import random
from collections import deque
from time import time
import matplotlib.pyplot as plt
import networkx as nx

# Loads graph from file (unweighted for traversal)
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

# Saves results to file
def save_results(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)

# Prints graph statistics
def display_graph_stats(graph_dict):
    nodes = len(graph_dict)
    edges = sum(len(graph_dict[node]) for node in graph_dict) // 2
    avg_degree = edges / nodes if nodes > 0 else 0
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Average degree: {avg_degree:.2f}")

# Validates source node
def validate_source_node(source, graph):
    if source not in graph:
        print(f"Source node {source} not in graph")
        return False
    return True

# Samples a smaller connected graph using BFS
def create_sampled_graph(graph, target_nodes, start_node=None):
    if target_nodes < 1 or not graph:
        print("Invalid target size or empty graph")
        return {}
    
    sampled_graph = {}
    max_tries = 5
    
    # Use provided start_node if valid, otherwise choose random
    if start_node is not None and start_node in graph:
        start = start_node
    else:
        start = random.choice(list(graph.keys()))
    
    for _ in range(max_tries):
        visited = {start}
        queue = [start]
        sampled_graph[start] = {}
        
        while queue and len(visited) < target_nodes:
            current = queue.pop(0)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    if neighbor not in sampled_graph:
                        sampled_graph[neighbor] = {}
                    sampled_graph[current][neighbor] = 1
                    sampled_graph[neighbor][current] = 1
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

# BFS algorithm
def bfs_algorithm(graph_dict, source_node):
    if not graph_dict:
        print("Graph is empty!")
        return []
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/BFS_trace.txt', 'w')
    trace.write(f"BFS Traversal Trace (Source: {source_node})\n")
    trace.write("=" * 50 + "\n\n")
    
    visited = set()
    queue = deque([source_node])
    traversal = []
    visited.add(source_node)
    traversal.append(source_node)
    
    trace.write(f"Start with queue: [{source_node}]\n")
    
    while queue:
        current = queue.popleft()
        trace.write(f"Dequeued: {current}\n")
        
        for neighbor in sorted(graph_dict.get(current, {})):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                traversal.append(neighbor)
                trace.write(f"Enqueued: {neighbor}, Added to traversal\n")
        
        trace.write(f"Current queue: {list(queue)}\n\n")
    
    trace.close()
    
    # Visualize BFS tree
    if traversal:
        G = nx.Graph()
        parent = {source_node: None}
        for node in traversal:
            for neighbor in graph_dict.get(node, {}):
                if neighbor in visited and neighbor not in parent:
                    parent[neighbor] = node
                    G.add_edge(node, neighbor)
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=50, node_color='lightblue', with_labels=False)
        plt.title(f"BFS Tree from Source {source_node}")
        plt.savefig("plots/bfs_traversal_visualization.png")
        plt.close()
    
    return traversal

# Runs BFS algorithm
def run_bfs_algorithm(file_path='datasets/roadNet-TX.txt', source=None, sample_size=1000):
    full_graph = load_graph_data(file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return []
    
    # Prompt for source node
    if source is None:
        try:
            source = int(input("Enter source node (integer): "))
        except ValueError:
            print("Invalid input, choosing random node")
            source = random.choice(list(full_graph.keys()))
    
    # Validate source node against original graph
    if source not in full_graph:
        print(f"Node {source} not in original graph, choosing random node")
        source = random.choice(list(full_graph.keys()))
    
    # Sample graph, ensuring source node is included
    if len(full_graph) > sample_size:
        print(f"Sampling to {sample_size} nodes")
        graph = create_sampled_graph(full_graph, sample_size, start_node=source)
    else:
        graph = full_graph
    
    if not graph:
        print("Sampled graph is empty")
        return []
    
    if not validate_source_node(source, graph):
        return []
    
    print("\nRunning BFS:")
    display_graph_stats(graph)
    
    traversal, exec_time = measure_time(bfs_algorithm, graph, source)
    
    print(f"BFS time: {exec_time:.6f} seconds")
    print(f"Traversal order: {' -> '.join(map(str, traversal))}")
    
    result_content = f"BFS Traversal from Source {source}:\n"
    result_content += f"Traversal Order: {' -> '.join(map(str, traversal))}\n"
    save_results('results/BFS_results.txt', result_content)
    
    save_execution_time("BFS", exec_time)
    
    return traversal

# Analyzes BFS performance
def analyze_bfs_performance(file_path='datasets/roadNet-TX.txt'):
    full_graph = load_graph_data(file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"BFS": []}
    
    for size in sizes:
        print(f"\nTesting BFS with {size} nodes")
        sampled = create_sampled_graph(full_graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["BFS"].append(0.0)
            continue
        source = random.choice(list(sampled.keys()))
        _, exec_time = measure_time(bfs_algorithm, sampled, source)
        print(f"BFS time: {exec_time:.6f} seconds")
        times["BFS"].append(exec_time)
    
    plot_line_graph(sizes, times, ["BFS"], "Number of Nodes", "Time (seconds)", 
                    "BFS: Time vs Nodes", "bfs_nodes.png")
    plot_line_graph(sizes, times, ["BFS"], "Vertices", "Time (seconds)", 
                    "BFS: Time vs Vertices", "bfs_vertices.png")

if __name__ == "__main__":
    run_bfs_algorithm(sample_size=1000)
    analyze_bfs_performance()