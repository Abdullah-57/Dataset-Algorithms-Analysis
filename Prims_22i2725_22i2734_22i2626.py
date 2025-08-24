import heapq
import os
import sys
import random
from time import time
import matplotlib.pyplot as plt
import networkx as nx

# Loads graph from file, assigns random weights
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
                # Random weight between 1 and 10
                weight = random.randint(1, 10)
                graph_dict[node1][node2] = weight
                if not is_directed:
                    graph_dict[node2][node1] = weight
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

# Prim's algorithm for MST
def prims_algorithm(graph_dict, start_node=None):
    if not graph_dict:
        print("Graph is empty!")
        return {}, 0
    
    if start_node is None or start_node not in graph_dict:
        start_node = random.choice(list(graph_dict.keys()))
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/Prims_trace.txt', 'w')
    trace.write(f"Prim's Algorithm Trace (Start node: {start_node})\n")
    trace.write("=" * 50 + "\n\n")
    
    mst_dict = {}
    total_weight = 0
    visited = {start_node}
    
    pq = []
    for neighbor in graph_dict[start_node]:
        heapq.heappush(pq, (graph_dict[start_node][neighbor], start_node, neighbor))
    
    trace.write(f"Starting with node {start_node}\n")
    trace.write(f"Initial queue: {pq}\n\n")
    
    while pq and len(visited) < len(graph_dict):
        weight, node1, node2 = heapq.heappop(pq)
        trace.write(f"Checking edge ({node1}, {node2}), weight {weight}\n")
        
        if node2 in visited:
            trace.write(f"Node {node2} already in MST, skipping\n\n")
            continue
        
        if node1 not in mst_dict:
            mst_dict[node1] = {}
        if node2 not in mst_dict:
            mst_dict[node2] = {}
        mst_dict[node1][node2] = weight
        mst_dict[node2][node1] = weight
        total_weight += weight
        visited.add(node2)
        
        trace.write(f"Added edge ({node1}, {node2}) to MST\n")
        trace.write(f"Current MST weight: {total_weight}\n")
        
        for neighbor in graph_dict[node2]:
            if neighbor not in visited:
                heapq.heappush(pq, (graph_dict[node2][neighbor], node2, neighbor))
                trace.write(f"Added edge ({node2}, {neighbor}) to queue\n")
        
        trace.write(f"Current queue: {pq}\n\n")
    
    if len(visited) < len(graph_dict):
        trace.write("Warning: Graph not connected, MST is partial\n")
    
    trace.close()
    
    # Visualize MST
    if mst_dict:
        G = nx.Graph()
        for node1 in mst_dict:
            for node2 in mst_dict[node1]:
                if node1 < node2:
                    G.add_edge(node1, node2, weight=mst_dict[node1][node2])
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=50, node_color='lightgreen', with_labels=False)
        plt.title(f"Prim's MST (Total Weight: {total_weight}, Start Node: {start_node})")
        plt.savefig("plots/prims_mst_visualization.png")
        plt.close()
    
    return mst_dict, total_weight

# Runs Prim's algorithm
def run_prims_algorithm(graph_file_path='datasets/roadNet-TX.txt', source=None, sample_size=1000, save_output=True):
    full_graph = load_graph_data(graph_file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return {}, 0, 0.0
    
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
        return {}, 0, 0.0
    
    print("\nRunning Prim's:")
    display_graph_stats(graph)
    
    (mst, total_weight), exec_time = measure_time(prims_algorithm, graph, source)
    
    print(f"Prim's time: {exec_time:.6f} seconds")
    print(f"MST total weight: {total_weight}")
    save_execution_time("Prim's", exec_time)
    
    if save_output:
        os.makedirs('results', exist_ok=True)
        with open('results/Prims_results.txt', 'w') as f:
            f.write(f"Prim's Results (Start node: {source})\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"MST total weight: {total_weight}\n")
            f.write("MST edges:\n")
            edge_count = 0
            for node1 in mst:
                for node2 in mst[node1]:
                    if node1 < node2:
                        edge_count += 1
                        f.write(f"Edge ({node1}, {node2}), Weight: {mst[node1][node2]}\n")
            f.write(f"\nTotal edges in MST: {edge_count}\n")
    
    return mst, total_weight, exec_time

# Analyzes Prim's performance
def analyze_prims_performance():
    full_graph = load_graph_data('datasets/roadNet-TX.txt', is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"Prim's": []}
    
    for size in sizes:
        print(f"\nTesting Prim's with {size} nodes")
        sampled = create_sampled_graph(full_graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["Prim's"].append(0.0)
            continue
        (_, _), exec_time = measure_time(prims_algorithm, sampled, None)
        print(f"Prim's time: {exec_time:.6f} seconds")
        times["Prim's"].append(exec_time)
    
    # Plot for nodes
    plot_line_graph(sizes, times, ["Prim's"], "Number of Nodes", "Time (seconds)", 
                    "Prim's: Time vs Nodes", "prims_nodes.png")
    # Plot for vertices
    plot_line_graph(sizes, times, ["Prim's"], "Vertices", "Time (seconds)", 
                    "Prim's: Time vs Vertices", "prims_vertices.png")

if __name__ == "__main__":
    run_prims_algorithm(sample_size=1000)
    analyze_prims_performance()