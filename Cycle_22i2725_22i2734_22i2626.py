import os
import sys
import random
from time import time
import matplotlib.pyplot as plt
import networkx as nx

# Loads graph from file (unweighted for cycle detection)
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

# Writes trace to file
def write_trace_to_file(trace_file, trace):
    os.makedirs(os.path.dirname(trace_file), exist_ok=True)
    with open(trace_file, 'w') as f:
        f.write("Cycle Detection (DFS) Trace\n")
        f.write("=" * 50 + "\n\n")
        for msg in trace:
            f.write(f"{msg}\n")

# Prints graph statistics
def display_graph_stats(graph_dict):
    nodes = len(graph_dict)
    edges = sum(len(graph_dict[node]) for node in graph_dict) // 2
    avg_degree = edges / nodes if nodes > 0 else 0
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Average degree: {avg_degree:.2f}")
    if edges == nodes - 1:
        print("Warning: Sampled graph is a tree (no cycles possible). Try a larger sample size.")

# Samples a smaller connected graph, preserving cycles
def create_sampled_graph(graph, target_nodes):
    if target_nodes < 1 or not graph:
        print("Invalid target size or empty graph")
        return {}
    
    sampled_graph = {}
    max_tries = 5
    for _ in range(max_tries):
        # Start with a random node
        start = random.choice(list(graph.keys()))
        sampled_nodes = {start}
        sampled_graph[start] = {}
        
        # Keep track of nodes that can connect to the subgraph
        candidates = set()
        for neighbor in graph[start]:
            candidates.add(neighbor)
        
        # Add nodes and their edges until target is reached
        while len(sampled_nodes) < target_nodes and candidates:
            # Pick a random candidate node
            new_node = random.choice(list(candidates))
            sampled_nodes.add(new_node)
            sampled_graph[new_node] = {}
            
            # Add all edges from new_node to existing sampled nodes
            for neighbor in graph[new_node]:
                if neighbor in sampled_nodes:
                    sampled_graph[new_node][neighbor] = 1
                    sampled_graph[neighbor][new_node] = 1
                else:
                    candidates.add(neighbor)
            
            # Update candidates
            candidates.remove(new_node)
            for neighbor in graph[new_node]:
                if neighbor not in sampled_nodes:
                    candidates.add(neighbor)
        
        if len(sampled_nodes) >= target_nodes:
            break
        sampled_graph = {}
        candidates = set()
    
    if len(sampled_nodes) < target_nodes:
        print(f"Warning: Sampled only {len(sampled_nodes)} nodes")
    
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

# Cycle detection using DFS (adapted from provided approach)
def detect_cycle(graph_dict, trace_file=None):
    """
    Detect cycles in a graph using DFS.
    Returns:
        has_cycle: Boolean indicating if a cycle exists
        cycle_path: List of vertices forming a cycle (empty if no cycle)
        trace: List of trace messages
    """
    visited = set()
    rec_stack = set()
    parent = {}
    trace = ["Cycle Detection using DFS"]
    
    def dfs_cycle(vertex, parent_vertex=None):
        visited.add(vertex)
        rec_stack.add(vertex)
        trace.append(f"Visit: {vertex}, RecStack: {list(rec_stack)}")
        
        for neighbor in sorted(graph_dict.get(vertex, {})):
            trace.append(f"  Checking neighbor {neighbor}")
            if neighbor == parent_vertex:  # Skip parent in undirected graph
                continue
            if neighbor not in visited:
                parent[neighbor] = vertex
                trace.append(f"  Set parent[{neighbor}] = {vertex}")
                result = dfs_cycle(neighbor, vertex)
                if result:
                    return result
            elif neighbor in rec_stack:
                parent[neighbor] = vertex
                trace.append(f"  Cycle detected, set parent[{neighbor}] = {vertex}")
                return neighbor
        
        rec_stack.remove(vertex)
        return None
    
    cycle_start = None
    for vertex in sorted(graph_dict.keys()):
        if vertex not in visited:
            trace.append(f"Starting DFS from unvisited vertex: {vertex}")
            cycle_start = dfs_cycle(vertex)
            if cycle_start:
                break
    
    if not cycle_start:
        trace.append("No cycle detected in the graph")
        if trace_file:
            write_trace_to_file(trace_file, trace)
        return False, [], trace
    
    cycle_path = []
    current = parent.get(cycle_start, None)
    if current is None:
        trace.append(f"Error: Cannot reconstruct cycle; no parent for vertex {cycle_start}")
        if trace_file:
            write_trace_to_file(trace_file, trace)
        return True, [], trace
    
    while current != cycle_start:
        if current not in parent:
            trace.append(f"Error: Cannot reconstruct cycle; no parent for vertex {current}")
            if trace_file:
                write_trace_to_file(trace_file, trace)
            return True, cycle_path, trace
        cycle_path.append(current)
        current = parent[current]
    
    cycle_path.append(cycle_start)
    cycle_path.reverse()
    cycle_path.append(cycle_path[0])  # Close the cycle
    trace.append(f"Cycle found: {' -> '.join(map(str, cycle_path))}")
    
    if trace_file:
        write_trace_to_file(trace_file, trace)
    
    return True, cycle_path, trace

# Runs cycle detection
def run_cycle_detection(file_path='datasets/roadNet-TX.txt', sample_size=2000):
    full_graph = load_graph_data(file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return False, [], 0.0
    
    if len(full_graph) > sample_size:
        print(f"Sampling to {sample_size} nodes")
        graph = create_sampled_graph(full_graph, sample_size)
    else:
        graph = full_graph
    
    if not graph:
        print("Sampled graph is empty")
        return False, [], 0.0
    
    print("\nRunning Cycle Detection:")
    display_graph_stats(graph)
    
    (has_cycle, cycle_path, _), exec_time = measure_time(detect_cycle, graph, 'traces/CycleDetection_trace.txt')
    
    print(f"Cycle Detection time: {exec_time:.6f} seconds")
    if has_cycle:
        print(f"Cycle found: {' -> '.join(map(str, cycle_path))}")
    else:
        print("No cycle found")
    
    save_execution_time("Cycle Detection", exec_time)
    
    result_content = "Cycle Detection Results\n"
    result_content += "=" * 50 + "\n\n"
    result_content += f"Nodes: {len(graph)}\n"
    result_content += f"Edges: {sum(len(graph[node]) for node in graph) // 2}\n"
    if has_cycle:
        result_content += f"Cycle found: {' -> '.join(map(str, cycle_path))}\n"
    else:
        result_content += "No cycle found\n"
    save_results('results/CycleDetection_results.txt', result_content)
    
    # Visualize result
    G = nx.Graph()
    if cycle_path:
        for i in range(len(cycle_path)-1):
            G.add_edge(cycle_path[i], cycle_path[i+1])
        plt.title(f"Cycle Detected: {' -> '.join(map(str, cycle_path))}")
    else:
        for node1 in graph:
            for node2 in graph[node1]:
                if node1 < node2:
                    G.add_edge(node1, node2)
        plt.title("No Cycle Detected")
    
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=50, node_color='lightcoral', with_labels=has_cycle)
    plt.savefig("plots/cycle_detection_visualization.png")
    plt.close()
    
    return has_cycle, cycle_path, exec_time

# Analyzes cycle detection performance
def analyze_cycle_detection_performance(file_path='datasets/roadNet-TX.txt'):
    full_graph = load_graph_data(file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"Cycle Detection": []}
    
    for size in sizes:
        print(f"\nTesting Cycle Detection with {size} nodes")
        sampled = create_sampled_graph(full_graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["Cycle Detection"].append(0.0)
            continue
        (_, _, _), exec_time = measure_time(detect_cycle, sampled)
        print(f"Cycle Detection time: {exec_time:.6f} seconds")
        times["Cycle Detection"].append(exec_time)
    
    plot_line_graph(sizes, times, ["Cycle Detection"], "Number of Nodes", "Time (seconds)", 
                    "Cycle Detection: Time vs Nodes", "cycle_detection_nodes.png")
    plot_line_graph(sizes, times, ["Cycle Detection"], "Vertices", "Time (seconds)", 
                    "Cycle Detection: Time vs Vertices", "cycle_detection_vertices.png")

if __name__ == "__main__":
    dataset_path = 'datasets/roadNet-TX.txt'
    sample_size = 2000
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            sample_size = int(sys.argv[2])
        except ValueError:
            print("Invalid sample size")
            sys.exit(1)
    
    run_cycle_detection(dataset_path, sample_size)
    analyze_cycle_detection_performance(dataset_path)