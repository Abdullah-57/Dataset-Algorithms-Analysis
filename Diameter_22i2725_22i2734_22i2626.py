import os
import random
from time import time
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# Loads graph with random weights
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

# Saves execution time
def save_execution_time(algo_name, time_taken):
    os.makedirs('results', exist_ok=True)
    with open('results/execution_times.txt', 'a') as f:
        f.write(f"{algo_name}: {time_taken:.6f} seconds\n")

# Prints graph stats
def display_graph_stats(graph_dict):
    nodes = len(graph_dict)
    edges = sum(len(graph_dict[node]) for node in graph_dict) // 2
    avg_degree = edges / nodes if nodes > 0 else 0
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Average degree: {avg_degree:.2f}")

# Samples a smaller graph
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

# Plots execution times
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

# BFS for diameter calculation (unweighted for efficiency)
def bfs_diameter(graph, start, trace):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = deque([start])
    
    trace.write(f"BFS from node {start}\n")
    trace.write(f"Initial queue: {list(queue)}\n\n")
    
    farthest_node = start
    max_dist = 0
    
    while queue:
        node = queue.popleft()
        trace.write(f"Processing {node}, dist={distances[node]}\n")
        
        for neighbor in graph[node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[node] + 1  # Unweighted for BFS
                queue.append(neighbor)
                trace.write(f"Added {neighbor}, dist={distances[neighbor]}\n")
                if distances[neighbor] > max_dist:
                    max_dist = distances[neighbor]
                    farthest_node = neighbor
            trace.write(f"Queue: {list(queue)}\n\n")
    
    return max_dist, farthest_node

# Double BFS for approximate diameter
def find_graph_diameter(graph, sample_size=1000):
    if not graph:
        print("Graph is empty!")
        return 0, (None, None)
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/Diameter_trace.txt', 'w')
    trace.write("Diameter Calculation\n" + "="*50 + "\n\n")
    
    if len(graph) > sample_size:
        print(f"Sampling to {sample_size} nodes")
        sampled_graph = create_sampled_graph(graph, sample_size)
    else:
        sampled_graph = graph
    
    if not sampled_graph:
        trace.write("Sampled graph is empty\n")
        trace.close()
        return 0, (None, None)
    
    trace.write(f"Graph size: {len(sampled_graph)} nodes\n\n")
    
    # Double BFS: Start from random node, find farthest, then find farthest from that
    start = random.choice(list(sampled_graph.keys()))
    trace.write(f"First BFS from {start}\n")
    dist1, node1 = bfs_diameter(sampled_graph, start, trace)
    trace.write(f"Farthest node: {node1}, dist={dist1}\n\n")
    
    trace.write(f"Second BFS from {node1}\n")
    dist2, node2 = bfs_diameter(sampled_graph, node1, trace)
    trace.write(f"Farthest node: {node2}, dist={dist2}\n\n")
    
    diameter = dist2
    farthest_nodes = (node1, node2)
    
    trace.write(f"Diameter: {diameter}, nodes: {farthest_nodes}\n")
    
    # Visualize diameter path
    if diameter > 0 and farthest_nodes[0] is not None:
        G = nx.Graph()
        for n1 in sampled_graph:
            for n2 in sampled_graph[n1]:
                if n1 < n2:  # Avoid duplicate edges
                    G.add_edge(n1, n2)
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=50, node_color='lightblue', with_labels=False)
        nx.draw_networkx_nodes(G, pos, nodelist=[farthest_nodes[0], farthest_nodes[1]], node_color='red', node_size=100)
        plt.title(f"Diameter Path: Nodes {farthest_nodes[0]} to {farthest_nodes[1]} (Distance: {diameter})")
        plt.savefig("plots/diameter_visualization.png")
        plt.close()
    
    trace.close()
    return diameter, farthest_nodes

# Runs diameter calculation
def run_diameter_calculation(file_path='datasets/roadNet-TX.txt', sample_size=1000):
    graph = load_graph_data(file_path, is_directed=False)
    
    if not graph:
        print("Failed to load graph")
        return 0, (None, None), 0.0
    
    print("\nRunning Diameter:")
    display_graph_stats(graph)
    
    (diameter, farthest_nodes), exec_time = measure_time(find_graph_diameter, graph, sample_size)
    
    print(f"Diameter: {diameter}")
    print(f"Farthest nodes: {farthest_nodes}")
    print(f"Diameter time: {exec_time:.6f} seconds")
    
    save_execution_time("Diameter", exec_time)
    
    os.makedirs('results', exist_ok=True)
    with open('results/Diameter_results.txt', 'w') as f:
        f.write("Diameter Results\n" + "="*50 + "\n\n")
        f.write(f"Diameter: {diameter}\n")
        f.write(f"Farthest nodes: {farthest_nodes}\n")
    
    return diameter, farthest_nodes, exec_time

# Analyzes diameter performance
def analyze_diameter_performance():
    graph = load_graph_data('datasets/roadNet-TX.txt', is_directed=False)
    if not graph:
        print("Failed to load graph")
        return
    
    sizes = [1000, 2000,3000,5000]
    times = {"Diameter": []}
    
    for size in sizes:
        print(f"\nTesting Diameter with {size} nodes")
        (_, _), exec_time = measure_time(find_graph_diameter, graph, size)
        print(f"Diameter time: {exec_time:.6f} seconds")
        times["Diameter"].append(exec_time)
    
    # Original plot (nodes)
    plot_line_graph(sizes, times, ["Diameter"], "Number of Nodes", "Time (seconds)", 
                    "Diameter: Time vs Nodes", "diameter_nodes.png")
    # New plot (vertices)
    plot_line_graph(sizes, times, ["Diameter"], "Vertices", "Time (seconds)", 
                    "Diameter: Time vs Vertices", "diameter_vertices.png")

if __name__ == "__main__":
    run_diameter_calculation(sample_size=1000)
    analyze_diameter_performance()