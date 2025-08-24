import os
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

# Disjoint Set for Kruskal's
class DisjointSet:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}
    
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 == root2:
            return False
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
        return True

# Kruskal's algorithm for MST
def kruskals_algorithm(graph_dict):
    if not graph_dict:
        print("Graph is empty!")
        return {}, 0
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/Kruskals_trace.txt', 'w')
    trace.write("Kruskal's Algorithm Trace\n")
    trace.write("=" * 50 + "\n\n")
    
    mst_dict = {}
    total_weight = 0
    
    edges = []
    for node1 in graph_dict:
        for node2 in graph_dict[node1]:
            if node1 < node2:
                edges.append((graph_dict[node1][node2], node1, node2))
    
    edges.sort()
    trace.write("Initial sorted edges (first 20 shown):\n")
    trace.write(f"{edges[:20]}\n\n")
    
    ds = DisjointSet(graph_dict.keys())
    
    for weight, node1, node2 in edges:
        trace.write(f"Checking edge ({node1}, {node2}), weight {weight}\n")
        if ds.union(node1, node2):
            if node1 not in mst_dict:
                mst_dict[node1] = {}
            if node2 not in mst_dict:
                mst_dict[node2] = {}
            mst_dict[node1][node2] = weight
            mst_dict[node2][node1] = weight
            total_weight += weight
            trace.write(f"Added edge ({node1}, {node2}) to MST\n")
            trace.write(f"Current MST weight: {total_weight}\n\n")
        else:
            trace.write(f"Edge ({node1}, {node2}) forms a cycle, skipped\n\n")
    
    mst_nodes = set(mst_dict.keys()).union(*[set(mst_dict[node].keys()) for node in mst_dict])
    if len(mst_nodes) < len(graph_dict):
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
        plt.title(f"Kruskal's MST (Total Weight: {total_weight})")
        plt.savefig("plots/kruskals_mst_visualization.png")
        plt.close()
    
    return mst_dict, total_weight

# Runs Kruskal's algorithm
def run_kruskals_algorithm(graph_file_path='datasets/roadNet-TX.txt', sample_size=1000, save_output=True):
    full_graph = load_graph_data(graph_file_path, is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return {}, 0, 0.0
    
    if len(full_graph) > sample_size:
        print(f"Sampling to {sample_size} nodes")
        graph = create_sampled_graph(full_graph, sample_size)
    else:
        graph = full_graph
    
    if not graph:
        print("Sampled graph is empty")
        return {}, 0, 0.0
    
    print("\nRunning Kruskal's:")
    display_graph_stats(graph)
    
    (mst, total_weight), exec_time = measure_time(kruskals_algorithm, graph)
    
    print(f"Kruskal's time: {exec_time:.6f} seconds")
    print(f"MST total weight: {total_weight}")
    save_execution_time("Kruskal's", exec_time)
    
    if save_output:
        os.makedirs('results', exist_ok=True)
        with open('results/Kruskals_results.txt', 'w') as f:
            f.write("Kruskal's Results\n")
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

# Analyzes Kruskal's performance
def analyze_kruskals_performance():
    full_graph = load_graph_data('datasets/roadNet-TX.txt', is_directed=False)
    
    if not full_graph:
        print("Failed to load graph")
        return
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"Kruskal's": []}
    
    for size in sizes:
        print(f"\nTesting Kruskal's with {size} nodes")
        sampled = create_sampled_graph(full_graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["Kruskal's"].append(0.0)
            continue
        (_, _), exec_time = measure_time(kruskals_algorithm, sampled)
        print(f"Kruskal's time: {exec_time:.6f} seconds")
        times["Kruskal's"].append(exec_time)
    
    # Plot for nodes
    plot_line_graph(sizes, times, ["Kruskal's"], "Number of Nodes", "Time (seconds)", 
                    "Kruskal's: Time vs Nodes", "kruskals_nodes.png")
    # Plot for vertices
    plot_line_graph(sizes, times, ["Kruskal's"], "Vertices", "Time (seconds)", 
                    "Kruskal's: Time vs Vertices", "kruskals_vertices.png")

if __name__ == "__main__":
    run_kruskals_algorithm(sample_size=1000)
    analyze_kruskals_performance()