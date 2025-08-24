import os
import random
from time import time
import matplotlib.pyplot as plt

# Loads the graph, assigns random weights (including negative)
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
                # Random weight between -5 and 10
                weight = random.randint(-5, 10)
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

# Samples a smaller connected graph, starting from a specified node if provided
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

# Bellman-Ford algorithm
def bellman_ford_algorithm(graph, start, is_directed=False):
    if not graph:
        print("Graph is empty!")
        return {}, {}, False
    
    if start not in graph:
        print(f"Node {start} not in graph, picking random")
        start = random.choice(list(graph.keys()))
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/BellmanFord_trace.txt', 'w')
    trace.write(f"Bellman-Ford from node {start}\n" + "="*50 + "\n\n")
    
    distances = {node: float('inf') for node in graph}
    predecessors = {node: None for node in graph}
    distances[start] = 0
    
    # Create edge list, avoid duplicates for undirected
    edges = []
    for node1 in graph:
        for node2 in graph[node1]:
            if not is_directed and node1 < node2:  # Process each edge once
                edges.append((node1, node2, graph[node1][node2]))
            elif is_directed:
                edges.append((node1, node2, graph[node1][node2]))
    
    trace.write(f"Initial distances: {distances}\n\n")
    
    for i in range(len(graph) - 1):
        trace.write(f"Iteration {i+1}:\n")
        changed = False
        for u, v, w in edges:
            trace.write(f"Edge ({u}, {v}), weight={w}\n")
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                trace.write(f"Relaxing: {v} from {distances[v]} to {distances[u] + w}\n")
                distances[v] = distances[u] + w
                predecessors[v] = u
                changed = True
            trace.write("\n")
        if not changed:
            trace.write(f"No changes, stopping early\n\n")
            break
    
    # Check for negative cycles
    trace.write("Checking negative cycles:\n")
    has_cycle = False
    for u, v, w in edges:
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            trace.write(f"Negative cycle at ({u}, {v})\n")
            has_cycle = True
            break
    trace.write("No negative cycles\n" if not has_cycle else "")
    
    trace.close()
    return distances, predecessors, has_cycle

# Reconstructs shortest path
def get_shortest_path(start, end, predecessors):
    path = []
    current = end
    if predecessors.get(end) is None and start != end:
        return []
    while current is not None:
        path.append(current)
        current = predecessors.get(current)
    path.reverse()
    return path

# Runs Bellman-Ford
def run_bellman_ford_algorithm(file_path='datasets/roadNet-TX.txt', source=None, sample_size=1000):
    graph = load_graph_data(file_path, is_directed=False)
    
    if not graph:
        print("Failed to load graph")
        return {}, 0.0
    
    # Prompt for source node
    if source is None:
        try:
            source = int(input("Enter source node (integer): "))
        except ValueError:
            print("Invalid input, choosing random node")
            source = random.choice(list(graph.keys()))
    
    # Validate source node against original graph
    if source not in graph:
        print(f"Node {source} not in original graph, choosing random node")
        source = random.choice(list(graph.keys()))
    
    # Sample graph, ensuring source node is included
    if len(graph) > sample_size:
        print(f"Sampling to {sample_size} nodes")
        graph = create_sampled_graph(graph, sample_size, start_node=source)
    
    if not graph:
        print("Sampled graph is empty")
        return {}, 0.0
    
    print("\nRunning Bellman-Ford:")
    display_graph_stats(graph)
    
    (distances, predecessors, has_cycle), exec_time = measure_time(bellman_ford_algorithm, graph, source)
    
    print(f"Bellman-Ford time: {exec_time:.6f} seconds")
    save_execution_time("Bellman-Ford", exec_time)
    
    os.makedirs('results', exist_ok=True)
    with open('results/BellmanFord_results.txt', 'w') as f:
        f.write(f"Bellman-Ford from node {source}\n" + "="*50 + "\n\n")
        if has_cycle:
            f.write("Warning: Negative cycle detected\n\n")
        for node, dist in sorted(distances.items(), key=lambda x: x[1]):
            if dist == float('inf'):
                continue
            path = get_shortest_path(source, node, predecessors)
            f.write(f"Node: {node}, Dist: {dist}, Path: {' -> '.join(map(str, path))}\n")
    
    return distances, exec_time

# Analyzes Bellman-Ford performance
def analyze_bellman_ford_performance():
    graph = load_graph_data('datasets/roadNet-TX.txt', is_directed=False)
    if not graph:
        print("Failed to load graph")
        return {}, []
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"Bellman-Ford": []}
    
    for size in sizes:
        print(f"\nTesting Bellman-Ford with {size} nodes")
        sampled = create_sampled_graph(graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["Bellman-Ford"].append(0.0)
            continue
        source = random.choice(list(sampled.keys()))
        (_, _, _), exec_time = measure_time(bellman_ford_algorithm, sampled, source)
        print(f"Bellman-Ford time: {exec_time:.6f} seconds")
        times["Bellman-Ford"].append(exec_time)
    
    # Original plot (nodes)
    plot_line_graph(sizes, times, ["Bellman-Ford"], "Number of Nodes", "Time (seconds)", 
                    "Bellman-Ford: Time vs Nodes", "bellman_ford_nodes.png")
    # New plot (vertices)
    plot_line_graph(sizes, times, ["Bellman-Ford"], "Vertices", "Time (seconds)", 
                    "Bellman-Ford: Time vs Vertices", "bellman_ford_vertices.png")
    
    return times, sizes

if __name__ == "__main__":
    run_bellman_ford_algorithm(sample_size=1000)
    analyze_bellman_ford_performance()