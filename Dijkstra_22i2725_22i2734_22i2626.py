import heapq
import os
import random
from time import time
import matplotlib.pyplot as plt

# Loads the graph from a file, assigns random weights to edges
def load_graph_data(file_path, is_directed=False):
    graph_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):  # Skip comments
                    continue
                node1, node2 = map(int, line.strip().split())
                if node1 not in graph_dict:
                    graph_dict[node1] = {}
                if node2 not in graph_dict:
                    graph_dict[node2] = {}
                # Assign random weight between 1 and 10
                weight = random.randint(1, 10)
                graph_dict[node1][node2] = weight
                if not is_directed:
                    graph_dict[node2][node1] = weight
    except FileNotFoundError:
        print(f"Oops! File {file_path} not found")
        return {}
    return graph_dict

# Measures how long a function takes to run
def measure_time(func, *args):
    start = time()
    result = func(*args)
    end = time()
    return result, end - start

# Saves execution time to a file
def save_execution_time(algo_name, time_taken):
    os.makedirs('results', exist_ok=True)
    with open('results/execution_times.txt', 'a') as f:
        f.write(f"{algo_name}: {time_taken:.6f} seconds\n")

# Prints basic stats about the graph
def display_graph_stats(graph_dict):
    nodes = len(graph_dict)
    edges = sum(len(graph_dict[node]) for node in graph_dict) // 2
    avg_degree = edges / nodes if nodes > 0 else 0
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")
    print(f"Average degree: {avg_degree:.2f}")

# Samples a smaller connected graph using BFS, starting from a specified node if provided
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

# Dijkstra's algorithm to find shortest paths
def dijkstra_algorithm(graph, start):
    if not graph:
        print("Graph is empty!")
        return {}, {}
    
    if start not in graph:
        print(f"Node {start} not in graph, picking random")
        start = random.choice(list(graph.keys()))
    
    os.makedirs('traces', exist_ok=True)
    trace = open('traces/Dijkstra_trace.txt', 'w')
    trace.write(f"Dijkstra from node {start}\n" + "="*50 + "\n\n")
    
    distances = {node: float('inf') for node in graph}
    predecessors = {node: None for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    
    trace.write(f"Start: Queue={pq}, Distances={distances}\n\n")
    
    while pq:
        dist, node = heapq.heappop(pq)
        trace.write(f"Popped node {node}, dist={dist}\n")
        
        if dist > distances[node]:
            trace.write(f"Skipping {node}, better path exists\n\n")
            continue
        
        for neighbor in graph[node]:
            weight = graph[node][neighbor]
            new_dist = dist + weight
            trace.write(f"Checking {neighbor}, weight={weight}, new_dist={new_dist}\n")
            
            if new_dist < distances[neighbor]:
                trace.write(f"Updating {neighbor}: {distances[neighbor]} -> {new_dist}\n")
                distances[neighbor] = new_dist
                predecessors[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))
                trace.write(f"Pushed ({new_dist}, {neighbor})\n")
            else:
                trace.write(f"No update for {neighbor}\n")
            
            trace.write(f"Queue: {pq}\n\n")
    
    trace.close()
    return distances, predecessors

# Reconstructs the shortest path from start to end
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

# Runs Dijkstra and saves results
def run_dijkstra_algorithm(file_path='datasets/roadNet-TX.txt', source=None, sample_size=1000):
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
    
    print("\nRunning Dijkstra:")
    display_graph_stats(graph)
    
    (distances, predecessors), exec_time = measure_time(dijkstra_algorithm, graph, source)
    
    print(f"Dijkstra time: {exec_time:.6f} seconds")
    save_execution_time("Dijkstra", exec_time)
    
    os.makedirs('results', exist_ok=True)
    with open('results/Dijkstra_results.txt', 'w') as f:
        f.write(f"Dijkstra from node {source}\n" + "="*50 + "\n\n")
        for node, dist in sorted(distances.items(), key=lambda x: x[1]):
            if dist == float('inf'):
                continue
            path = get_shortest_path(source, node, predecessors)
            f.write(f"Node: {node}, Dist: {dist}, Path: {' -> '.join(map(str, path))}\n")
    
    return distances, exec_time

# Analyzes Dijkstra performance across different sizes
def analyze_dijkstra_performance():
    graph = load_graph_data('datasets/roadNet-TX.txt', is_directed=False)
    if not graph:
        print("Failed to load graph")
        return {}, []
    
    sizes = [1000, 2000, 3000, 5000]
    times = {"Dijkstra": []}
    
    for size in sizes:
        print(f"\nTesting Dijkstra with {size} nodes")
        sampled = create_sampled_graph(graph, size)
        if not sampled:
            print(f"Failed to sample {size} nodes")
            times["Dijkstra"].append(0.0)
            continue
        source = random.choice(list(sampled.keys()))
        (_, _), exec_time = measure_time(dijkstra_algorithm, sampled, source)
        print(f"Dijkstra time: {exec_time:.6f} seconds")
        times["Dijkstra"].append(exec_time)
    
    # Original plot (nodes)
    plot_line_graph(sizes, times, ["Dijkstra"], "Number of Nodes", "Time (seconds)", 
                    "Dijkstra: Time vs Nodes", "dijkstra_nodes.png")
    # New plot (vertices)
    plot_line_graph(sizes, times, ["Dijkstra"], "Vertices", "Time (seconds)", 
                    "Dijkstra: Time vs Vertices", "dijkstra_vertices.png")
    
    return times, sizes

if __name__ == "__main__":
    run_dijkstra_algorithm(sample_size=1000)
    analyze_dijkstra_performance()