import matplotlib.pyplot as plt
import os

# Data from the provided output
algorithms = [
    "Dijkstra", "Bellman-Ford", "Prim's", "Kruskal's", "BFS", "DFS", "Diameter",
    "Cycle Detection", "Average Degree"
]
times_1000 = [0.085222, 0.004979, 8.083588, 6.740354, 9.348617, 9.395629, 8.297032, 0.000217, 0.003828]
times_2000 = [0.204954, 0.020971, 20.451639, 19.051277, 26.385010, 21.242152, 23.149405, 0.000422, 0.006248]
times_3000 = [0.455496, 0.014502, 40.815759, 40.111710, 41.992999, 40.748333, 45.454053, 0.000610, 0.008280]
times_5000 = [1.095678, 0.033552, 99.185100, 98.097542, 120.610987, 102.454390, 109.048112, 0.001029, 0.014883]

# Bar chart for execution times at 5000 nodes
plt.figure(figsize=(12, 6))
plt.bar(algorithms, times_5000, color='blue')
plt.xlabel('Algorithm')
plt.ylabel('Time (seconds)')
plt.title('Execution Times of All Algorithms at 5000 Nodes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/execution_times_bar.png')
plt.close()

# Line graph for scaling comparison
sizes = [1000, 2000, 3000, 5000]
plt.figure(figsize=(10, 6))
for i, algo in enumerate(algorithms):
    plt.plot(sizes, [times_1000[i], times_2000[i], times_3000[i], times_5000[i]], marker='o', label=algo)
plt.xlabel('Number of Vertices')
plt.ylabel('Execution Time (seconds)')
plt.title('Algorithm Scaling Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/scaling_comparison.png')
plt.close()