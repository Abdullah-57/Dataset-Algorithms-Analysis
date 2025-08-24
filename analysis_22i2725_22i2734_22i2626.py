import matplotlib.pyplot as plt
from Dijkstra_22i2725_22i2734_22i2626 import analyze_dijkstra_performance
from BellmanFord_22i2725_22i2734_22i2626 import analyze_bellman_ford_performance
import os

# Compares Dijkstra and Bellman-Ford execution times with a bar chart
def plot_comparison_bar():
    # Get performance data
    dijkstra_times, sizes = analyze_dijkstra_performance()
    bellman_times, _ = analyze_bellman_ford_performance()
    
    # Prepare data for bar chart
    bar_width = 0.35
    x = range(len(sizes))
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - bar_width/2 for i in x], dijkstra_times["Dijkstra"], bar_width, label="Dijkstra", color='blue')
    plt.bar([i + bar_width/2 for i in x], bellman_times["Bellman-Ford"], bar_width, label="Bellman-Ford", color='orange')
    
    plt.xlabel("Number of Vertices")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Dijkstra vs Bellman-Ford: Execution Time Comparison")
    plt.xticks(x, sizes)
    plt.legend()
    plt.grid(True, axis='y')
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig("plots/dijkstra_vs_bellman_ford_bar.png")
    plt.close()

if __name__ == "__main__":
    plot_comparison_bar()