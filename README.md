# Graph Algorithms - Design and Analysis of Algorithms Project

This project implements and analyzes various **graph algorithms** with detailed performance evaluation on large-scale real-world datasets.

---

## 📚 Algorithms Implemented
- **Single Source Shortest Path**
  - Dijkstra’s Algorithm
  - Bellman-Ford Algorithm
- **Minimum Spanning Tree**
  - Prim’s Algorithm
  - Kruskal’s Algorithm
- **Graph Traversal**
  - Breadth-First Search (BFS)
  - Depth-First Search (DFS)
- **Other Graph Metrics**
  - Graph Diameter (Longest Shortest Path)
  - Cycle Detection
  - Average Degree Calculation

---

## ⚙️ Implementation Details
- **Language:** Python 3.9  
- **Libraries Used:** 
  - [NetworkX](https://networkx.org/) (graph representation & manipulation)  
  - [Matplotlib](https://matplotlib.org/) (visualization & plotting)  
- **Dataset:** [roadNet-TX](http://snap.stanford.edu/data/roadNet-TX.html) from Stanford SNAP  
  - ~1.37M nodes  
  - ~1.92M edges  
  - Undirected, sparse real-world road network  

Each algorithm saves:
- Results in `/results/`
- Execution traces in `/traces/`
- Visualizations in `/plots/`

Execution times were measured using Python’s `time` module for accuracy.

---

## 📊 Performance Analysis
Algorithms were tested on sampled subgraphs of **1000, 2000, 3000, and 5000 nodes**.  

- **Fastest:** Cycle Detection (~0.001s) & Average Degree (~0.015s)  
- **Moderate:** Dijkstra (~1.09s), Bellman-Ford (~0.034s)  
- **Heavy:** BFS, DFS, Diameter (>100s at 5000 nodes)  
- **MST Algorithms:** Prim’s (~99s) and Kruskal’s (~98s)  

Visual comparisons:
- **Bar Chart:** Execution times at 5000 nodes  
- **Line Graph:** Scaling performance across input sizes  

---

## 📁 Project Structure (WIP)

```bash
├── results/            # Algorithm results (paths, MSTs, cycles, etc.)
├── traces/             # Execution traces of operations
├── plots/              # Visualizations (MSTs, traversals, performance charts)
├── Dijkstra.py
├── BellmanFord.py
├── Prims.py
├── Kruskals.py
├── BFS.py
├── DFS.py
├── Diameter.py
├── CycleDetection.py
├── AverageDegree.py
└── README.md

```

## 🚀 How to Run
   ```bash
   
   git clone https://github.com/Abdullah-57/graph-algorithms-analysis.git
   cd graph-algorithms-analysis

```

## Install required libraries:
 ```bash

  bashpip install networkx matplotlib

```

## Run any algorithm:
 ```bash

bashpython Dijkstra.py
python BFS.py

```
---

## Check outputs in:

/results/ (algorithm outputs)

/traces/ (detailed steps)

/plots/ (visualizations)

---


## 👨‍💻 Contributors
- **Abdullah Daoud (22I-2626)**  
- **Usman Ali (22I-2725)**  
- **Faizan Rasheed (22I-2734)**

---

## ⚖️ License
This project is for **academic and personal skill development purposes only**.  
Reuse is allowed for **learning and research** with proper credit.

---
