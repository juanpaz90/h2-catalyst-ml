# EDA
# STEP 3: Geometric & Graph Topology
# --------

import torch
import numpy as np
import matplotlib.pyplot as plt

def perform_geometric_eda(pt_tensor):
    """
    Geometric & Graph Topology EDA.
    Analyzes bond lengths (edge distances) and node connectivity.
    
    Parameters:
    pt_tensor: List of PyTorch Geometric Data objects.
    """
    all_edge_distances = []
    avg_degrees = []
    max_distances_per_graph = []

    print(f"Calculating Geometric Properties for {len(pt_tensor)} systems...")

    for data in pt_tensor:
        # Extract positions and edge index (handles standard and legacy PyG)
        if hasattr(data, 'pos'):
            pos = data.pos
            edge_index = data.edge_index
        else:
            pos = getattr(data, '_store', {}).get('pos', None)
            edge_index = getattr(data, '_store', {}).get('edge_index', None)

        if pos is None or edge_index is None:
            continue

        # 1. Edge Distance Calculation
        # edge_index is [2, num_edges] -> src and dst nodes
        src, dst = edge_index[0], edge_index[1]
        
        # Calculate Euclidean distance between connected atoms
        # Formula: ||pos[src] - pos[dst]||
        diff = pos[src] - pos[dst]
        distances = torch.norm(diff, dim=-1).cpu().numpy()
        
        if len(distances) > 0:
            all_edge_distances.extend(distances.tolist())
            max_distances_per_graph.append(np.max(distances))

        # 2. Node Connectivity (Degree)
        num_atoms = pos.size(0)
        # Using scatter_add to count occurrences of each node index in edge_index
        degrees = torch.zeros(num_atoms)
        degrees.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        avg_degrees.append(degrees.mean().item())

    # --- Plot 1: Bond Length Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_edge_distances, bins=60, color='mediumpurple', edgecolor='black', alpha=0.7)
    mean_dist = np.mean(all_edge_distances)
    plt.axvline(mean_dist, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dist:.2f}Å')
    plt.title("Distribution of Bond Lengths (Existing Edges)")
    plt.xlabel("Distance (Angstroms Å)")
    plt.ylabel("Frequency (Edge Count)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # --- Plot 2: Average Connectivity (Graph Density) ---
    plt.figure(figsize=(10, 6))
    plt.hist(avg_degrees, bins=25, color='orange', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(avg_degrees), color='blue', linestyle='dashed', linewidth=2, label=f'Avg: {np.mean(avg_degrees):.1f}')
    plt.title("System-wide Connectivity (Average Node Degree)")
    plt.xlabel("Average Neighbors per Atom")
    plt.ylabel("Number of Systems (Frequency)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # --- Summary Statistics ---
    print(f"\n--- Geometric Summary ---")
    print(f"Total Edges Analyzed: {len(all_edge_distances)}")
    print(f"Global Mean Bond Length: {mean_dist:.2f} Å")
    print(f"Shortest Detected Bond: {np.min(all_edge_distances):.2f} Å")
    print(f"Longest Detected Bond: {np.max(all_edge_distances):.2f} Å")
    print(f"Global Average Neighbors: {np.mean(avg_degrees):.2f} per atom")
    
    print("\n---> Insights for Model Configuration ---")
    print(f"1. Cutoff Suggestion: Your longest edge is {np.max(all_edge_distances):.2f}Å.")
    print(f"2. Graph Density: On average, each atom is connected to {np.mean(avg_degrees):.1f} others.")
    print("   High connectivity increases the number of message-passing operations.")