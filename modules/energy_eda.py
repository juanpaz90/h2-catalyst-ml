# EDA
# STEP 4: Energy & Target Analysis
# --------

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from torch_geometric.data import Data

def calculate_energy_statistics(dataset: List[Data]):
    """
    Extracts energy values and calculates deltas for the dataset.
    """
    y_init = []
    y_relaxed = []
    energy_deltas = []

    print(f"Extracting energy data from {len(dataset)} systems...")

    for data in dataset:
        # Compatibility check for energy targets
        # IS2RE targets are typically stored as y_init and y_relaxed
        yi = getattr(data, 'y_init', None)
        yr = getattr(data, 'y_relaxed', None)

        # Fallback for legacy storage
        if yi is None: yi = getattr(data, '_store', {}).get('y_init', None)
        if yr is None: yr = getattr(data, '_store', {}).get('y_relaxed', None)

        if yi is not None and yr is not None:
            # Energies are usually scalars or 1-element tensors
            val_init = yi.item() if torch.is_tensor(yi) else yi
            val_relaxed = yr.item() if torch.is_tensor(yr) else yr
            
            y_init.append(val_init)
            y_relaxed.append(val_relaxed)
            energy_deltas.append(val_relaxed - val_init)

    return np.array(y_init), np.array(y_relaxed), np.array(energy_deltas)

def plot_energy_distributions(y_init, y_relaxed, deltas):
    """
    Generates modular plots for target energy analysis.
    """
    # 1. Target Energy Distribution (Relaxed)
    plt.figure(figsize=(10, 5))
    plt.hist(y_relaxed, bins=50, color='seagreen', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(y_relaxed), color='red', linestyle='--', label=f'Mean: {np.mean(y_relaxed):.2f} eV')
    plt.title("Distribution of Target Relaxed Energies (y_relaxed)")
    plt.xlabel("Energy (electron volts eV)")
    plt.ylabel("Frequency (System Count)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 2. Energy Delta Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(deltas, bins=50, color='crimson', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(deltas), color='blue', linestyle='--', label=f'Mean Delta: {np.mean(deltas):.2f} eV')
    plt.title("Energy Change During Relaxation (y_relaxed - y_init)")
    plt.xlabel("Energy Change (eV)")
    plt.ylabel("Frequency (System Count)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 3. Correlation: Initial vs Relaxed
    plt.figure(figsize=(8, 8))
    plt.scatter(y_init, y_relaxed, alpha=0.4, s=10, color='darkblue')
    # Plot ideal x=y line
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='y_init = y_relaxed')
    plt.title("Correlation: Initial vs Relaxed Energy")
    plt.xlabel("Initial Energy (eV)")
    plt.ylabel("Relaxed Energy (eV)")
    plt.legend()
    plt.axis('equal')
    plt.show()

def perform_energy_eda(dataset: List[Data]):
    """
    Main entry point for Phase 4 EDA.
    """
    y_init, y_relaxed, deltas = calculate_energy_statistics(dataset)

    if len(y_relaxed) == 0:
        print("Error: No energy attributes found. Are you using a test split? (Test splits don't have targets).")
        return

    plot_energy_distributions(y_init, y_relaxed, deltas)

    print(f"\n--- Energy EDA Summary ---")
    print(f"Total valid samples: {len(y_relaxed)}")
    print(f"Relaxed Energy Range: [{np.min(y_relaxed):.2f}, {np.max(y_relaxed):.2f}] eV")
    print(f"Avg Relaxation 'Drop': {np.mean(deltas):.4f} eV")
    print(f"Std Dev of Targets: {np.std(y_relaxed):.4f} eV")
    
    print("\n--- Model Training Insights ---")
    print("1. Target Scale: If your energy values are large (e.g., hundreds of eV),")
    print("   consider using 'Mean/Std Normalization' or 'Subtracting the mean' for the loss.")
    print("2. Delta Analysis: The mean delta shows how much 'information' is gained")
    print("   by the relaxation. If the delta is very small, the initial structure")
    print("   is already very close to the minimum.")
