# EDA
# STEP 5: Visual Inspection (Sanity Check)
# --------

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ase.visualize.plot import plot_atoms
from modules.understand_data import ocp_to_ase, add_legend_to_ax

def visualize_and_compare_relaxations(dataset, num_samples: int):
    """
    Visualizes Adsorption Sites and Relaxation Trajectories.
    
    1. Plots the Initial Structure (Catalyst + Adsorbate)
    2. Overlays the Relaxed Adsorbate positions to show movement.
    
    Parameters:
    dataset: List of PyG Data objects (output of load_ocp_dataset)
    num_samples: Number of random samples to visualize
    """
    
    # Filter for samples that actually have relaxed positions (Train/Val sets)
    # Check relaxed positions existence safely
    valid_indices = []
    for i, d in enumerate(dataset):
        if hasattr(d, 'pos_relaxed') and d.pos_relaxed is not None:
             valid_indices.append(i)
        elif hasattr(d, '_store') and 'pos_relaxed' in d._store and d._store['pos_relaxed'] is not None:
             valid_indices.append(i)

    if not valid_indices:
        print("No samples with 'pos_relaxed' found. Cannot visualize trajectories.")
        return

    # Select random indices
    selected_indices = np.random.choice(valid_indices, min(num_samples, len(valid_indices)), replace=False)
    
    for idx in selected_indices:
        data = dataset[idx]
        
        # 1. Create ASE Atoms object for Initial State
        # Reuse ocp_to_ase from understand_data.py
        atoms_init = ocp_to_ase(data)
        
        if atoms_init is None:
            continue

        # 2. Create ASE Atoms object for Relaxed State
        # Since ocp_to_ase only reads 'pos', we manually update positions for the relaxed state
        atoms_relaxed = atoms_init.copy()
        
        # Extract relaxed positions safely
        pos_relaxed = getattr(data, 'pos_relaxed', None)
        if pos_relaxed is None and hasattr(data, '_store'):
             pos_relaxed = data._store.get('pos_relaxed', None)
             
        if pos_relaxed is not None:
            # Handle potential tensor/numpy difference based on understand_data implementation
            if torch.is_tensor(pos_relaxed):
                atoms_relaxed.set_positions(pos_relaxed.cpu().numpy())
            else:
                atoms_relaxed.set_positions(pos_relaxed)
        else:
             continue
            
        # 3. Identify Adsorbate Atoms (Tag == 2) for Analysis
        # 0=Subsurface, 1=Surface, 2=Adsorbate
        try:
            tags = atoms_init.get_tags()
            adsorbate_mask = (tags == 2)
        except:
             # Fallback if tags aren't set
             adsorbate_mask = np.zeros(len(atoms_init), dtype=bool)
        
        # Calculate average displacement of adsorbate
        if np.any(adsorbate_mask):
            init_ads_pos = atoms_init.positions[adsorbate_mask]
            relax_ads_pos = atoms_relaxed.positions[adsorbate_mask]
            
            # Euclidean distance between init and relaxed
            diff = relax_ads_pos - init_ads_pos
            dists = np.linalg.norm(diff, axis=1)
            avg_shift = np.mean(dists)
        else:
            avg_shift = 0.0

        # --- Visualization ---
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        # Title with info
        sid = getattr(data, 'sid', 'Unknown')
        if torch.is_tensor(sid): sid = sid.item()
        
        fig.suptitle(f"System ID: {sid} | Formula: {atoms_init.get_chemical_formula()} | Ads Shift: {avg_shift:.2f} Å", fontsize=16)

        # PLOT 1: Initial State (Top-down view)
        # Use existing logic or custom colors if tags exist
        if np.any(adsorbate_mask):
             colors_init = []
             for tag in tags:
                 if tag == 2: colors_init.append('red')       # Adsorbate
                 else: colors_init.append('#CCCCCC')          # Catalyst (Grey)
             plot_atoms(atoms_init, ax[0], radii=0.8, colors=colors_init, rotation=('0x,0y,0z'))
        else:
             plot_atoms(atoms_init, ax[0], radii=0.8, rotation=('0x,0y,0z'))

        ax[0].set_title("Initial State (Top View)\nRed=Adsorbate, Grey=Catalyst")
        ax[0].set_axis_off()
        
        # Add legend using reused function
        add_legend_to_ax(ax[0], atoms_init)

        # PLOT 2: Relaxation Trajectory (Side view)
        if np.any(adsorbate_mask):
            # 1. Catalyst (Base)
            catalyst_mask = (tags != 2)
            cat_atoms = atoms_init[catalyst_mask]
            plot_atoms(cat_atoms, ax[1], radii=0.8, colors=['#CCCCCC']*len(cat_atoms), rotation=('90x,0y,0z'))
            
            # 2. Initial Adsorbate (Red)
            ads_atoms_init = atoms_init[adsorbate_mask]
            plot_atoms(ads_atoms_init, ax[1], radii=0.8, colors=['red']*len(ads_atoms_init), rotation=('90x,0y,0z'))
            
            # 3. Relaxed Adsorbate (Blue)
            ads_atoms_relax = atoms_relaxed[adsorbate_mask]
            plot_atoms(ads_atoms_relax, ax[1], radii=0.8, colors=['blue']*len(ads_atoms_relax), rotation=('90x,0y,0z'))
            
            # Custom legend for trajectory
            legend_elements = [
                Patch(facecolor='red', edgecolor='black', label='Initial Pos'),
                Patch(facecolor='blue', edgecolor='black', label='Relaxed Pos'),
                Patch(facecolor='#CCCCCC', edgecolor='black', label='Catalyst')
            ]
            ax[1].legend(handles=legend_elements, loc='upper right')
        else:
            plot_atoms(atoms_init, ax[1], radii=0.8, rotation=('90x,0y,0z'))
            ax[1].text(0.5, 0.5, "No Tags Found", ha='center')

        ax[1].set_title("Relaxation Trajectory (Side View)")
        ax[1].set_axis_off()
        
        plt.tight_layout()
        plt.show()