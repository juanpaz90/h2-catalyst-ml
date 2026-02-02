# EDA
# STEP 6: Interactive 3D Visualization
# --------

import py3Dmol
import numpy as np
import torch
from ase.io import write
import io
from modules.understand_data import ocp_to_ase

def make_view(atoms, width=600, height=400):
    """
    Creates a py3Dmol view object from an ASE Atoms object.
    """
    # Convert ASE atoms to PDB format string in memory
    # PDB is a robust format for transferring structure data to py3Dmol
    xyz_buffer = io.StringIO()
    write(xyz_buffer, atoms, format='xyz')
    xyz_str = xyz_buffer.getvalue()

    # Initialize viewer
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, 'xyz')
    
    # Style settings:
    # 1. Stick representation for bonds
    view.setStyle({'stick': {'radius': 0.15}})
    
    # 2. Sphere representation for atoms (scaled by element)
    # We can color by element or custom properties
    view.addStyle({'sphere': {'scale': 0.25}})
    
    # 3. Zoom to fit structure
    view.zoomTo()
    
    return view

def visualize_interactive_relaxation(dataset, sample_idx=0):
    """
    Creates an interactive 3D visualization comparing Initial vs Relaxed states.
    
    Parameters:
    dataset: List of PyG Data objects
    sample_idx: Index of the sample to visualize
    """
    data = dataset[sample_idx]
    
    # 1. Get Initial Structure (using your existing function)
    atoms_init = ocp_to_ase(data)
    
    # 2. Get Relaxed Structure
    # Check for relaxed positions
    pos_relaxed = getattr(data, 'pos_relaxed', None)
    if pos_relaxed is None and hasattr(data, '_store'):
        pos_relaxed = data._store.get('pos_relaxed', None)
        
    if pos_relaxed is None:
        print(f"Sample {sample_idx} has no relaxed positions.")
        return

    atoms_relaxed = atoms_init.copy()
    if torch.is_tensor(pos_relaxed):
        atoms_relaxed.set_positions(pos_relaxed.cpu().numpy())
    else:
        atoms_relaxed.set_positions(pos_relaxed)

    # 3. Identify Adsorbate vs Catalyst
    # Tags: 0=Subsurface, 1=Surface, 2=Adsorbate
    try:
        tags = atoms_init.get_tags()
    except:
        tags = np.zeros(len(atoms_init))
    
    # --- Build Interactive View ---
    
    # We will create a combined view showing:
    # A. Catalyst (Grey) - Using positions from Initial (assumed mostly static)
    # B. Initial Adsorbate (Red, Transparent)
    # C. Relaxed Adsorbate (Blue, Solid)
    
    view = py3Dmol.view(width=800, height=500)
    
    # Helper to add atoms to view with specific style
    def add_atoms_to_view(atoms_obj, color, opacity=1.0, label=""):
        # Write single structure to PDB block
        buf = io.StringIO()
        write(buf, atoms_obj, format='xyz')
        model_txt = buf.getvalue()
        
        view.addModel(model_txt, 'xyz')
        
        # Apply style to the *last added model*
        # Coloring logic: py3Dmol doesn't support list-of-colors easily in python wrapper
        # So we add models separately based on their tag groups
        
        style = {
            'sphere': {'color': color, 'scale': 0.28, 'opacity': opacity},
            'stick': {'color': color, 'radius': 0.1, 'opacity': opacity}
        }
        view.setStyle({'model': -1}, style)

    # --- Group 1: Catalyst (Tags 0 & 1) ---
    catalyst_mask = (tags != 2)
    catalyst_atoms = atoms_init[catalyst_mask]
    if len(catalyst_atoms) > 0:
        add_atoms_to_view(catalyst_atoms, color='#C0C0C0', opacity=1.0) # Grey/Silver

    # --- Group 2: Initial Adsorbate (Tag 2) ---
    ads_mask = (tags == 2)
    ads_init = atoms_init[ads_mask]
    if len(ads_init) > 0:
        add_atoms_to_view(ads_init, color='#FF4444', opacity=0.5) # Red, Ghost

    # --- Group 3: Relaxed Adsorbate (Tag 2) ---
    ads_relax = atoms_relaxed[ads_mask]
    if len(ads_relax) > 0:
        add_atoms_to_view(ads_relax, color='#4444FF', opacity=1.0) # Blue, Solid
        
        # Add Lines connecting Initial -> Relaxed for adsorbate atoms
        # This draws a trajectory line for visual clarity
        init_pos = ads_init.get_positions()
        relax_pos = ads_relax.get_positions()
        
        for p_start, p_end in zip(init_pos, relax_pos):
             view.addCylinder({
                'start': {'x':p_start[0], 'y':p_start[1], 'z':p_start[2]},
                'end':   {'x':p_end[0],   'y':p_end[1],   'z':p_end[2]},
                'radius': 0.05,
                'color': 'yellow',
                'dashed': True
            })

    # Final Setup
    view.zoomTo()
    view.setBackgroundColor('white')
    
    # Labels
    print(f"Interactive View for Sample {sample_idx}")
    print("Legend: Grey=Catalyst | Red(Transparent)=Initial Adsorbate | Blue=Relaxed Adsorbate | Yellow Line=Movement")
    
    return view