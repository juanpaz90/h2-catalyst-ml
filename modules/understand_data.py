import lmdb
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from torch_geometric.data import Data
from modules.data_integrity import fix_pyg_data


def get_lmdb_sample(lmdb_path, index):
    """
    Retrieves a single sample from the LMDB at the specified index.
    Returns the deserialized PyG Data object.
    """
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    
    with env.begin(write=False) as txn:
        # Fetch data by string index key
        key = f"{index}".encode("ascii")
        data_bytes = txn.get(key)
        
        if data_bytes is None:
            print(f"Sample index {index} not found.")
            return None
            
        data_obj = pickle.loads(data_bytes)

    return fix_pyg_data(data_obj)


def ocp_to_ase(data):
    """
    Converts an OCP PyG Data object to an ASE Atoms object for visualization.
    """
    if not data:
        return None
        
    # Extract atomic numbers and positions
    # Note: Using getattr(data, 'attr') is safer than data.attr for optional fields
    numbers = data.atomic_numbers.numpy()
    positions = data.pos.numpy()
    
    # Create ASE object
    # cell and pbc (periodic boundary conditions) are usually needed for bulk/surfaces
    cell = data.cell.numpy().reshape(3, 3) if hasattr(data, 'cell') and data.cell is not None else None
    pbc = True if cell is not None else False
    
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc
    )
    
    # Add tags if available (0=Adsorbate, 1=Surface, 2=Subsurface)
    if hasattr(data, 'tags') and data.tags is not None:
        atoms.set_tags(data.tags.numpy())
        
    return atoms


def add_legend_to_ax(ax, atoms):
        """Creates a custom legend for the elements present in the Atoms object."""
        unique_elements = np.unique(atoms.numbers)
        legend_handles = []
        
        for atomic_num in unique_elements:
            # Get symbol and color for the element
            symbol = chemical_symbols[atomic_num]
            color = jmol_colors[atomic_num]
            
            # Create a patch for the legend
            patch = Patch(facecolor=color, edgecolor='black', label=symbol)
            legend_handles.append(patch)
            
        ax.legend(handles=legend_handles, loc='upper right', title="Elements", framealpha=0.9)


def compare_samples(lmdb_path, idx1, idx2):
    """
    Loads two samples, prints a side-by-side comparison of their attributes,
    and visualizes their structures.
    """
    print(f"--- Comparing Samples {idx1} and {idx2} ---")
    
    data1 = get_lmdb_sample(lmdb_path, idx1)
    data2 = get_lmdb_sample(lmdb_path, idx2)
    
    if not data1 or not data2:
        return

    # --- 1. Attribute Comparison ---
    print(f"{'Attribute':<25} | {'Sample ' + str(idx1):<30} | {'Sample ' + str(idx2):<30}")
    print("-" * 90)
    
    # Compare Chemical Formula (via atomic numbers)
    atoms1 = ocp_to_ase(data1)
    atoms2 = ocp_to_ase(data2)
    print(f"{'Formula':<25} | {atoms1.get_chemical_formula():<30} | {atoms2.get_chemical_formula():<30}")
    
    # Compare Energy
    # Try multiple common keys for energy
    def get_energy(d):
        for key in ['y_relaxed', 'y', 'energy']:
            if hasattr(d, key) and getattr(d, key) is not None:
                val = getattr(d, key)
                # Handle 0-dim tensor vs scalar
                if torch.is_tensor(val):
                    return f"{val.item():.4f}"
                return f"{val:.4f}"
        return "N/A"

    e1 = get_energy(data1)
    e2 = get_energy(data2)
    
    print(f"{'Energy (eV)':<25} | {e1:<30} | {e2:<30}")
    
    # Compare Number of Atoms
    print(f"{'Total Atoms':<25} | {len(data1.atomic_numbers):<30} | {len(data2.atomic_numbers):<30}")
    
    # Compare Adsorbate Atoms (Tag == 0) (If tags exist)
    if hasattr(data1, 'tags') and data1.tags is not None:
        ads1 = (data1.tags == 0).sum().item()
        ads2 = (data2.tags == 0).sum().item()
        print(f"{'Adsorbate Atoms':<25} | {str(ads1):<30} | {str(ads2):<30}")

    print("\n")

    # --- 2. Visualization ---
    fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Sample 1
    # rotation='10x,10y,0z' gives a slightly tilted view to see depth
    plot_atoms(atoms1, axarr[0], radii=0.5, rotation=('10x,45y,0z'))
    axarr[0].set_title(f"Sample {idx1}: {atoms1.get_chemical_formula()}")
    axarr[0].set_axis_off()
    add_legend_to_ax(axarr[0], atoms1)
    
    # Plot Sample 2
    plot_atoms(atoms2, axarr[1], radii=0.5, rotation=('10x,45y,0z'))
    axarr[1].set_title(f"Sample {idx2}: {atoms2.get_chemical_formula()}")
    axarr[1].set_axis_off()
    add_legend_to_ax(axarr[1], atoms2)
    
    plt.tight_layout()
    plt.show()