import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from ase.data import chemical_symbols

def get_symbol(atomic_num):
    """Maps atomic numbers to symbols using ASE's periodic table."""
    try:
        return chemical_symbols[int(atomic_num)]
    except (IndexError, TypeError, ValueError):
        return f"Unk-{atomic_num}"

def perform_chemical_eda(dataset_df):
    """
    Performs EDA on a pre-loaded dataset variable.
    Assumes dataset_variable is an iterable of PyG Data objects 
    or objects with .atomic_numbers and .tags attributes.
    """
    all_atomic_nums = []
    adsorbate_elements = []
    catalyst_elements = []
    natoms_list = []
    tag_counts = Counter()
    unique_elements_found = set()

    print(f"Starting EDA on {len(dataset_df)} samples...")

    for data in dataset_df:
        # Compatibility check: handle both PyG Data objects and dictionaries
        if hasattr(data, 'atomic_numbers'):
            atomic_numbers = data.atomic_numbers
            tags = data.tags
        elif isinstance(data, dict):
            atomic_numbers = data['atomic_numbers']
            tags = data['tags']
        else:
            # Handle PyG Legacy _store if necessary
            atomic_numbers = getattr(data, '_store', {}).get('atomic_numbers', data.atomic_numbers)
            tags = getattr(data, '_store', {}).get('tags', data.tags)

        # Ensure tensors are on CPU and converted to numpy for stats
        atomic_numbers_np = atomic_numbers.view(-1).cpu().numpy()
        tags_np = tags.view(-1).cpu().numpy()
        
        unique_elements_found.update(atomic_numbers_np.tolist())

        # 1. System Scale
        natoms_list.append(len(atomic_numbers_np))
        
        # 2. Atomic Distributions
        all_atomic_nums.extend(atomic_numbers_np.tolist())
        
        # 3. Separate Adsorbate (Tag 2) vs Catalyst (Tags 0, 1)
        adsorbate_mask = (tags_np == 2)
        catalyst_mask = (tags_np < 2)
        
        adsorbate_elements.extend(atomic_numbers_np[adsorbate_mask].tolist())
        catalyst_elements.extend(atomic_numbers_np[catalyst_mask].tolist())
        
        # 4. Tag Ratios
        tag_counts.update(tags_np.tolist())

    # --- Visualizations ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot A: System Sizes
    axes[0, 0].hist(natoms_list, bins=min(30, len(natoms_list)), color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Distribution of Atoms per System")
    axes[0, 0].set_xlabel("Number of Atoms")

    # Plot B: Global Element Frequency (Top 10)
    elem_counts = Counter([get_symbol(n) for n in all_atomic_nums])
    if elem_counts:
        labels, values = zip(*elem_counts.most_common(10))
        axes[0, 1].bar(labels, values, color='salmon')
    axes[0, 1].set_title("Top 10 Elements Discovered (Global)")

    # Plot C: Adsorbate Element Diversity (Pie Chart)
    ads_counts = Counter([get_symbol(n) for n in adsorbate_elements])
    if ads_counts:
        labels, values = zip(*ads_counts.items())
        axes[1, 0].pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    axes[1, 0].set_title("Adsorbate Composition (Tag 2)")

    # Plot D: Tag Distribution (Role of atoms)
    tag_labels = ["Fixed (0)", "Surface (1)", "Adsorbate (2)"]
    tag_vals = [tag_counts.get(0, 0), tag_counts.get(1, 0), tag_counts.get(2, 0)]
    axes[1, 1].bar(tag_labels, tag_vals, color=['grey', 'blue', 'green'])
    axes[1, 1].set_title("Atom Roles Distribution (Tags)")

    plt.tight_layout()
    plt.show()

    # Summary
    discovered_symbols = sorted([get_symbol(n) for n in unique_elements_found])
    print(f"\n--- EDA Summary ---")
    print(f"Total systems analyzed: {len(dataset_df)}")
    print(f"Avg atoms/system: {np.mean(natoms_list):.2f} (Min: {np.min(natoms_list)}, Max: {np.max(natoms_list)})")
    print(f"Elements present: {', '.join(discovered_symbols)}")