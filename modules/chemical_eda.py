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
    Performs EDA on a list of PyTorch Geometric Data objects,
    generating four separate, high-resolution plots.
    """
    all_atomic_nums = []
    adsorbate_elements = []
    natoms_list = []
    tag_counts = Counter()
    unique_elements_found = set()

    print(f"Analyzing {len(dataset_df)} systems...")

    for data in dataset_df:
        # Compatibility check for tensor extraction
        if hasattr(data, 'atomic_numbers'):
            atomic_numbers = data.atomic_numbers
            tags = data.tags
        else:
            atomic_numbers = getattr(data, '_store', {}).get('atomic_numbers', None)
            tags = getattr(data, '_store', {}).get('tags', None)

        if atomic_numbers is None or tags is None:
            continue

        atomic_numbers_np = atomic_numbers.view(-1).cpu().numpy()
        tags_np = tags.view(-1).cpu().numpy()
        
        unique_elements_found.update(atomic_numbers_np.tolist())
        natoms_list.append(len(atomic_numbers_np))
        all_atomic_nums.extend(atomic_numbers_np.tolist())
        
        # Adsorbate only (Tag 2)
        adsorbate_mask = (tags_np == 2)
        adsorbate_elements.extend(atomic_numbers_np[adsorbate_mask].tolist())
        tag_counts.update(tags_np.tolist())

    # --- Plot 1: System Size Distribution ---
    # Vertical Axis: Number of Systems (Frequency)
    plt.figure(figsize=(8, 5))
    plt.hist(natoms_list, bins=min(30, len(natoms_list)), color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of System Sizes (Complexity)")
    plt.xlabel("Number of Atoms in the System")
    plt.ylabel("Number of Systems (Frequency)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- Plot 2: Global Atomic Frequency ---
    # Vertical Axis: Total Count of individual atoms
    plt.figure(figsize=(8, 5))
    elem_counts = Counter([get_symbol(n) for n in all_atomic_nums])
    if elem_counts:
        labels, values = zip(*elem_counts.most_common(10))
        plt.bar(labels, values, color='salmon', edgecolor='black', alpha=0.8)
    plt.title("Top 10 Most Frequent Elements (Global)")
    plt.xlabel("Chemical Element Symbol")
    plt.ylabel("Total Atom Count")
    plt.show()

    # --- Plot 3: Adsorbate Composition ---
    # Represents proportions (percentage of atoms in Tag 2)
    plt.figure(figsize=(8, 5))
    ads_counts = Counter([get_symbol(n) for n in adsorbate_elements])
    if ads_counts:
        labels, values = zip(*ads_counts.items())
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Adsorbate Chemical Diversity (Tag 2)")
    plt.show()

    # --- Plot 4: Tag Distribution (Role of atoms) ---
    # Vertical Axis: Total count of atoms assigned to each role
    plt.figure(figsize=(8, 5))
    tag_labels = ["Fixed (0)", "Surface (1)", "Adsorbate (2)"]
    tag_vals = [tag_counts.get(0, 0), tag_counts.get(1, 0), tag_counts.get(2, 0)]
    colors = ['#95a5a6', '#3498db', '#2ecc71'] # Grey, Blue, Green
    plt.bar(tag_labels, tag_vals, color=colors, edgecolor='black', alpha=0.8)
    plt.title("Distribution of Atom Roles (Tags)")
    plt.xlabel("Atom Classification (Tag)")
    plt.ylabel("Total Atom Count")
    plt.show()

    print(f"\n--- Summary ---")
    print(f"Elements Discovered: {sorted([get_symbol(n) for n in unique_elements_found])}")
    print(f"Vertical Axis Meaning:")
    print(f" - System Sizes: Count of distinct catalyst+adsorbate systems.")
    print(f" - Global Frequency: Absolute sum of atoms across all loaded systems.")
    print(f" - Tag Distribution: Population of atoms categorized by their mobility (Fixed vs Free).")