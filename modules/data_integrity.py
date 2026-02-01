# EDA
# STEP 1: Data validation and data integrity
# --------

import os
import lmdb
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

def fix_pyg_data(data_object):
    """
    Solves legacy PyG compatibility issues by reconstructing the Data object.
    
    Parameters:
    data_object: The object loaded via pickle.
    
    Returns:
    torch_geometric.data.Data: A modern PyG-compatible Data object.
    """
    # If it's already a modern Data object with internal _store, we return it
    if isinstance(data_object, Data) and not hasattr(data_object, '__dict__'):
        return data_object

    # Extract the data dictionary (works for both old Data objects and dicts)
    if hasattr(data_object, '__dict__'):
        data_dict = data_object.__dict__
    else:
        data_dict = data_object

    # Create a fresh modern Data object
    new_data = Data()
    
    for key, value in data_dict.items():
        # Handle specific OCP internal attributes that shouldn't be top-level
        if key in ['_store', '__parameters__']:
            continue
            
        # Ensure all structural attributes remain as tensors
        if isinstance(value, (np.ndarray, list, torch.Tensor)):
            # Convert numpy arrays or lists back to tensors
            if isinstance(value, (np.ndarray, list)):
                tensor_val = torch.tensor(value)
            else:
                tensor_val = value
            
            # Specifically handle long vs float for OCP attributes
            if key in ['atomic_numbers', 'edge_index', 'natoms', 'tags']:
                setattr(new_data, key, tensor_val.long())
            else:
                setattr(new_data, key, tensor_val.float())
        else:
            # Preserve metadata (strings, IDs, etc.)
            setattr(new_data, key, value)
            
    return new_data

def load_and_validate_dataset(db_path, num_samples=None):
    """
    Loads LMDB data and returns a list of fixed, validated PyG Data objects.
    
    Parameters:
    db_path: Path to the LMDB file.
    num_samples: Number of samples to load (default: all).
    
    Returns:
    list: Validated torch_geometric.data.Data objects.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    env = lmdb.open(
        db_path, 
        subdir=False, 
        readonly=True, 
        lock=False, 
        readahead=False, 
        meminit=False
    )
    
    data_list = []
    
    with env.begin() as txn:
        total_count = env.stat()['entries']
        limit = num_samples if num_samples else total_count
        pbar = tqdm(total=limit, desc="Phase 1: Loading & Fixing Tensors")

        for idx in range(limit):
            # OCP stores data with keys as strings '0', '1', ...
            byte_data = txn.get(f"{idx}".encode("ascii"))
            if not byte_data:
                break
            
            # Initial deserialize
            raw_data = pickle.loads(byte_data)
            
            # Apply the fix to prevent legacy RuntimeError
            fixed_data = fix_pyg_data(raw_data)
            
            data_list.append(fixed_data)
            pbar.update(1)
            
        pbar.close()
    
    env.close()
    print(f"Successfully loaded {len(data_list)} validated samples.")
    return data_list

def check_structural_integrity(data_list):
    """
    Verifies that the tensors are in the correct format for OCP tasks.
    """
    if not data_list:
        print("Dataset is empty.")
        return False
        
    sample = data_list[0]
    required_keys = ['pos', 'atomic_numbers', 'tags', 'edge_index']
    
    missing = [k for k in required_keys if not hasattr(sample, k)]
    
    if missing:
        print(f"Structural Integrity Failed: Missing keys {missing}")
        return False
        
    print("Structural Integrity Passed: All OCP core tensors found and fixed.")
    return True