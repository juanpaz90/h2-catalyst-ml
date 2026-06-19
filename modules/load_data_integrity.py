import os
import pickle
from typing import List, Optional, Any

import lmdb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from torch_geometric.data import Data


class OCPLmdbDataset(torch.utils.data.Dataset):
    """
    Directly reads PyG Data objects from OCP LMDB files.
    """
    _env_cache = {}

    def __init__(self, lmdb_path):
        super().__init__()
        assert os.path.isfile(lmdb_path), f"LMDB file not found: {lmdb_path}"
        self.lmdb_path = os.path.abspath(lmdb_path)

        # Reuse open environments to avoid LMDB "already open in this process" errors.
        if self.lmdb_path in self._env_cache:
            self.env = self._env_cache[self.lmdb_path]
        else:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
            self._env_cache[self.lmdb_path] = self.env

        # Read the length from LMDB metadata when available, otherwise infer it from numeric keys.
        with self.env.begin() as txn:
            length_value = txn.get(b'length')
            if length_value is not None:
                self.length = int(length_value.decode('utf-8'))
            else:
                numeric_keys = []
                for key, _ in txn.cursor():
                    try:
                        numeric_keys.append(int(key.decode('utf-8')))
                    except (ValueError, UnicodeDecodeError):
                        continue

                if not numeric_keys:
                    raise ValueError(f"Could not infer dataset length from LMDB: {lmdb_path}")

                self.length = max(numeric_keys) + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            # OCP stores keys as byte-encoded strings of the index
            datapoint_pickled = txn.get(str(idx).encode('utf-8'))
            data = pickle.loads(datapoint_pickled)
            return fix_pyg_data(data)
        

def fix_pyg_data(data_object: Any) -> Data:
    """
    Solves legacy PyG compatibility issues by reconstructing the Data object.
    The OCP dataset may contain objects created with older torch_geometric versions.
    These objects lack the '_store' attribute required by newer PyG versions (2.0+)
    
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
        

# The following 2 functions were used only for EDA, nothing else. 
def load_and_validate_dataset(db_path: str, num_samples: Optional[int] = None) -> List[Data]:
    """
    Eagerly loads LMDB data and returns a list of fixed, validated PyG Data objects.
    MUCHO OJO: Only use this for EDA or small subsets. For full training, use OCPLMDBDataset.
    
    Args:
        db_path (str): Path to the LMDB file.
        num_samples (Optional[int]): Number of samples to load. If None, loads all.
    
    Returns:
        List[Data]: A list of validated torch_geometric.data.Data objects.
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


def load_data_as_df(db_path: str, num_samples) -> pd.DataFrame:
    """
    Args:
        db_path: Path to the LMDB database
        num_samples (optional): Number of samples to load. If None, load all samples.

    Returns:
        pandas.DataFrame
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False)
    data_items = []

    with env.begin() as txn:
        cursor = txn.cursor()
        total_count = env.stat()['entries']
        pbar = tqdm(total=num_samples if num_samples else total_count, desc="Loading data", disable=False)

        for idx, (key, value) in enumerate(cursor):
            if num_samples is not None and idx >= num_samples:
                break

            data_object = pickle.loads(value)
            item = {}
            data_dict = data_object.__dict__

            # Extract all available keys
            for k in data_dict:
                if isinstance(data_dict.get(k), torch.Tensor):
                    item[k] = data_dict.get(k).numpy()
                else:
                    item[k] = data_dict.get(k)

            data_items.append(item)
            pbar.update(1)
        pbar.close()

    data_as_df = pd.DataFrame(data_items)
    return data_as_df