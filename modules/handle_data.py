import os 
import lmdb
from tqdm import tqdm
import pandas as pd
import pickle
import torch
import numpy as np
from torch_geometric.data import Data

## PANDAS DATAFRAME
def load_df_data(db_path: str, num_samples):
    """
    Load Dataset

    Parameters:
    db_path: Path to the LMDB database
    num_samples (optional): Number of samples to load. If None, load all samples.

    Returns:
    pandas.DataFrame
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    # Open the LMDB environment
    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False)

    # Lists to store the data
    data_items = []

    # Open a transaction
    with env.begin() as txn:
        # Get a cursor
        cursor = txn.cursor()

        # Determine total number of entries for progress bar
        total_count = env.stat()['entries']
        pbar = tqdm(total=num_samples if num_samples else total_count, desc="Loading data", disable=False)

        # Iterate over the database
        for idx, (key, value) in enumerate(cursor):
            if num_samples is not None and idx >= num_samples:
                break

            # Safely deserialize without triggering attribute access
            data_object = pickle.loads(value)

            # Access internal attributes directly to avoid PyG's __getattr__
            item = {}
            data_dict = data_object.__dict__

            # Extract all available keys
            for k in data_dict:
                if isinstance(data_dict.get(k), torch.Tensor):
                    item[k] = data_dict.get(k).numpy()
                else:
                    item[k] = data_dict.get(k)

            # Add to our list
            data_items.append(item)
            pbar.update(1)

        pbar.close()

    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(data_items)
    return df

## PYTORCH TENSOR
def load_ocp_dataset(db_path: str, num_samples=None):
    """
    Loads OCP data from LMDB and returns a list of PyTorch Geometric Data objects.
    This preserves Tensors instead of converting them to NumPy/Pandas.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
    data_list = []

    with env.begin() as txn:
        total_count = env.stat()['entries']
        limit = num_samples if num_samples else total_count
        pbar = tqdm(total=limit, desc="Loading OCP Tensors")

        for idx in range(limit):
            byte_data = txn.get(f"{idx}".encode("ascii"))
            if not byte_data:
                break
            
            # Load the pickled object
            data_object = pickle.loads(byte_data)
            
            # Ensure it is a proper Data object and tensors are on CPU
            # If the object is a dictionary or legacy format, we wrap it
            if not isinstance(data_object, Data):
                # OCP objects often store data in a ._store or __dict__
                d = data_object.__dict__ if hasattr(data_object, '__dict__') else data_object
                new_data = Data()
                for k, v in d.items():
                    setattr(new_data, k, v)
                data_list.append(new_data)
            else:
                data_list.append(data_object)
            
            pbar.update(1)
        pbar.close()
    
    env.close()
    return data_list