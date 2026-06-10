import os
import lmdb
import pickle
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool
import numpy as np
from tqdm import tqdm
from modules.load_data_integrity import fix_pyg_data


# Baseline mean target from EDA
MEAN_TARGET = -1.54 


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


def initialize_base_model(device, hidden_channels, num_filters, num_interactions, num_gaussians, cutoff):
    model = SchNet(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
    ).to(device)
    return model


def get_model_output(model, batch):
    """
    Bypasses SchNet.forward() to avoid the torch-cluster/radius_graph dependency.
    Manually calculates energy using the existing edge_index from the dataset.
    Incorporates Periodic Boundary Conditions (PBC) via cell & cell_offsets.
    """

    # Support both plain SchNet and wrapper models (e.g., SchNetWithMHA)
    # by selecting the inner SchNet module when present.
    schnet_model = model.schnet if hasattr(model, 'schnet') else model

    # 1. Get initial embeddings for atomic numbers
    h = schnet_model.embedding(batch.atomic_numbers)
    row, col = batch.edge_index

    # 2. Map each edge to its parent graph in the batch / Calculate structural shift across periodic boundaries
    edge_batch = batch.batch[row] 
    
    # 3. Extract the 3x3 cell matrix for each edge [E, 3, 3]
    cells = batch.cell[edge_batch] 

    # 4. Multiply cell_offsets [E, 3] by the cell matrices to get physical shift [E, 3]
    offsets = batch.cell_offsets.float()
    shift = (offsets.unsqueeze(-1) * cells).sum(dim=1) 
    
    # 5. Calculate true distance: norm(pos_row - pos_col + periodic_shift)
    dist = torch.norm(batch.pos[row] - batch.pos[col] + shift, dim=-1)
    
    edge_attr = schnet_model.distance_expansion(dist)
    
    for interaction in schnet_model.interactions:
        h = h + interaction(h, batch.edge_index, dist, edge_attr)
        
    h = schnet_model.lin1(h)
    h = schnet_model.act(h)
    h = schnet_model.lin2(h)
    
    # Pool the node-level energies into a system-level energy
    return global_mean_pool(h, batch.batch)


def train_one_epoch(model, loader, optimizer, criterion, device, mean_energy, ewt_threshold=0.02):
    # ewt_threshold=0.02 ==> threshold for Chemical Accuracy.
    model.train()
    total_loss = 0
    total_ewt = 0
    
    for batch in tqdm(loader, desc="Training Epoch"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Using the custom output function to bypass forward() and radius_graph
        out = get_model_output(model, batch)
        
        # IS2RE Target: Relaxed Energy (y_relaxed)
        target = batch.y_relaxed.view(-1) - mean_energy
        
        loss = criterion(out.view(-1), target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        total_ewt += (torch.abs(out.view(-1) - target) <= ewt_threshold).sum().item()

    mae = total_loss / len(loader.dataset)
    ewt = (total_ewt / len(loader.dataset)) * 100
    return mae, ewt


def validate(model, loader, criterion, device, mean_energy, ewt_threshold=0.02):
    model.eval()
    total_loss = 0
    total_ewt = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Using the custom output function to bypass forward() and radius_graph
            out = get_model_output(model, batch)
            target = batch.y_relaxed.view(-1) - mean_energy
            loss = criterion(out.view(-1), target)
            total_loss += loss.item() * batch.num_graphs
            total_ewt += (torch.abs(out.view(-1) - target) <= ewt_threshold).sum().item()
    
    mae = total_loss / len(loader.dataset)
    ewt = (total_ewt / len(loader.dataset)) * 100
    return mae, ewt


def configure_base_and_run_training(train_lmdb_path, 
                                    val_lmdb_path, 
                                    epochs, 
                                    batch_size, 
                                    lr, 
                                    hidden_channels, 
                                    num_filters, 
                                    num_interactions, 
                                    num_gaussians, 
                                    cutoff, 
                                    early_stop_patience
                                    ):
    """
    Main training configuration and execution.
    Returns:
        model: The trained SchNet model.
        train_maes: List of training MAE per epoch.
        val_maes: List of validation MAE per epoch.
        val_loader: The DataLoader used for validation.
        device: The device (cpu/cuda) used for training.
        preds: Numpy array of final predictions on validation set.
        targets: Numpy array of final ground truth targets on validation set.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Training on: {device}")

    train_dataset = OCPLmdbDataset(train_lmdb_path)
    val_dataset = OCPLmdbDataset(val_lmdb_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = initialize_base_model(device, hidden_channels, num_filters, num_interactions, num_gaussians, cutoff)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() # L1Loss is the exact mathematical calculation of MAE, so we keep it.
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    best_val_mae = float('inf')
    
    # Track metrics for visualization
    epochs_no_improve = 0 # Counter for early stopping
    train_maes, val_maes, train_ewts, val_ewts = [], [], [], []

    for epoch in range(epochs):
        train_mae, train_ewt = train_one_epoch(model, train_loader, optimizer, criterion, device, MEAN_TARGET)
        val_mae, val_ewt = validate(model, val_loader, criterion, device, MEAN_TARGET)
        
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_ewts.append(train_ewt)
        val_ewts.append(val_ewt)
        scheduler.step(val_mae)
        
        print(f"Epoch {epoch+1:02d} | Train MAE: {train_mae:.4f} eV, EwT: {train_ewt:.2f}% | Val MAE: {val_mae:.4f} eV, EwT: {val_ewt:.2f}%")
        
        # Early Stopping Logic
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0 # Reset counter
            torch.save(model.state_dict(), 'SchNet_base_model.pt')
            print("Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    return model, train_maes, val_maes, train_ewts, val_ewts, val_loader, device
