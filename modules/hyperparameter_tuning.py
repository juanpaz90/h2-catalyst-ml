import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool
from torch_geometric.utils import to_dense_batch
import numpy as np
import optuna 

from modules.SchNet_Base import OCPLmdbDataset

# --- Custom SchNet with Multihead Attention ---
class SchNetWithMHA(nn.Module):
    def __init__(self, hidden_channels, num_filters, num_interactions, cutoff, num_gaussians, num_heads):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        
        # Step 2: Integrating Multihead Attention (MHA)
        # batch_first=True aligns with [Batch, Max_Atoms, Embed_Dim] shapes
        self.mha = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads, batch_first=True)
        
    def forward(self, batch):
        h = self.schnet.embedding(batch.atomic_numbers)
        row, col = batch.edge_index
        
        # PBC Physics Fix
        edge_batch = batch.batch[row] 
        cells = batch.cell[edge_batch] 
        offsets = batch.cell_offsets.float()
        shift = (offsets.unsqueeze(-1) * cells).sum(dim=1) 
        
        dist = torch.norm(batch.pos[row] - batch.pos[col] + shift, dim=-1)
        edge_attr = self.schnet.distance_expansion(dist)
        
        for interaction in self.schnet.interactions:
            h = h + interaction(h, batch.edge_index, dist, edge_attr)
            
        # Step 1: Bridging Sparse Graphs to Dense Attention
        # h_dense shape: [Batch_Size, Max_Atoms, Hidden_Channels]
        # mask shape: [Batch_Size, Max_Atoms] (True for real atoms, False for padding)
        h_dense, mask = to_dense_batch(h, batch.batch)
        
        # Step 3: Attention Masking (Crucial for Physics)
        # We pass ~mask (bitwise NOT) because key_padding_mask expects True for atoms to IGNORE.
        attn_out, _ = self.mha(query=h_dense, key=h_dense, value=h_dense, key_padding_mask=~mask)
        
        # Re-flatten back to sparse tensor and add residual connection
        h_attn_sparse = attn_out[mask]
        h = h + h_attn_sparse
        
        # Final MLPs and System-Level Pooling
        h = self.schnet.lin1(h)
        h = self.schnet.act(h)
        h = self.schnet.lin2(h)
        
        return global_mean_pool(h, batch.batch)


# --- Step 5: Guided Search & Validation Scoring ---
def objective(trial, train_dataset_path, val_dataset_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 4: Defining the Optuna Search Space
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_filters = trial.suggest_categorical("num_filters", [64, 128, 256])
    num_interactions = trial.suggest_int("num_interactions", 3, 6)
    cutoff = trial.suggest_float("cutoff", 5.0, 8.0)
    num_gaussians = trial.suggest_int("num_gaussians", 25, 100)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_heads = trial.suggest_categorical("num_heads", [4, 8]) 
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    
    # Enforce MHA architecture rule: embed_dim must be divisible by num_heads
    if hidden_channels % num_heads != 0:
        raise optuna.exceptions.TrialPruned()

    train_dataset = OCPLmdbDataset(train_dataset_path)
    val_dataset = OCPLmdbDataset(val_dataset_path) 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = SchNetWithMHA(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        cutoff=cutoff,
        num_gaussians=num_gaussians,
        num_heads=num_heads
    ).to(device)
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
    criterion = nn.L1Loss() # L1Loss is the exact mathematical calculation of MAE, so we keep it.
    MEAN_TARGET = -1.54
    
    best_val_mae = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 4 # Fast fail for bad hyperparameters
    max_epochs = 15 
    
    for epoch in range(max_epochs):
        # Training Phase
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            target = batch.y_relaxed.view(-1) - MEAN_TARGET
            loss = criterion(out.view(-1), target)
            loss.backward()
            optimizer.step()
            
        # Validation Phase
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                target = batch.y_relaxed.view(-1) - MEAN_TARGET
                loss = criterion(out.view(-1), target)
                total_loss += loss.item() * batch.num_graphs
                
        val_mae = total_loss / len(val_dataset)
        
        # Step 5: Guided Search Pruning 
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                break
                
    return best_val_mae

def get_hyperparameters(train_dataset_path, val_dataset_path):
    print("Starting Hyperparameter Search via Optuna (MHA Enabled)...")
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(
        lambda trial: objective(trial, train_dataset_path, val_dataset_path),
        n_trials = 20
    )
    
    # --- Generate Final Output Table ---
    print("\n" + "="*50)
    print(">> HYPERPARAMETER SEARCH COMPLETE <<")
    print("="*50)
    print(f"Best Validation MAE achieved: {study.best_value:.4f} eV")
    print(f"Trial Number: {study.best_trial.number}")
    print("\nUse the following values in your final Training Script:\n")
    
    print(f"| {'Parameter':<20} | {'Optimal Value':<20} |")
    print(f"|{'-'*22}|{'-'*22}|")
    for key, value in study.best_trial.params.items():
        val_str = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"| {key:<20} | {val_str:<20} |")
    print("="*50)