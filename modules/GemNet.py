import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

from modules.SchNet_Base import OCPLmdbDataset

class GaussianSmearing(nn.Module):
    """Expands distances into a Gaussian basis."""
    def __init__(self, start=0.0, stop=6.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GemNetInteraction(MessagePassing):
    """Simplified GemNet Directional Message Passing layer."""
    def __init__(self, hidden_channels, num_gaussians):
        super().__init__(aggr='add') # Additive aggregation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels + num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp(z)

class GemNetOC_GNN(nn.Module):
    """
    GemNet-OC inspired architecture adapted for OC20 IS2RE.
    Includes PBC correction and Tags embedding.
    """
    def __init__(self, hidden_channels=128, num_blocks=4, num_gaussians=50, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        
        # Embeddings (Atomic numbers + Tags)
        self.atom_emb = nn.Embedding(100, hidden_channels)
        self.tag_emb = nn.Embedding(3, hidden_channels) 
        
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.interactions = nn.ModuleList([
            GemNetInteraction(hidden_channels, num_gaussians)
            for _ in range(num_blocks)
        ])
        
        self.out_net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        z = data.atomic_numbers.long().squeeze()
        pos = data.pos
        tags = data.tags.long().squeeze()
        edge_index = data.edge_index
        
        # Combine atomic embedding with tag embedding
        x = self.atom_emb(z) + self.tag_emb(tags)
        
        # PBC CORRECTIONS: Calculate true distance using cell and cell_offsets
        row, col = edge_index
        if hasattr(data, 'cell') and hasattr(data, 'cell_offsets'):
            # reshape cell to ensure it's [num_graphs, 3, 3]
            cell_view = data.cell.view(-1, 3, 3)
            
            if hasattr(data, 'batch') and data.batch is not None:
                # Map each edge to its corresponding graph index
                edge_batch = data.batch[row]
                # Gather the 3x3 cell matrix for each edge
                edge_cell = cell_view[edge_batch]
            else:
                # Fallback if no batching is used (single graph)
                edge_cell = cell_view.expand(data.cell_offsets.size(0), -1, -1)
                
            # Now both operands have the same first dimension (number of edges 'e')
            offsets = torch.einsum('ei,eij->ej', data.cell_offsets.float(), edge_cell.float())
            edge_vec = pos[row] - pos[col] + offsets
        else:
            edge_vec = pos[row] - pos[col]
            
        edge_dist = torch.norm(edge_vec, dim=-1)
        edge_attr = self.distance_expansion(edge_dist)
        
        for interaction in self.interactions:
            x = x + interaction(x, edge_index, edge_attr)
            
        if hasattr(data, 'batch') and data.batch is not None:
            from torch_geometric.nn import global_mean_pool
            x_pooled = global_mean_pool(x, data.batch)
        else:
            x_pooled = x.mean(dim=0, keepdim=True)
            
        energy_pred = self.out_net(x_pooled)
        return energy_pred.squeeze(-1)

def configure_and_run_gemnet_training(train_data_path, val_data_path, epochs=20, batch_size=32, lr=0.0005):
    """
    Configures DataLoaders, initializes GemNetOC_GNN, trains the model, 
    and returns variables formatted exactly like the baseline modules.
    """
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = OCPLmdbDataset(train_data_path)
    val_dataset = OCPLmdbDataset(val_data_path)

    # 2. Setup DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model, Optimizer, Scheduler, and Loss
    model = GemNetOC_GNN(hidden_channels=128, num_blocks=4).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.L1Loss() # MAE Loss

    train_maes = []
    val_maes = []

    # 4. Training Loop
    print(f"Starting GemNet-OC training for {epochs} epochs...")
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            preds = model(batch)
            loss = criterion(preds, batch.y_relaxed.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0) # Prevent exploding gradients
            optimizer.step()
            
            train_loss += loss.item() * batch.num_graphs
            
        train_mae = train_loss / len(train_loader.dataset)
        train_maes.append(train_mae)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                preds = model(batch)
                loss = criterion(preds, batch.y_relaxed.float())
                val_loss += loss.item() * batch.num_graphs
                
        val_mae = val_loss / len(val_loader.dataset)
        val_maes.append(val_mae)
        
        scheduler.step()
        print(f"Epoch {epoch+1:03d}/{epochs} | Train MAE: {train_mae:.4f} eV | Val MAE: {val_mae:.4f} eV")

    # 5. Extract Final Predictions & Targets for Analysis
    print("Extracting final predictions on validation set...")
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y_relaxed.cpu().numpy())

    # Return predictions and targets as NumPy arrays for compatibility with results.py
    return model, train_maes, val_maes, val_loader, device, np.array(all_preds), np.array(all_targets)