import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
from modules.SchNet_Base import OCPLmdbDataset


# Baseline mean target from EDA
MEAN_TARGET = -1.54 


class SchNetWithMHA(nn.Module):
    """
    Final Custom Architecture integrating SchNet local physics 
    with Multihead Attention global context.
    """
    def __init__(self, hidden_channels, num_filters, num_interactions, cutoff, num_gaussians, num_heads):
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
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
            
        # Global Multihead Attention Step
        h_dense, mask = to_dense_batch(h, batch.batch)
        attn_out, _ = self.mha(query=h_dense, key=h_dense, value=h_dense, key_padding_mask=~mask)
        
        # Residual connection and flatten back to sparse
        h_attn_sparse = attn_out[mask]
        h = h + h_attn_sparse
        
        # Final MLPs and System-Level Pooling
        h = self.schnet.lin1(h)
        h = self.schnet.act(h)
        h = self.schnet.lin2(h)
        
        return global_mean_pool(h, batch.batch)


def  initialize_mha_model(device, hidden_channels, num_filters, num_interactions, cutoff, num_gaussians, num_heads):
    model = SchNetWithMHA(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        cutoff=cutoff,
        num_gaussians=num_gaussians,
        num_heads=num_heads
    ).to(device)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, mean_energy, ewt_threshold=0.02):
    model.train()
    total_loss = 0
    total_ewt = 0
    
    for batch in tqdm(loader, desc="Training Epoch"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
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
            out = model(batch)
            target = batch.y_relaxed.view(-1) - mean_energy
            loss = criterion(out.view(-1), target)
            total_loss += loss.item() * batch.num_graphs
            total_ewt += (torch.abs(out.view(-1) - target) <= ewt_threshold).sum().item()
            
    mae = total_loss / len(loader.dataset)
    ewt = (total_ewt / len(loader.dataset)) * 100 
    return mae, ewt

def configure_MHA_and_run_training(train_lmdb_path,
                               val_lmdb_path, 
                               epochs, 
                               batch_size, 
                               lr, 
                               hidden_channels, 
                               num_filters, 
                               num_interactions, 
                               num_gaussians, 
                               cutoff, 
                               num_heads, 
                               early_stop_patience
                               ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Final Training on: {device}")

    train_dataset = OCPLmdbDataset(train_lmdb_path)
    val_dataset = OCPLmdbDataset(val_lmdb_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize MHA model with optimal hyperparams
    model = initialize_mha_model(device, hidden_channels, num_filters, num_interactions, cutoff, num_gaussians, num_heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.L1Loss() 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )
    best_val_mae = float('inf')

    # Track metrics for visualization
    epochs_no_improve = 0 
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
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'SchNet_MHA_model.pt')
            print("Checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    return model, train_maes, val_maes, train_ewts, val_ewts, val_loader, device