import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import math

# 1. Configuration
CONFIG = {
    "batch_size": 128,
    "lr": 1e-4,
    "weight_decay": 1e-5,  # L2 Regularization
    "epochs": 20,
    "cutoff": 6.0,         # 6 Angstrom radius
    "hidden_channels": 128,
    "num_filters": 128,
    "num_interactions": 3,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_workers": 2, # Reduced slightly to avoid overhead if dataset is small
    "pin_memory": True
}

# --- Custom SchNet Implementation (No torch-cluster dependency) ---

def radius_graph_custom(pos, r, batch):
    """
    Computes radius graph using pure PyTorch cdist.
    Avoids torch-cluster dependency.
    """
    # 1. Compute pairwise distances
    # Note: For very large batches, this N^2 operation might be memory intensive.
    # But for OCP batches (~3000-5000 atoms), it fits on modern GPUs.
    dist = torch.cdist(pos, pos)
    
    # 2. Create batch mask to ensure edges only exist within the same graph
    # batch_mask[i, j] is True if atom i and atom j belong to the same graph
    batch_mask = batch.unsqueeze(1) == batch.unsqueeze(0)
    
    # 3. Apply masks
    # Distance must be <= r AND atoms must be in the same graph
    mask = (dist <= r) & batch_mask
    
    # 4. Remove self-loops (diagonal)
    mask.fill_diagonal_(False)
    
    # 5. Extract indices
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    
    return edge_index, dist[src, dst]

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softplus(x) - math.log(2.0)
    
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(), # Corrected usage
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus() # Corrected usage
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = torch.nn.Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * math.pi / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)
        x = self.lin1(x)
        return self.propagate(edge_index, x=x, W=W)

    def message(self, x_j, W):
        return x_j * W

    def update(self, aggr_out):
        return self.lin2(aggr_out)

class CustomSchNet(nn.Module):
    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=3, num_gaussians=50, cutoff=6.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        
        self.embedding = nn.Embedding(100, hidden_channels) # Atomic numbers up to 100
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)
            
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus() # Corrected usage
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

    def forward(self, z, pos, batch):
        # 1. Edge Construction (Custom, no torch-cluster)
        edge_index, edge_weight = radius_graph_custom(pos, self.cutoff, batch)
        edge_attr = self.distance_expansion(edge_weight)
        
        # 2. Embedding
        h = self.embedding(z)
        
        # 3. Interactions
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
            
        # 4. Readout
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)
        
        # Global Pooling (Sum)
        # Using pure torch scatter_add logic
        out = torch.zeros(batch.max() + 1, 1, dtype=h.dtype, device=batch.device)
        out.scatter_add_(0, batch.unsqueeze(1), h)
        
        return out

# --- Training Logic ---

def get_energy_target(batch):
    """Safely extract y_relaxed from batch."""
    if hasattr(batch, 'y_relaxed') and batch.y_relaxed is not None:
        return batch.y_relaxed
    if hasattr(batch, '_store') and 'y_relaxed' in batch._store:
        return batch._store['y_relaxed']
    raise RuntimeError("Batch is missing 'y_relaxed' target. Ensure you are using the training set.")

def train_step(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device, non_blocking=True) # non_blocking for overlap
        optimizer.zero_grad()
        
        # Mixed Precision Context
        with torch.amp.autocast('cuda'):
            # Custom SchNet Forward Pass
            out = model(batch.atomic_numbers.long(), batch.pos, batch.batch)
            
            # Target: y_relaxed (Energy)
            target = get_energy_target(batch)
            if target.dim() == 1:
                target = target.view(-1, 1)
                
            loss = F.l1_loss(out, target) # MAE Loss

        # Scales loss, calls backward() to create scaled gradients
        scaler.scale(loss).backward()
        
        # Unscales gradients and updates optimizer
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_step(model, loader, device):
    model.eval()
    total_mae = 0
    preds = []
    targets = []
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        # Eval doesn't need scaler, but autocast helps speed
        with torch.amp.autocast('cuda'):
            out = model(batch.atomic_numbers.long(), batch.pos, batch.batch)
            
            target = get_energy_target(batch)
            target = target.view(-1, 1)
            
            mae = F.l1_loss(out, target)
        
        total_mae += mae.item() * batch.num_graphs
        
        preds.append(out.float().cpu().numpy()) # Cast to float for numpy
        targets.append(target.float().cpu().numpy())
        
    return total_mae / len(loader.dataset), np.concatenate(preds), np.concatenate(targets)

def train_model(dataset):
    """
    Main training function to be called from the notebook.
    """
    print(f"Using device: {CONFIG['device']}")
    if CONFIG['device'].type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Train/Val Split (80/20)
    # Ensure dataset is shuffled before split if it wasn't already
    # Assuming dataset is a list of Data objects
    
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples.")
    
    # Optimized DataLoaders
    # Note: num_workers>0 requires data to be pickleable. PyG Data objects usually are.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    # 2. Model Initialization (Using Custom SchNet)
    model = CustomSchNet(
        hidden_channels=CONFIG['hidden_channels'],
        num_filters=CONFIG['num_filters'],
        num_interactions=CONFIG['num_interactions'],
        num_gaussians=50,
        cutoff=CONFIG['cutoff']
    ).to(CONFIG['device'])

    # Compile model for speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Compilation failed: {e}. Proceeding without compilation.")
    
    # Optimizer with L2 Regularization (weight_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    # Gradient Scaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop
    history = {'train_loss': [], 'val_mae': []}
    
    print("Starting Training...")
    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_step(model, train_loader, optimizer, CONFIG['device'], scaler)
        val_mae, _, _ = eval_step(model, val_loader, CONFIG['device'])
        
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} eV")

    # 4. Visualization of Results
    val_mae, val_preds, val_targets = eval_step(model, val_loader, CONFIG['device'])
    
    plt.figure(figsize=(10, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Energy MAE (eV)')
    plt.legend()
    plt.title('Training Convergence')
    
    # Parity Plot
    plt.subplot(1, 2, 2)
    plt.scatter(val_targets, val_preds, alpha=0.5, s=10)
    # x=y line
    lims = [np.min([val_targets.min(), val_preds.min()]), np.max([val_targets.max(), val_preds.max()])]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    plt.xlabel('Ground Truth Energy (eV)')
    plt.ylabel('Predicted Energy (eV)')
    plt.title(f'Parity Plot (Final Val MAE: {val_mae:.3f})')
    
    plt.tight_layout()
    plt.show()
    
    return model, history