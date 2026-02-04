import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import numpy as np

# --- 1. Data Handling & Compatibility Fixes ---

def fix_pyg_data(data_obj):
    """Ensures legacy OCP Data objects are compatible with modern PyG."""
    from torch_geometric.data import Data
    if hasattr(data_obj, '_store'):
        d = dict(data_obj._store)
    elif hasattr(data_obj, '__dict__'):
        d = data_obj.__dict__
    else:
        d = data_obj

    attrs = {}
    for k, v in d.items():
        if k in ['_store', '__parameters__', '_edge_index', '_pos', '_face']: 
            continue
        if isinstance(v, (np.ndarray, list, torch.Tensor)):
            t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            t = t.long() if k in ['atomic_numbers', 'edge_index', 'natoms', 'tags'] else t.float()
            if k in ['atomic_numbers', 'tags'] and t.dim() > 1: t = t.squeeze()
            attrs[k] = t
        else:
            attrs[k] = v
    return Data(**attrs)

class OCPDataset(torch.utils.data.Dataset):
    """Wrapper for IS2RE datasets."""
    def __init__(self, raw_dataset):
        self.dataset = raw_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return fix_pyg_data(self.dataset[idx])

# --- 2. PaiNN Model Components ---

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=6.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class PaiNNMessage(nn.Module):
    """Equivariant Message Passing Block."""
    def __init__(self, hidden_channels, num_gaussians):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.scalar_lin = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, 3 * hidden_channels))
        self.filter_lin = Linear(num_gaussians, 3 * hidden_channels)

    def forward(self, s, v, edge_index, edge_dist, edge_vector):
        src, dst = edge_index
        w = self.filter_lin(edge_dist)
        s_src = self.scalar_lin(s)[src]
        gate = s_src * w
        split_gate = torch.split(gate, self.hidden_channels, dim=-1)
        
        ds = split_gate[0]
        dv = v[src] * split_gate[1].unsqueeze(1) + edge_vector.unsqueeze(2) * split_gate[2].unsqueeze(1)
             
        s_out = torch.zeros_like(s); v_out = torch.zeros_like(v)
        s_out.index_add_(0, dst, ds); v_out.index_add_(0, dst, dv)
        return s_out, v_out

class PaiNNUpdate(nn.Module):
    """Equivariant Update Block."""
    def __init__(self, hidden_channels):
        super().__init__()
        self.vec_lin = Linear(hidden_channels, 2 * hidden_channels, bias=False)
        self.update_lin = Sequential(Linear(2 * hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, 3 * hidden_channels))

    def forward(self, s, v):
        v_transformed = self.vec_lin(v)
        V_1, V_2 = torch.split(v_transformed, s.size(-1), dim=-1)
        v_norm = torch.norm(V_1, dim=1)
        
        u = self.update_lin(torch.cat([s, v_norm], dim=-1))
        a, b, c = torch.split(u, s.size(-1), dim=-1)
        
        s_new = s + a
        v_new = v * b.unsqueeze(1) + V_2 * c.unsqueeze(1)
        return s_new, v_new

class PaiNN(nn.Module):
    def __init__(self, hidden_channels=128, num_interactions=3, num_gaussians=50, cutoff=6.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList([PaiNNMessage(hidden_channels, num_gaussians) for _ in range(num_interactions)])
        self.updates = nn.ModuleList([PaiNNUpdate(hidden_channels) for _ in range(num_interactions)])
        self.readout = Sequential(Linear(hidden_channels, hidden_channels // 2), ReLU(), Linear(hidden_channels // 2, 1))

    def forward(self, batch):
        s = self.embedding(batch.atomic_numbers)
        v = torch.zeros(s.size(0), 3, self.hidden_channels, device=s.device)
        edge_index, pos = batch.edge_index, batch.pos
        dist_vec = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(dist_vec, dim=-1)
        dist_vec_norm = dist_vec / (dist.unsqueeze(-1) + 1e-8)
        edge_rbf = self.distance_expansion(dist)

        for i in range(len(self.interactions)):
            ds, dv = self.interactions[i](s, v, edge_index, edge_rbf, dist_vec_norm)
            s, v = s + ds, v + dv
            s, v = self.updates[i](s, v)
            
        return global_mean_pool(self.readout(s), batch.batch)

# --- 3. Training & Validation Logic ---

def train_one_epoch(model, loader, optimizer, criterion, device, mean_target):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="PaiNN Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.view(-1), batch.y_relaxed.view(-1) - mean_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device, mean_target):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.view(-1), batch.y_relaxed.view(-1) - mean_target)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def get_final_metrics(model, loader, device, mean_target):
    """Calculates final preds and targets for visualization."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.extend((out.view(-1) + mean_target).cpu().numpy())
            targets.extend(batch.y_relaxed.view(-1).cpu().numpy())
    return np.array(preds), np.array(targets)

def configure_and_run_painn_training(train_data, val_data, epochs=10, batch_size=32, lr=0.0005):
    """
    Main Entry Point for PaiNN Training.
    Returns model, training history, and final evaluation data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing PaiNN on {device}...")

    train_loader = DataLoader(OCPDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OCPDataset(val_data), batch_size=batch_size)

    model = PaiNN(hidden_channels=128, num_interactions=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    MEAN_TARGET = -1.54
    train_maes, val_maes = [], []

    for epoch in range(epochs):
        t_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, MEAN_TARGET)
        v_mae = validate(model, val_loader, criterion, device, MEAN_TARGET)
        train_maes.append(t_mae); val_maes.append(v_mae)
        scheduler.step(v_mae)
        print(f"Epoch {epoch+1} | Train MAE: {t_mae:.4f} | Val MAE: {v_mae:.4f}")

    preds, targets = get_final_metrics(model, val_loader, device, MEAN_TARGET)
    return model, train_maes, val_maes, val_loader, device, preds, targets