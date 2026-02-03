import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet, global_mean_pool
from tqdm import tqdm
import numpy as np

# Re-using the fix from Phase 1 to ensure runtime compatibility with older OCP objects
def fix_pyg_data(data_obj):
    from torch_geometric.data import Data
    
    # Extract the internal dictionary regardless of whether it's a legacy Data object or a dict
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
            
            if k in ['atomic_numbers', 'edge_index', 'natoms', 'tags']:
                t = t.long()
            else:
                t = t.float()
            
            if k in ['atomic_numbers', 'tags'] and t.dim() > 1:
                t = t.squeeze()
                
            attrs[k] = t
        else:
            attrs[k] = v

    return Data(**attrs)

class OCPDataset(torch.utils.data.Dataset):
    """
    Modular wrapper to handle pre-loaded data variables and apply PyG version fixes.
    """
    def __init__(self, raw_dataset):
        self.dataset = raw_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return fix_pyg_data(data)

def initialize_model(device, cutoff=6.0):
    """
    Initializes a SchNet GNN.
    """
    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=cutoff,
    ).to(device)
    return model

def get_model_output(model, batch):
    """
    Bypasses SchNet.forward() to avoid the torch-cluster/radius_graph dependency.
    Manually calculates energy using the existing edge_index from the dataset.
    """
    # 1. Get initial embeddings for atomic numbers
    h = model.embedding(batch.atomic_numbers)
    
    # 2. Calculate edge distances manually for the existing edge_index
    edge_index = batch.edge_index
    dist = torch.norm(batch.pos[edge_index[0]] - batch.pos[edge_index[1]], dim=-1)
    edge_attr = model.distance_expansion(dist)
    
    # 3. Pass through interaction blocks
    for interaction in model.interactions:
        h = h + interaction(h, edge_index, dist, edge_attr)
        
    # 4. Final MLP and global pooling to get scalar energy per system
    h = model.lin1(h)
    h = model.act(h)
    h = model.lin2(h)
    
    # Pool the node-level energies into a system-level energy
    return global_mean_pool(h, batch.batch)

def train_one_epoch(model, loader, optimizer, criterion, device, mean_energy=0.0):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training Epoch"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Using the custom output function to bypass forward() and radius_graph
        out = get_model_output(model, batch)
        
        target = batch.y_relaxed.view(-1) - mean_energy
        
        loss = criterion(out.view(-1), target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device, mean_energy=0.0):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # Using the custom output function to bypass forward() and radius_graph
            out = get_model_output(model, batch)
            target = batch.y_relaxed.view(-1) - mean_energy
            loss = criterion(out.view(-1), target)
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def configure_and_run_training(train_data, val_data, epochs=15, batch_size=32, lr=0.001):
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

    train_loader = DataLoader(OCPDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(OCPDataset(val_data), batch_size=batch_size)

    model = initialize_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() 
    
    # Baseline mean target from EDA
    MEAN_TARGET = -1.54 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2
    )

    best_val_mae = float('inf')
    
    # Track metrics for visualization
    train_maes = []
    val_maes = []

    for epoch in range(epochs):
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device, MEAN_TARGET)
        val_mae = validate(model, val_loader, criterion, device, MEAN_TARGET)
        
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        
        scheduler.step(val_mae)
        
        print(f"Epoch {epoch+1:02d} | Train MAE: {train_mae:.4f} eV | Val MAE: {val_mae:.4f} eV")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), 'best_ocp_model.pt')
            print("Checkpoint saved.")

    # Generate final predictions and targets for visualization functions
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = get_model_output(model, batch)
            # De-normalize: Add back the mean energy subtracted during training
            p = out.view(-1).cpu().numpy() + MEAN_TARGET
            t = batch.y_relaxed.view(-1).cpu().numpy()
            preds.extend(p.tolist())
            targets.extend(t.tolist())
    
    preds = np.array(preds)
    targets = np.array(targets)

    return model, train_maes, val_maes, val_loader, device, preds, targets