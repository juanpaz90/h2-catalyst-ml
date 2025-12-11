import os 
import lmdb
from tqdm import tqdm
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from matplotlib.ticker import MaxNLocator
import random

def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy and PyTorch to ensure reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#function to read lmdb data types
def load_lmdb_data(db_path, num_samples=None):
    """
    Loads data from an LMDB database.

    Parameters:
    -----------
    db_path : str
        Path to the LMDB database
    num_samples : int, optional
        Number of samples to load. If None, load all samples.

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the loaded data
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

def train(loader, model, device:str='cpu'):
    """Performs specified epoch of training."""
    global optimizer, criterion
    
    model.to(device)
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", disable=True):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
   
    return total_loss / len(loader.dataset)

def evaluate(loader, model, device: str = 'cpu'):
    """Evaluates the model on a validation or test set.
    Returns MAE, MSE, and R² computed over the full dataset."""
    model.to(device)
    model.eval()

    total_mae = 0.0
    total_mse = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", disable=True):
            batch = batch.to(device)
            preds = model(batch)

            # store predictions and true values for R²
            all_preds.append(preds.detach().cpu())
            all_targets.append(batch.y.detach().cpu())

            # batch-level MAE and MSE
            mae = F.l1_loss(preds, batch.y, reduction='mean')
            mse = F.mse_loss(preds, batch.y, reduction='mean')

            total_mae += mae.item() * batch.num_graphs
            total_mse += mse.item() * batch.num_graphs

    # Concatenate tensors
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute global R²
    r2 = r2_score(all_targets, all_preds)

    # Average MAE and MSE per graph
    avg_mae = total_mae / len(loader.dataset)
    avg_mse = total_mse / len(loader.dataset)

    return avg_mae, avg_mse, r2
    
def train_evaluate_epoch(train_loader, val_loader, device, model, num_epochs:int = 60, 
file_name:str='best_model', learning_rate = 0.001):
    global optimizer, criterion
    history = {'train_loss': [], 'val_mae': [], 'val_mse': [], 'val_r2':[], 'lr':[]}
    print(f'training with lr = {learning_rate}')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_val_mae = float('inf')
    patience_counter = 0
    patience_limit = 15 # Stop after 15 epochs with no improvement
    # ADD a learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print("\nStarting training with scheduler and early stopping...")
    start = time.time()
    for epoch in tqdm(list(range(1, num_epochs + 1)), desc='training_epoch'):
        train_loss = train(train_loader, model, device)
        val_mae, val_mse, val_r2 = evaluate(val_loader, model, device)
        
        
        history['train_loss'].append(train_loss)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)
        history['val_r2'].append(val_r2)


        # Scheduler Step
        scheduler.step(val_mae)
        epoch_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(epoch_lr)
        # Early Stopping Logic
        torch.save(model.state_dict(), f'{file_name}.pt')
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            # print(f"Epoch: {epoch:02d}, Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}")
            # print(f"  -> New best validation MAE: {best_val_mae:.4f}. Model saved.")
            os.rename(f'{file_name}.pt', f'{file_name}_({best_val_mae:.3f}).pt')
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"\nEarly stopping triggered after {epoch} epochs.")

            break
    end = time.time()
    t_time = end - start
    print(f'Elapsed training time {t_time}')
    return history, t_time

def predict_and_visualize(
    model,
    data_loader,
    device: str = 'cuda',
    num_samples: int = 500,
    unit: str = "eV",
    bins: int = 60,
    dpi: int = 130,
    figsize=(11, 9),
    save_path:str  = None
):
    """
    Runs inference and renders a 2x2 diagnostic figure:
      [0,0] Parity plot (MAE in title, y=x dashed)
      [0,1] Residual plot (residuals vs predicted, y=0 dashed)
      [1,0] Residual histogram (density) with sigma in title
      [1,1] Absolute-error percentiles (0..100%)
    Returns (fig, metrics_dict).
    """
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Making predictions", disable=True):
            data = data.to(device)
            output = model(data)
            pred = output.detach().cpu().numpy()
            target = data.y.detach().cpu().numpy()
            all_preds.extend(pred.ravel().tolist())
            all_targets.extend(target.ravel().tolist())

    all_preds = np.asarray(all_preds, dtype=float)
    all_targets = np.asarray(all_targets, dtype=float)

    # --- Metrics ---
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    resid = all_preds - all_targets              # residuals = predicted - true
    abs_err = np.abs(resid)
    sigma = np.std(resid, ddof=1)
    percentiles = np.linspace(0, 100, 101)
    ae_p = np.percentile(abs_err, percentiles)

    # --- Prepare sampled points for scatter density (parity & residual plots) ---
    if len(all_preds) > num_samples:
        idx = np.random.choice(len(all_preds), num_samples, replace=False)
        p_pred, p_true, p_resid = all_preds[idx], all_targets[idx], resid[idx]
    else:
        p_pred, p_true, p_resid = all_preds, all_targets, resid

    # --- Figure and axes ---
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    ax11, ax12, ax21, ax22 = axes.ravel()

    # ===== (1,1) Parity plot =====
    ax11.scatter(p_true, p_pred, s=8, alpha=0.6)
    # limits and y=x
    vmin = min(all_targets.min(), all_preds.min())
    vmax = max(all_targets.max(), all_preds.max())
    pad = 0.03 * (vmax - vmin if vmax > vmin else 1.0)
    lo, hi = vmin - pad, vmax + pad
    ax11.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.2, color='tab:red', label='Perfect prediction')
    ax11.set_xlim(lo, hi); ax11.set_ylim(lo, hi)
    ax11.set_xlabel(f"True Energy ({unit})")
    ax11.set_ylabel(f"Predicted Energy ({unit})")
    ax11.legend(frameon=False, loc="upper left")
    ax11.set_title(f"Parity Plot (MAE: {mae:.4f} {unit})")
    ax11.grid(True, alpha=0.4); ax11.minorticks_on()
    ax11.set_aspect('equal', adjustable='box')

    # ===== (1,2) Residual plot =====
    ax12.scatter(p_pred, p_resid, s=8, alpha=0.6, color='tab:green')
    ax12.axhline(0.0, linestyle='--', linewidth=1.2, color='tab:red')
    ax12.set_xlabel(f"Predicted Energy ({unit})")
    ax12.set_ylabel(f"Residuals ({unit})")
    ax12.set_title("Residual Plot")
    ax12.grid(True, alpha=0.4); ax12.minorticks_on()

    # ===== (2,1) Residual histogram (density) =====
    ax21.hist(resid, bins=bins, density=True, alpha=0.8, color='tab:purple')
    ax21.set_xlabel(f"Residuals ({unit})")
    ax21.set_ylabel("Density")
    ax21.set_title(f"Error Distribution (σ = {sigma:.4f} {unit})")
    ax21.grid(True, alpha=0.3); ax21.minorticks_on()

    # ===== (2,2) Absolute-error percentiles =====
    ax22.plot(percentiles, ae_p, marker='o', markersize=3)
    ax22.set_xlabel("Percentile")
    ax22.set_ylabel(f"Absolute Error ({unit})")
    ax22.set_title("Error Percentiles")
    ax22.grid(True, alpha=0.4); ax22.minorticks_on()
    ax22.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    fig.tight_layout()

    if save_path != None:
        fig.savefig(save_path)


    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "sigma_residuals": float(sigma),
        "percentiles": percentiles,
        "abs_error_percentiles": ae_p,
    }
    return fig, metrics


def plot_graphs_pyg(graphs, max_cols=3, node_size=50):
    """
    Plotea varios objetos Data de PyTorch Geometric en subplots 3D,
    asignando colores discretos a cada elemento químico (número atómico).
    La leyenda muestra los nombres de los elementos, fuera del área de gráficos.
    """
    atomic_symbols = {
    1: 'H',   6: 'C',   7: 'N',   8: 'O',
    11: 'Na', 13: 'Al', 14: 'Si', 19: 'K',
    33: 'As', 39: 'Y',  45: 'Rh', 46: 'Pd',
    49: 'In'
    }
    n = len(graphs)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    # Obtener todos los números atómicos únicos
    all_Z = np.concatenate([g.x[:, 0].cpu().numpy().ravel() for g in graphs])
    unique_Z = np.unique(all_Z)

    # Crear paleta discreta (usando una tabla categórica con colores bien diferenciables)
    base_colors = plt.get_cmap('tab20').colors[:13]  # 10 colores nítidos
    # Si hay más elementos que colores disponibles, repetir el patrón
    color_cycle = [base_colors[i % len(base_colors)] for i in range(len(unique_Z))]

    # Asignar un color fijo a cada elemento
    color_map = {z: color_cycle[i] for i, z in enumerate(unique_Z)}

    fig = plt.figure(figsize=(5 * cols + 2, 4.5 * rows))

    for i, g in enumerate(graphs, start=1):
        pos = g.pos.cpu().numpy()
        edge_index = g.edge_index.cpu().numpy()
        Z = g.x[:, 0].cpu().numpy().ravel()

        ax = fig.add_subplot(rows, cols, i, projection='3d')

        # Colorear nodos según elemento
        node_colors = [color_map[z] for z in Z]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                   c=node_colors, s=node_size, alpha=0.85)

        # Dibujar enlaces
        for u, v in edge_index.T:
            ax.plot(
                [pos[u, 0], pos[v, 0]],
                [pos[u, 1], pos[v, 1]],
                [pos[u, 2], pos[v, 2]],
                color='lightgray', alpha=0.5, linewidth=0.7
            )

        sid = getattr(g, 'sid', f'Estructura {i}')
        ax.set_title(f'Estructura {sid}', pad=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])

    # Crear leyenda discreta con nombres de elementos
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=f"{atomic_symbols.get(int(z), f'Z={int(z)}')}",
                   markerfacecolor=color_map[z], markersize=10)
        for z in unique_Z
    ]

    # Ubicar la leyenda fuera del área de las estructuras
    fig.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        title='Elementos químicos',
        frameon=False
    )

    # Dejar espacio libre a la derecha para la leyenda
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
# plot training_history
def see_training_history(history, save_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    ax = ax.flatten()
    # Plot Training Loss on the primary y-axis
    color = 'tab:blue'
    ax[0].set_xlabel('Epoch', fontsize=12)
    ax[0].set_ylabel('Training Loss (MSE)', fontsize=12)
    ax[0].plot(history['train_loss'], color=color, linestyle='-', label='Train Loss (MSE)')
    ax[0].set_title('Training Loss over Epochs', fontsize=14, pad=20)

    # Create a second axis for the Validation MAE
    color = 'tab:red'
    ax[1].set_ylabel('Validation MAE', fontsize=12)
    ax[1].set_xlabel('Epoch', fontsize=12)
    ax[1].plot(history['val_mae'], color=color, linestyle='-', label='Validation MAE')
    ax[1].set_title('Validation MAE over Epochs', fontsize=14, pad=20)
    secondary_color = 'tab:green'
    best_MAE = [min(history['val_mae']) for _ in range(len(history['val_mae']))]
    ax[1].plot(best_MAE, color=secondary_color, linestyle='--', label='Best Validation MAE')
    ax[1].legend(loc='upper right')
    
    color = 'tab:purple'
    ax[2].set_ylabel('Validation MAE', fontsize=12)
    ax[2].set_xlabel('Epoch', fontsize=12)
    ax[2].plot(np.sqrt(history['val_mse']), color=color, linestyle='-', label='Validation MAE')
    ax[2].set_title('Validation RMSE', fontsize=14, pad=20)

    color = 'tab:orange'
    ax[3].set_ylabel('Learning Rate', fontsize=12)
    ax[3].set_xlabel('Epoch', fontsize=12)
    ax[3].plot(history['lr'], color=color, linestyle='-', label='Learning Rate')
    ax[3].set_title('Learning Rate over Epochs', fontsize=14, pad=20)


    # Final plot adjustments
    fig.tight_layout()
    if save_path != None:
        fig.savefig(save_path)
# Create training curve
import torch
import matplotlib.pyplot as plt

def learning_curve_first_epoch(model, train_loader, val_loader, device:str="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.train()
    running_train_loss = 0.0
    total_samples = 0
    train_curve_x = []
    train_curve_y = []
    val_curve_x = []
    val_curve_y = []

    for batch in tqdm(train_loader, desc="Training", disable=True):
        batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x)
        loss = criterion(preds, batch.y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        total_samples += batch.x.size(0)
        avg_train_loss = running_train_loss / (batch + 1)

        # Record training loss vs number of samples
        train_curve_x.append(total_samples)
        train_curve_y.append(avg_train_loss)

        # Evaluate validation loss at the same points
        model.eval()
        with torch.no_grad():
            val_loss_sum, val_count = 0.0, 0
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vpred = model(vx)
                val_loss_sum += criterion(vpred, vy).item()
                val_count += 1
            avg_val_loss = val_loss_sum / val_count
            val_curve_x.append(total_samples)
            val_curve_y.append(avg_val_loss)
        model.train()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx:03d} | samples {total_samples} | "
                  f"train loss {avg_train_loss:.4f} | val loss {avg_val_loss:.4f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_curve_x, train_curve_y, label="Training loss", marker="o")
    plt.plot(val_curve_x, val_curve_y, label="Validation loss", marker="s")
    plt.xlabel("Number of samples seen")
    plt.ylabel("Loss")
    plt.title("Learning Curve — First Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "train_curve_x": train_curve_x,
        "train_curve_y": train_curve_y,
        "val_curve_x": val_curve_x,
        "val_curve_y": val_curve_y,
    }
