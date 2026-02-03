import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader


def plot_training_curves(train_maes, val_maes):
    """
    Plots the Training and Validation MAE over epochs.
    Shows the learning progress and potential overfitting.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_maes) + 1)
    
    plt.plot(epochs, train_maes, 'o-', label='Train MAE', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_maes, 's-', label='Validation MAE', color='#ff7f0e', linewidth=2)
    
    plt.title('Learning curves - Mean Absolute Error over Epochs', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MAE (electron volts eV)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def get_predictions_and_targets(model, loader, device, predict_fn, mean_energy=-1.54):
    """
    Helper function to collect all predictions and targets for a dataset.
    Takes the prediction function as an argument to avoid local module dependencies.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # The user provides the prediction logic (predict_fn) as a parameter
            out = predict_fn(model, batch)
            
            # De-normalize: Add back the mean energy subtracted during training
            p = out.view(-1).cpu().numpy() + mean_energy
            t = batch.y_relaxed.view(-1).cpu().numpy()
            
            all_preds.extend(p.tolist())
            all_targets.extend(t.tolist())
            
    return np.array(all_preds), np.array(all_targets)


def plot_parity_results(preds, targets, title="Parity Plot"):
    """
    Generates a Parity Plot (Predicted vs Ground Truth).
    Ideally, all points should lie on the diagonal red line.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.4, color='teal', s=20, label='Predicted Systems')
    
    # Plot ideal x=y line
    mn = min(targets.min(), preds.min())
    mx = max(targets.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal Prediction (x=y)')
    
    mae = np.mean(np.abs(preds - targets))
    
    plt.title(f'{title}\nMean Absolute Error: {mae:.4f} eV', fontsize=14)
    plt.xlabel('Ground Truth Relaxed Energy (eV)', fontsize=12)
    plt.ylabel('Model Predicted Energy (eV)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_error_distribution(preds, targets):
    """
    Visualizes the distribution of absolute errors.
    Helps identify if the model has bias or many large outliers.
    """
    abs_errors = np.abs(preds - targets)
    
    plt.figure(figsize=(10, 6))
    plt.hist(abs_errors, bins=40, color='crimson', edgecolor='black', alpha=0.7)
    
    plt.axvline(np.mean(abs_errors), color='blue', linestyle='--', label=f'MAE: {np.mean(abs_errors):.4f}')
    plt.axvline(np.median(abs_errors), color='green', linestyle='-', label=f'Median: {np.median(abs_errors):.4f}')
    
    plt.title('Distribution of Absolute Errors', fontsize=14)
    plt.xlabel('Absolute Error |Pred - Truth| (eV)', fontsize=12)
    plt.ylabel('Number of Systems', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def summarize_model_metrics(preds, targets, set_name="Validation"):
    """
    Prints a detailed numerical summary of the model performance based on raw outputs.
    """
    errors = preds - targets
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    r2 = 1 - (np.sum(errors**2) / np.sum((targets - np.mean(targets))**2))
    
    print(f"\n--- Model Performance Summary ({set_name}) ---")
    print(f"Total Samples:      {len(targets)}")
    print(f"Mean Absolute Error (MAE):  {mae:.4f} eV")
    print(f"Root Mean Sq. Error (RMSE): {rmse:.4f} eV")
    print(f"R-squared (R²):             {r2:.4f}")
    print(f"Median Absolute Error:      {np.median(abs_errors):.4f} eV")
    print(f"90th Percentile Error:      {np.percentile(abs_errors, 90):.4f} eV")
    print("-" * 45)