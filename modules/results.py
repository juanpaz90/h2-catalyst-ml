import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(train_maes, val_maes):
    """
    Plots the Training and Validation MAE over epochs.
    Shows the learning progress and potential overfitting.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_maes) + 1)
    
    plt.plot(epochs, train_maes, 'o-', label='Train MAE', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_maes, 's-', label='Validation MAE', color='#ff7f0e', linewidth=2)
    
    plt.title('Learning curves - Mean Absolute Error over Epochs', fontsize=14, fontweight="bold")
    plt.xlabel('Epochs', fontsize=12, fontweight="bold")
    plt.ylabel('MAE (electron volts eV)', fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    

def plot_parity_results(preds, targets):
    """
    Generates a Parity Plot (Predicted vs Real).
    Ideally, all points should lie on the diagonal red line.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.4, color='teal', s=20, label='Predicted Systems')
    
    # Plot ideal x=y line
    mn = min(targets.min(), preds.min())
    mx = max(targets.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Ideal Prediction (x=y)')
    
    mae = np.mean(np.abs(preds - targets))
    
    plt.title(f'Parity Plot - Mean Absolute Error: {mae:.4f} eV', fontsize=14, fontweight="bold")
    plt.xlabel('Real Relaxed Energy (eV)', fontsize=12, fontweight="bold")
    plt.ylabel('Model Predicted Energy (eV)', fontsize=12, fontweight="bold")
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
    
    plt.title('Distribution of Absolute Errors', fontsize=14, fontweight="bold")
    plt.xlabel('Absolute Error |Pred - Truth| (eV)', fontsize=12, fontweight="bold")
    plt.ylabel('Number of Systems', fontsize=12, fontweight="bold")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def summarize_model_metrics(preds, targets, model):
    """
    Prints a detailed numerical summary of the model performance based on raw outputs.
    """
    errors = preds - targets
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors**2))
    r2 = 1 - (np.sum(errors**2) / np.sum((targets - np.mean(targets))**2))
    
    print(f"\n>> Performance Summary - Validation <<")
    print(f"Model: {model}\n")
    print("-" * 45)
    print(f"Total Samples:              {len(targets)}")
    print(f"Mean Absolute Error (MAE):  {mae:.4f} eV")
    # print(f"Root Mean Sq. Error (RMSE): {rmse:.4f} eV")                       ----> NOT USEFUL
    # print(f"R-squared (R²):             {r2:.4f}")                            ----> NOT USEFUL
    # print(f"Median Absolute Error:      {np.median(abs_errors):.4f} eV")      ----> NOT USEFUL
    # print(f"90th Percentile Error:      {np.percentile(abs_errors, 90):.4f} eV")
    print("-" * 45)