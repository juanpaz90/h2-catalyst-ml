import os
import torch
from torch_geometric.loader import DataLoader
from modules.SchNet_Base import initialize_base_model, OCPLmdbDataset
from modules.SchNet_MHA import initialize_mha_model


def load_model(model_name, model_path, val_dataset_path, batch_size, **model_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_dataset = OCPLmdbDataset(val_dataset_path)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    if model_name == 'base':
        print(">> Selected model: Base SchNet")
        model = initialize_base_model(
            device, 
            hidden_channels = model_kwargs.get('hidden_channels'),
            num_filters = model_kwargs.get('num_filters'), 
            num_interactions = model_kwargs.get('num_interactions'), 
            num_gaussians = model_kwargs.get('num_gaussians'), 
            cutoff = model_kwargs.get('cutoff')
        )
    elif model_name == 'mha':
        print(">> Selected model: SchNet with Multihead attention (MHA)")
        model = initialize_mha_model(
            device, 
            hidden_channels = model_kwargs.get('hidden_channels'), 
            num_filters = model_kwargs.get('num_filters'), 
            num_interactions = model_kwargs.get('num_interactions'), 
            cutoff = model_kwargs.get('cutoff'), 
            num_gaussians = model_kwargs.get('num_gaussians'), 
            num_heads = model_kwargs.get('num_heads')
        )
    else:
        raise ValueError(f"Unknown model_name: '{model_name}'. Please choose 'base' or 'mha'.")
    
    # 3. Load the trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Running with untrained weights for testing.")
    
    return model, val_loader, device
