import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import copy

# setting up Data loader

def create_dataloader(train_df, val_df=None, target_col='y_relaxed', batch_size=32, val_split=0.1):
    '''Creates PyTorch Geometric DataLoaders for training a validation'''
    
    # if there is no validation dataset, split the training data
    if val_df is None and val_split>0:
        # calulate each dataset size
        val_size = int(val_split*len(train_df))
        train_size = len(train_df) - val_size

        # split dataset

        train_dataset, val_dataset = torch.utils.data.random_split(train_df, [train_size, val_size])

        print(f'Split dataset into {train_size} for training and {val_size} for validation')

    else:
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
    )

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle=True
        )
    
    return train_loader, val_loader

'''Setting up GNN'''

class GNN(torch.nn.Module):
    '''Graph Neural Network'''
    def __init__(self, node_features=100, hidden_channels=64, num_conv_layers=3):
        super(GNN, self).__init__()

        # node embedding
        self.node_embedding = nn.Linear(node_features, hidden_channels)

        # Convolutional layers
        self.conv_layers = nn.ModuleList()

        for _ in range(num_conv_layers):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        # readour layers (used to predict properties)
        self.readout_layers = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear( hidden_channels//2, 1),

        )
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # initial node embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        # apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)

        # pooling
        x = global_mean_pool(x, batch)
        # final prediction
        energy_pred = self.readout_layers(x)

        return energy_pred

def train_model(model, train_loader, val_loader, num_epochs = 100, lr = 0.001, weight_decay = 1e-5,
                patience = 10 , device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    print(f'Training on {device}')

    # initialize optimizer

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay= weight_decay)

    # loss function
    criterion = nn.MSELoss()

    # training history

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
    }
    
    # early stopping variables
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    no_imporve_epochs = 0


    # early stopping
    start_time= time.time()
    for epoch in range(num_epochs):
        # training phase
        model.train()
        train_loss = 0.0

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data.y)

            # backward
            loss.backward()
            optimizer.step()

            # update info
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            

            # validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)

                    # Forward
                    output = model(data)
                    
                    # compute loss
                    loss = criterion(output, data.y)

                    #Update statistics
                    val_loss += loss.item() * data.num_graphs

            # Calculate average validation loss and MAE

            val_loss /= len(val_loader.dataset)
            val_mae = mean_absolute_error(val_targets, val_preds)
            val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)


            # print progress
            print(f'Epoch {epoch +1}/{num_epochs} | Train Loss: {train_loss:.4f}'
                  f'Val Loss: {val_loss:4.f} | Val MAE: {val_mae:4.f} | Val RMSE: {val_rmse:.4f}')
                
            # check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break 
    # trainnign complete
    time_elapsed = time.time() - start_time
    print(f'Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best validation loss: {best_val_loss:.4f}')

    # load best model weights

    model.load_state_dict(best_model_wts)

    return model, history


