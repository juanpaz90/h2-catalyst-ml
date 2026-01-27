import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATv2Conv, GINConv, global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm, MessagePassing
)
from torch_geometric.utils import add_self_loops, softmax
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
import lmdb
from tqdm import tqdm
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

class GIN_GNN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6,dropout=.15):
        super(GIN_GNN, self).__init__()

        # define embeddings
        self.atom_embedding = nn.Embedding(100, hidden_dim)
        self.tag_embedding = nn.Embedding(4, 64)
        # position encoding: Transforms the position from a 3 dimentional space into a suited dimention for the model.
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, hidden_dim // 2)
        )

        # input projection to allow the model to properly integrate all the embeddings.
        self.input_proj = nn.Linear(hidden_dim + 64 + hidden_dim // 2, hidden_dim)

        # edge feature projection (5D: distance + length + unit_vector)
        self.edge_proj = nn.Linear(5, 128)

        # Main GNN layers with different architectures
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.residual_weights = nn.ParameterList()

        for i in range(num_layers):
            
            
            # GIN layers
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            layer = GINConv(gin_nn, train_eps=True)
            
        
            self.gnn_layers.append(layer)
            self.layer_norms.append(LayerNorm(hidden_dim))
            self.residual_weights.append(nn.Parameter(torch.tensor(0.5)))
        # Multi-scale pooling

        self.pool_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim // 4)
            for _ in range(3)
        ])

        # calculate correct input dimension fo routput projection
        # final_mean + final_max + final_add = 3 * hidden_dim
        # 3 intermediate features with pooling = 3 * (Hidden_dim // 4)
        output_input_dim = 3 * hidden_dim + 3 * (hidden_dim // 4)
        # output projectoin with deeper network
        self.output_proj = nn.Sequential(
            nn.Linear(output_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # define forward function
    def forward(self, batch):
        # prepare features
        # atom embedding
        atom_emb = self.atom_embedding(batch.x.squeeze(-1))
        tag_emb = self.tag_embedding(batch.tags)
        pos_enc = self.pos_encoder(batch.pos)

        # combinefeatures
        x = torch.cat([atom_emb, tag_emb, pos_enc], dim=-1)
        x = self.input_proj(x)

        # project edge features
        edge_attr = self.edge_proj(batch.edge_attr)
        # apply GNN layers with residual connections
        intermediate_features = []
        for i, (layer, norm, alpha) in enumerate(
            zip(self.gnn_layers, self.layer_norms, self.residual_weights)
        ):
            residual = x

            # apply layer based on type 
            x = layer(x, batch.edge_index)
            
            x = norm(x)
            x = F.silu(x)
            x = self.dropout(x)

            # learnable residual connections
            x = alpha * x + (1- alpha) * residual

            # store intermediatefeature for multiscale aggregations
            if i in [1, 3, 5]:
                intermediate_features.append(x)

        # multi-scale pooling
        pooled_features = []
        for i , (feat, pool_layer) in enumerate(zip(intermediate_features, self.pool_layers)):
            # apply pooling to porhjected features 
            projected = pool_layer(feat)
            mean_pool = global_mean_pool(projected, batch.batch)
            pooled_features.append(mean_pool)
        
        # final pooling of last layer
        final_mean = global_mean_pool(x, batch.batch)
        final_max = global_max_pool(x, batch.batch)
        final_add = global_add_pool(x, batch.batch)

        # combine pooled features
        combined = torch.cat([final_mean, final_max, final_add] + pooled_features, dim=1)
        #output projections
        energy = self.output_proj(combined)
        return energy.squeeze(-1)


