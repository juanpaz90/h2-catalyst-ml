import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
class IS2RE_GCN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4, dropout=0.15):
        super(IS2RE_GCN, self).__init__()
        self.atom_embedding = nn.Embedding(100, hidden_dim)
        self.tag_embedding = nn.Embedding(3, 32)
        self.input_proj = nn.Linear(hidden_dim + 32, hidden_dim)
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = dropout

    def forward(self, batch):
        # CORRECTED: Changed batch.atomic_numbers to batch.x
        atom_emb = self.atom_embedding(batch.x.squeeze(-1))
        tag_emb = self.tag_embedding(batch.tags)

        x = torch.cat([atom_emb, tag_emb], dim=1)
        x = self.input_proj(x)

        for gcn, bn in zip(self.gcn_layers, self.batch_norms):
            residual = x
            x = gcn(x, batch.edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        x = global_mean_pool(x, batch.batch)
        energy = self.output_proj(x)
        return energy.squeeze(-1)