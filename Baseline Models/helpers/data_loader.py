import torch
from torch import utils
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
class CatalystDataset(Dataset):
    def __init__(self, df, transform=None, pre_transform=None):
        super(CatalystDataset, self).__init__(transform, pre_transform)
        self.df = df

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        x = torch.from_numpy(row['atomic_numbers']).long().view(-1, 1)
        edge_index = torch.from_numpy(row['edge_index']).long()
        pos = torch.from_numpy(row['pos']).float()
        tags = torch.from_numpy(row['tags']).long()
        y = torch.tensor([row['y_relaxed']], dtype=torch.float)

        # Include interatomic distances as edge attributes
        edge_attr = torch.from_numpy(row['distances']).float().view(-1, 1)

        data = Data(
            x=x,
            edge_index=edge_index,
            pos=pos,
            y=y,
            tags=tags,
            sid=row['sid'],
            edge_attr=edge_attr  # Add to the data object
        )
        return data

class AdvancedCatalystDataset(Dataset):
    def __init__(self, df, transform=None, pre_transform=None, normalize_distances=True):
        super(AdvancedCatalystDataset, self).__init__(transform, pre_transform)
        self.df = df
        self.normalize_distances = normalize_distances

        # estimate normalization statistics for distanves
        if normalize_distances:
            all_distances = np.concatenate([row['distances'] for _, row in df.iterrows()])
            self.dist_mean = all_distances.mean()
            self.dist_std = all_distances.std()
            print(f'Distanve stats - Mean: {self.dist_mean:.4f}, std: {self.dist_std:.4f}')

    def len(self):
        return len(self.df)
    
    def get(self, idx):
        row = self.df.iloc[idx]
        # atomic numbers as features
        x = torch.from_numpy(row['atomic_numbers']).long().view(-1,1)
        # edge connections
        edge_index = torch.from_numpy(row['edge_index']).long()
        # atom postitions
        pos = torch.from_numpy(row['pos']).float()
        # tags (surface, bulk, adsorbate)
        tags = torch.tensor(row['tags']).long()
        #target
        y = torch.tensor([row['y_relaxed']], dtype = torch.float)
        #edge distances with normalization
        distances = row['distances']
        if self.normalize_distances:
            distances = (distances - self.dist_mean) /self.dist_std
        edge_attr = torch.from_numpy(distances).float().view(-1, 1)

        # calculate aditional edge features
        edge_vectors = pos[edge_index[1]] - pos[edge_index[0]]
        edge_lengths = torch.norm(edge_vectors, dim=1, keepdim=True)
        edge_unit_vectors = edge_vectors / (edge_lengths + 1e-8)

        # combine edge_features
        edge_attr = torch.cat([
            edge_attr,
            edge_lengths,
            edge_unit_vectors
        ], dim=1)

        # create graph
        data = Data(
            x =x,
            edge_index = edge_index,
            pos = pos, 
            y = y,
            tags = tags,
            sid = row['sid'],
            edge_attr = edge_attr
        )
        return data
