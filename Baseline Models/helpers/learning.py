from torch_geometric.datasets import QM9
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

# Load dataset
dataset = QM9(root='data/QM9')

print(f"Number of molecules: {len(dataset)}")
print(f"First molecule: {dataset[0]}")
print(f"Number of atom features: {dataset.num_features}")

class AtomEmbedder(torch.nn.Module):
    def __init__(self, num_features=11, embedding_dim=64):
        super().__init__()
        # First linear layer creates initial atom embeddings
        self.embed = nn.Linear(num_features, embedding_dim)
        
        # GNN layers refine embeddings using molecular structure
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        self.conv2 = GCNConv(embedding_dim, embedding_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 1. Create initial atom embeddings
        x = self.embed(x)
        x = F.relu(x)
        
        # 2. Refine embeddings using molecular structure
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        return x

# Example usage
model = AtomEmbedder()
molecule = dataset[0]  # Get first molecule
atom_embeddings = model(molecule)
print(f"Input atom features shape: {molecule.x.shape}")
print(f"Output atom embeddings shape: {atom_embeddings.shape}")

class MoleculeEmbedder(torch.nn.Module):
    def __init__(self, num_features=11, embedding_dim=64):
        super().__init__()
        self.atom_embedder = AtomEmbedder(num_features, embedding_dim)
        # Additional layers for molecular properties
        self.fc = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, data):
        # 1. Get atom embeddings
        atom_embeds = self.atom_embedder(data)
        
        # 2. Pool to get molecule embedding (using mean pooling)
        mol_embed = global_mean_pool(atom_embeds, data.batch)
        
        # 3. Final transformation
        mol_embed = self.fc(mol_embed)
        return mol_embed

# Create a batch of molecules
loader = DataLoader(dataset[:32], batch_size=8, shuffle=True)

# Process a batch
for batch in loader:
    mol_embeddings = MoleculeEmbedder()(batch)
    print(f"Molecule embeddings shape: {mol_embeddings.shape}")
    break

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_atom_embeddings(model, dataset):
    # Get embeddings for all atoms in first 100 molecules
    all_embeddings = []
    all_atom_types = []
    
    for mol in dataset[:100]:
        embeddings = model(mol)
        all_embeddings.append(embeddings)
        all_atom_types.extend(mol.x[:,0].tolist())  # Atomic number
    
    all_embeddings = torch.cat(all_embeddings, dim=0).detach().numpy()
    
    # Reduce to 2D with t-SNE
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Plot
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], 
                         c=all_atom_types, alpha=0.6)
    plt.colorbar(scatter, label='Atomic Number')
    plt.title("Atom Embeddings Visualization")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

visualize_atom_embeddings(AtomEmbedder(), dataset)