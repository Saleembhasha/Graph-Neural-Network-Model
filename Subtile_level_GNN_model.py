import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale


# 1. Sub-Tile GNN
class SubTileGNN(nn.Module):
    """
    GNN to aggregate sub-tile (patch) features into a single tile embedding.
    Each tile is represented as a small graph whose nodes are sub-tiles.
    """
    def __init__(self, in_dim=1024, hidden_dim=256, out_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x, edge_index, batch=None):
        """
        x         : [N, in_dim]         (N sub-tiles across the batch)
        edge_index: [2, E] in COO format
        batch     : [N], batch indices if processing multiple tiles at once
        
        Returns: [batch_size, out_dim] tile embeddings, \\\
                 or [1, out_dim] if processing a single tile at a time.
        """
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling: aggregates sub-tiles in each tile
        # If batch is not None, we can handle multiple tiles together
        if batch is not None:
            x = global_mean_pool(x, batch)  # shape: [batch_size, hidden_dim]
        else:
            x = x.mean(dim=0, keepdim=True) # shape: [1, hidden_dim] if single tile

        # Map to final tile-embedding dimension
        tile_emb = self.fc_out(x)  # shape: [batch_size, out_dim]
        return tile_emb

# 2. Tile-Level MIL Aggregator
class TileMILAggregator(nn.Module):
    """
    Aggregates all tile embeddings in a slide into a single slide-level embedding,
    then outputs a slide-level classification.
    """
    def __init__(self, tile_dim=256, hidden_dim=256, num_classes=2):
        super().__init__()
        # Encodes tile embeddings into a latent space for attention
        self.encoder = nn.Sequential(
            nn.Linear(tile_dim, hidden_dim),
            nn.ReLU()
        )
        # Attention scoring function
        self.attn_fc = nn.Linear(hidden_dim, 1)
        # Final classification from slide embedding
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, tile_embs):
        """
        tile_embs: [n_tiles, tile_dim] for a single slide
        Returns:
          - logits: [1, num_classes], slide-level prediction
          - attn_weights: [n_tiles, 1], attention per tile
        """
        # Encode each tile embedding
        z = self.encoder(tile_embs)         # [n_tiles, hidden_dim]

        # Compute scalar attention scores
        attn_scores = self.attn_fc(z)       # [n_tiles, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [n_tiles, 1]

        # Weighted sum -> single slide-level embedding
        slide_emb = (attn_weights * z).sum(dim=0)  # [hidden_dim]

        # Classify at the slide level
        slide_emb = slide_emb.unsqueeze(0)          # [1, hidden_dim]
        logits = self.classifier(slide_emb)         # [1, num_classes]

        return logits, attn_weights


# 3. Full Hierarchical Model
class HierarchicalMILGNN(nn.Module):
    """
    Complete end-to-end model:
      - Sub-tile GNN for each tile
      - Tile-level MIL aggregator for slide classification
    Trained with only slide-level labels (weak supervision).
    """
    def __init__(self, 
                 subtile_in_dim=1024, 
                 subtile_hidden_dim=256, 
                 subtile_out_dim=256,
                 tile_hidden_dim=256, 
                 num_classes=2):
        super().__init__()
        
        # Stage 1: GNN for sub-tiles -> single tile embedding
        self.subtile_gnn = SubTileGNN(
            in_dim=subtile_in_dim,
            hidden_dim=subtile_hidden_dim,
            out_dim=subtile_out_dim
        )
        
        # Stage 2: MIL aggregator across tile embeddings -> slide prediction
        self.tile_aggregator = TileMILAggregator(
            tile_dim=subtile_out_dim,
            hidden_dim=tile_hidden_dim,
            num_classes=num_classes
        )
    
    def forward(self, slide_data):
        """
        slide_data: a structure that holds all sub-tile graphs for each tile in the slide.
                    For example, slide_data.tiles is a list of length n_tiles, 
                    each element containing:
                       tile['x'] -> sub-tile features [n_subtiles, 1024]
                       tile['edge_index'] -> adjacency [2, E]
                       (optional) tile['batch'] -> [n_subtiles] if using PyG batching
                    slide_data.label -> the slide-level label (if used internally)
        
        Returns:
          logits: [1, num_classes] for the slide
          attn_weights: [n_tiles, 1] indicating importance of each tile
          tile_embs: [n_tiles, tile_dim] the learned tile embeddings
        """
        tile_embs = []
        
        # 1) Convert each tile from sub-tile graph -> tile embedding
        for tile in slide_data.tiles:
            x_subtile = tile['x']             # shape [n_subtiles, 1024]
            edge_index_subtile = tile['edge_index']
            # If you want to do them one by one, typically no 'batch' is needed 
            # unless you're combining multiple tiles in one shot.
            emb = self.subtile_gnn(x_subtile, edge_index_subtile)
            # emb shape: [1, subtile_out_dim]
            tile_embs.append(emb.squeeze(0))
        
        tile_embs = torch.stack(tile_embs, dim=0)  # [n_tiles, subtile_out_dim]
        
        # 2) Aggregate tile embeddings via MIL attention -> slide logits
        logits, attn_weights = self.tile_aggregator(tile_embs)
        
        return logits, attn_weights, tile_embs


## creaing graph data from subtile level features
def combined_distance_matrix(coords, features, alpha=0.5):
    """
    coords  : np.array [n_subtiles, 2] (x, y) for spatial positions
    features: np.array [n_subtiles, d] for sub-tile feature vectors
    alpha   : float, weight for spatial distance vs. feature distance
              (0.5 => equal weighting)
    
    Returns:
        dist_matrix: np.array [n_subtiles, n_subtiles], combined distances
    """
    n = coords.shape[0]
    
    # 1) Spatial distance matrix
    #    E.g., Euclidean over (x, y)
    diff_coords = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_space = np.sqrt(np.sum(diff_coords**2, axis=-1))  # shape [n, n]
    
    # 2) Feature distance matrix
    #    E.g., Euclidean over 1024-d
    diff_features = features[:, np.newaxis, :] - features[np.newaxis, :, :]
    dist_feat = np.sqrt(np.sum(diff_features**2, axis=-1))  # shape [n, n]
    
    # 3) (Optional) scale each distance matrix to [0,1] 
    dist_space_scaled = minmax_scale(dist_space.ravel()).reshape(n, n)
    dist_feat_scaled  = minmax_scale(dist_feat.ravel()).reshape(n, n)
    
    # 4) Combine distances
    dist_combined = alpha * dist_space_scaled + (1 - alpha) * dist_feat_scaled
    return dist_combined

def build_edges_combined_knn(coords, features, k=5, alpha=0.5):
    """
    Returns:
      edge_index: np.array of shape [2, E],
                  edges based on k-NN in combined distance space
    """
    n_subtiles = coords.shape[0]
    dist_matrix = combined_distance_matrix(coords, features, alpha=alpha)
    
    edges = []
    for i in range(n_subtiles):
        # Sort neighbors by distance
        nearest_indices = np.argsort(dist_matrix[i])[:k+1]  # +1 to account for self
        for j in nearest_indices:
            if i != j:
                edges.append([i, j])  # i -> j
                edges.append([j, i])  # j -> i (undirected)
    
    # Convert to np.array of shape [2, E]
    edges = np.array(edges).T
    return edges

def create_tile_graph(subtile_features, edge_index):
    """
    Convert sub-tile info into a PyG Data object for one tile.
    """
    # Convert to torch tensors
    x = torch.tensor(subtile_features, dtype=torch.float)    # [n_subtiles, 1024]
    edge_idx = torch.tensor(edge_index, dtype=torch.long)    # [2, E]
    
    data = Data(x=x, edge_index=edge_idx)
    return data

def build_slide_data(slide):
    """
    slide: some structure containing:
       - tile-level sub-tile features, coords
       - slide-level label
    """
    tiles = []
    for tile_info in slide['tiles']:
        # tile_info might have:
        #   subtile_features -> np.array [n_subtiles, 1024]
        #   subtile_coords   -> np.array [n_subtiles, 2]
        
        # 1) Build adjacency
        edge_index = build_edges_combined_knn(tile_info['subtile_coords'],tile_info['subtile_features'], k=5, alpha=0.5)
        
        # 2) Create PyG Data for the tile
        tile_data = create_tile_graph(tile_info['subtile_features'], edge_index)
        
        # We'll store it in a simple dict that the Hierarchical model expects
        tile_dict = {
            'x': tile_data.x,  
            'edge_index': tile_data.edge_index
        }
        
        tiles.append(tile_dict)
    
    # slide_label can be stored as a torch tensor
    slide_label = torch.tensor(slide['label'], dtype=torch.long)
    
    # Return a dictionary describing the entire slide
    slide_data = {
        'tiles': tiles,        # list of tile dicts
        'label': slide_label   # slide-level label
    }
    return slide_data

# 4. Dataset and DataLoader Organization
class SlideDataset(Dataset):
    def __init__(self, raw_slides):
        # raw_slides: a list of slide structures with tile info + label
        self.slide_data_list = []
        for slide in raw_slides:
            slide_data = build_slide_data(slide)
            self.slide_data_list.append(slide_data)
    
    def __len__(self):
        return len(self.slide_data_list)
    
    def __getitem__(self, idx):
        return self.slide_data_list[idx]


