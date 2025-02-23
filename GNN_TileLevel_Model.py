import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GraphConv, global_mean_pool
import torch.distributed as dist
from datetime import timedelta
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset, SubsetRandomSampler
from torch.amp import GradScaler, autocast
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from GNN_model_utils import *
import optuna
import argparse

class GatedAttentionWithInstanceClassifier(nn.Module):
    """
    A variant that computes instance-level logits + an attention weighting,
    then aggregates instance logits to produce a slide-level prediction.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # For gating-based attention
        self.w_h = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w_g = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w_a = nn.Linear(hidden_dim, 1, bias=False)
        
        # Instance-level classifier
        self.instance_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # Per-tile logits
        )

    def forward(self, x, batch=None):
        """
        Args:
            x: [N, in_dim] tile embeddings
            batch: [N], which slide each tile belongs to
        Returns:
            slide_logits: [num_slides_in_batch, num_classes]
            instance_logits: [N, num_classes]
            att_weights: [N], attention weight per tile
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 1) Per-tile (instance) logits
        instance_logits = self.instance_classifier(x)  # shape [N, num_classes]
        
        # 2) Compute gated attention scores
        h = torch.tanh(self.w_h(x))      # [N, hidden_dim]
        g = torch.sigmoid(self.w_g(x))   # [N, hidden_dim]
        alpha_logits = self.w_a(h * g).squeeze(-1)  # shape [N]
        
        # 3) Aggregate instance logits -> slide logits
        unique_graph_ids = batch.unique()
        slide_logits_list = []
        att_weights_list = torch.zeros_like(alpha_logits)

        for gid in unique_graph_ids:
            mask = (batch == gid)
            alpha_g = alpha_logits[mask]         # shape [n_tiles_in_slide]
            inst_logits_g = instance_logits[mask]  # [n_tiles_in_slide, num_classes]
            
            # Normalize attention weights within this slide
            alpha_g_softmax = F.softmax(alpha_g, dim=0)
            alpha_g_softmax = alpha_g_softmax.to(att_weights_list.dtype)  
            att_weights_list[mask] = alpha_g_softmax

            # Weighted sum of instance logits => slide-level logit
            slide_logit = torch.sum(alpha_g_softmax.unsqueeze(1) * inst_logits_g, dim=0)
            slide_logits_list.append(slide_logit.unsqueeze(0))

        slide_logits = torch.cat(slide_logits_list, dim=0)  # [num_slides_in_batch, num_classes]
        
        return slide_logits, instance_logits, att_weights_list


# The GNN Model
class SlideGNN_AdvancedMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, num_classes=2,num_layers=3,dropout=0.3):
        super().__init__()

        # Initialize a ModuleList to hold your GraphConv layers
        self.convs = nn.ModuleList()

        # Add the first layer (input layer)
        self.convs.append(GraphConv(input_dim, hidden_dim))

        # Add intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        # Add the last layer
        self.convs.append(GraphConv(hidden_dim, hidden_dim))

        # Store dropout
        self.dropout = dropout

        # Batch normalization layers
        self.bns = torch.nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # Attention-based pooling for slide-level embedding
        self.mil_pool = GatedAttentionWithInstanceClassifier(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

        # Final classification from the pooled slide-level embedding
        self.classifier = nn.Linear(hidden_dim, num_classes)


    def forward(self, x, edge_index, batch):
        """
        x: Node features [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges] 
        batch: Node-to-graph assignment [num_nodes]
        """

        # Pass through each graph convolutional layer
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # MIL aggregator
        slide_logits, tile_logits, att_weights = self.mil_pool(x, batch)

        return slide_logits, tile_logits, att_weights


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    scaler = GradScaler()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass (in mixed-precision)
            slide_logits, _, _= model(data.x, data.edge_index, data.batch)
            
            # Suppose we only have slide-level labels
            loss = criterion(slide_logits, data.y)

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()
        
        # Step with the optimizer using the scaler
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
    return train_loss / len(train_loader)
   

def test(model,test_loader,device):
    model.eval()
    slide_preds, slide_labels, slide_score, tile_preds, tile_score, tile_batch = [],[], [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            slide_logits, tile_logits, att_weights = model(data.x, data.edge_index, data.batch)
            
            # ---------- Slide-level ----------
            # Slide-level predictions
            s_preds = slide_logits.argmax(dim=1)
            s_score = F.softmax(slide_logits, dim=1)
            slide_preds.extend(s_preds.cpu().numpy())
            slide_score.extend(s_score.cpu().numpy())
            slide_labels.extend(data.y.cpu().numpy())

            # ---------- Tile-level ----------
            # Convert tile_logits to probabilities
            t_preds = tile_logits.argmax(dim=1)
            t_score = F.softmax(tile_logits, dim=1)

            #t_preds = att_weights.argmax(dim=1)
            tile_preds.append(t_preds.cpu())
            tile_score.append(t_score.cpu())
            tile_batch.append(data.batch.cpu())

    # Concatenate tile-level outputs across all batches
    tile_preds = torch.cat(tile_preds, dim=0).numpy()
    tile_score = torch.cat(tile_score, dim=0).numpy()  # shape [TotalTiles, num_classes]
    tile_batch = torch.cat(tile_batch, dim=0).numpy()    # shape [TotalTiles]

    # Slide-level accuracy
    slide_accuracy = accuracy_score(slide_labels, slide_preds)
    return slide_accuracy, slide_score, slide_preds, slide_labels, tile_score, tile_preds, tile_batch

def test_with_coords(model, loader, device):
    model.eval()
    slide_preds = slide_labels = slide_score = tile_preds = tile_score = tile_coords = tile_batch = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            slide_logits, tile_logits, att_weights = model(data.x, data.edge_index, data.batch)

            # Slide-level predictions
            s_preds = slide_logits.argmax(dim=1)
            s_score = F.softmax(slide_logits, dim=1)
            slide_preds.extend(s_preds.cpu().numpy())
            slide_score.extend(s_score.cpu().numpy())
            slide_labels.extend(data.y.cpu().numpy())

            # Convert tile_logits to probabilities
            t_preds = tile_logits.argmax(dim=1)
            t_score = F.softmax(tile_logits, dim=1)
            
            # Store
            tile_preds.append(t_preds.cpu())
            tile_score.append(t_score.cpu())
            if hasattr(data, 'coords'):
                tile_coords.append(data.coords.cpu())
            tile_batch.append(data.batch.cpu())
    
    # Concatenate
    tile_probs = torch.cat(tile_probs, dim=0).numpy()
    tile_batch = torch.cat(all_batch, dim=0).numpy()
    if all_coords:
        tile_coords = torch.cat(tile_coords, dim=0).numpy()

    # Slide-level accuracy
    slide_accuracy = accuracy_score(slide_labels, slide_preds)

    return slide_accuracy, slide_score, slide_preds, slide_labels, tile_scores, tile_preds, tile_batch, tile_coords


def run(world_size: int, rank: int, graph_dataset,sample_names,label_encoder):
    # Will query the runtime environment for `MASTER_ADDR` and `MASTER_PORT`.
    # Make sure, those are set!
    os.environ["MASTER_ADDR"] = os.popen(f"scontrol show hostnames {os.environ['SLURM_NODELIST']}").read().splitlines()[0]
    os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_PROCID", 0))))
       
    #print(f"Initializing process group: Rank={rank}, World Size={world_size}, Local Rank={local_rank}")
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    print(f"Process group initialized: Rank={rank}")


    # Move to device for faster feature fetch.
    #data = data.to(local_rank, 'x', 'y')

    folds = stratified_kfold_split(graph_dataset,  n_splits=5)
   
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Create a directory for this fold
        fold_dir = os.path.join(outdir, f"result_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)  # Create folder if it doesn't exist
        if rank == 0:
            print(f"  Training samples: {len(train_idx)}")
            print(f"  Validation samples: {len(val_idx)}")

        # Split dataset
        train_dataset = Subset(graph_dataset, train_idx)
        test_dataset = Subset(graph_dataset, val_idx)

        print(f"Rank={rank}, train_data:{len(train_dataset)}, test_data:{len(test_dataset)}")

        # Use DistributedSampler
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler)

        # Validate data loaders
        if rank == 0:
            print("Checking train loader...")
            check_batches(train_loader)
        if rank == 0:
            print("Checking validation loader...")
            check_batches(test_loader)

        # Initialize model
        model = SlideGNN_AdvancedMIL(input_dim=1024, hidden_dim=512, num_classes=4,num_layers=3, dropout=0.3).to(rank)
        model = DDP(model, device_ids=[rank])

        # Calculate class weights
        class_weights = calculate_class_weights_from_graph(train_dataset, device=rank)

        # Use CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Train and evaluate
        for epoch in range(1, 101):
            train_loss = train(model, train_loader, optimizer, criterion, device=rank)
            dist.barrier()
            slide_accuracy, slide_score, slide_preds, slide_labels, tile_score, tile_preds, tile_batch = test(model,test_loader, device=rank)
            dist.barrier()
            if rank == 0:
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Accuracy:{slide_accuracy:.4f}')


        if world_size > 1:
            dist.all_reduce(slide_score, op=dist.ReduceOp.AVG)
            dist.all_reduce(slide_preds, op=dist.ReduceOp.AVG)
            dist.all_reduce(slide_labels, op=dist.ReduceOp.AVG)
            dist.all_reduce(tile_score, op=dist.ReduceOp.AVG)
            dist.all_reduce(tile_preds, op=dist.ReduceOp.AVG)
            dist.all_reduce(tile_batch, op=dist.ReduceOp.AVG)
        if rank == 0:
            # Save model for this fold
            model_path = os.path.join(fold_dir, "GNN_ViT_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model for Fold {fold_idx + 1} saved at {model_path}")

            # Save results for this fold in a text file
            save_results_to_text(fold_dir, sample_names, val_idx, slide_labels, slide_preds, slide_score, label_encoder)
            
            # Save to an .npz file
            np.savez(os.path.join(fold_dir,"tile_results.npz"),
                     tile_probs=tile_preds,
                     tile_score=tile_score,
                     tile_batch=tile_batch)
            print("Saved test results to tile_results.npz")

        dist.barrier()

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna Search")
    parser.add_argument('--column', type=str, required=True, help='Column name to select from NCI_CBTN_class')
    parser.add_argument('--outdir', type=str, required=False, default=None, help='Directory to save the outputs')
    #parser.add_argument('--SMOTE', type=str, choices=['TRUE', 'FALSE'], required=False, default='FALSE', help='SMOTE for imbalacnce classe')
    args = parser.parse_args()

    # set randome seed
    set_random_seed(seed=42)
    # Convert the string to a boolean
    #use_smote = args.SMOTE == 'TRUE'
    
    # Intiate environment
    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    # Likewise for RANK and 
:
    #rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    #local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

    # Generate the output directory automatically if not provided
    if args.outdir:
        outdir=args.outdir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outdir = f"./MLP_Bagging_model_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    print(f"loading data!")

    ## load NCI CBTN Group 3 and 4 features and classes
    slide_features=np.load("../features/NCI_CBTN_GP34_features.npy", allow_pickle=True)
    slide_classes=pd.read_csv("../features/NCI_CBTN_GP34_Malignant&Myeloid_Cellstate_classes.csv",index_col=0)

    

    # Seperate sample names and image tile level features
    sample_names, features = extract_sample_names_and_features(slide_features)

    # Scale the features
    scaled_features = scale_tile_features(features)

    # Encode the target classes
    slide_classes=slide_classes[args.column]
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(slide_classes)

    # creating graph data from tile level featutes
    graph_dataset = TileGraphDataset(scaled_features, y_labels, k=5)

    print(len(graph_dataset))
    print(graph_dataset[0])
    
    
    #run(world_size, rank, local_rank, graph_dataset, sample_names,label_encoder)
    mp.spawn(run, args=(world_size, graph_dataset, sample_names,label_encoder), nprocs=world_size, join=True)

    # Collect data
    #true_labels, predicted_labels, predicted_scores = collect_and_combine_results(args.outdir)

    # Annalyze data
    #analyze_results(true_labels, predicted_labels, predicted_scores, label_encoder,outdir=args.outdir)
   
    print("All GNN and ViT hybride modle 5 fold cross validation process is Done...!")
    
