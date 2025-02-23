import numpy as np
import pandas as pd
import os
import torch
import random
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset, SubsetRandomSampler
from torch.cuda.amp import GradScaler, autocast
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader,DataListLoader
from torch_geometric.utils import add_self_loops
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from GNN_model_utils import *
import optuna
import argparse

# Set random seed for reproducibility
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if dist.is_initialized():
        torch.cuda.manual_seed(dist.get_rank() + seed)

set_random_seed(seed=42)



class GNN(torch.nn.Module):
    """
    A Graph Neural Network (GNN) for graph classification tasks with tunable number of layers.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layers.
        num_classes (int): Number of output classes.
        num_layers (int): Number of graph convolutional layers.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=3, dropout=0.5):
        super(GNN, self).__init__()
        torch.manual_seed(42)
        self.convs = torch.nn.ModuleList()

        # Add the first layer (input layer)
        self.convs.append(GraphConv(input_dim, hidden_dim))

        # Add intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        # Add the last layer
        self.convs.append(GraphConv(hidden_dim, hidden_dim))

        # Batch normalization layers
        self.bns = torch.nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # Pass through each graph convolutional layer
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout layer
        x = global_mean_pool(x, batch)

        # Final classification layer
        x = self.classifier(x)
        return x

def train_with_early_stopping_ddp(
    model: nn.Module,
    train_loader,
    test_loader,
    device,
    test_fn,             # (model, val_loader, device) -> (local_val_acc, local_preds, local_labels)
    criterion,
    optimizer,
    rank: int,
    world_size: int,
    max_epochs: int = 100,
    patience: int = 10
):
    """
    Multi-GPU (DDP) training loop with early stopping based on global validation accuracy.

    Args:
        model (nn.Module): Your PyTorch model (already wrapped in DDP).
        train_loader (DataLoader): Dataloader for training subset (with DistributedSampler).
        val_loader (DataLoader): Dataloader for validation subset (with DistributedSampler).
        device (torch.device): The local GPU device for this rank.
        test_fn (callable): Evaluates (model, val_loader, device) -> returns
                            (local_val_acc: float, local_preds: list, local_labels: list).
        criterion (nn.Module): Loss function, e.g. CrossEntropyLoss.
        optimizer (torch.optim.Optimizer): e.g. Adam or SGD.
        rank (int): The global rank of the current process.
        world_size (int): Total number of ranks (GPUs).
        max_epochs (int): Maximum training epochs.
        patience (int): Number of epochs to wait for improvement before stopping.
        verbose (bool): If True, rank=0 prints logs.

    Returns:
        best_model_state (dict): state_dict of the best model (stored on rank=0).
        best_global_acc (float): The best global validation accuracy found.
    """
    best_global_acc = 0.0
    patience_counter = 0

    # We'll store the best model weights only on rank=0
    best_model_state = None

    for epoch in range(1, max_epochs + 1):
        # ---------------------- TRAIN -----------------------
        model.train()
        total_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Local average training loss (for reference; not necessarily global)
        train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Ensure all ranks finish training step before validation
        dist.barrier()

        # ---------------------- VALIDATE ---------------------
        # test_fn returns local results for this rank
        if rank == 0:
            local_val_acc, local_preds, local_labels, local_socre = test_fn(model, test_loader, device)

        # Gather local predictions & labels on rank=0
        local_data = (local_preds, local_labels,local_socre)
        # Initialize gather_list on rank 0 with placeholders
        if rank == 0:
            gather_list = [None for _ in range(world_size)]
        else:
            gather_list = None

        dist.gather_object(local_data, gather_list, dst=0)

        # Now rank=0 computes global accuracy from all preds/labels
        if rank == 0:
            global_preds = []
            global_labels = []
            global_score = []
            for gathered in gather_list:
                preds, labels, score = gathered
                global_preds.extend(preds)
                global_labels.extend(labels)
                global_score.append(score)

            # Compute global accuracy
            correct = sum(1 for pr, lab in zip(global_preds, global_labels) if pr == lab)
            total   = len(global_labels)
            global_acc = correct / total if total > 0 else 0.0
            print(f"[Epoch {epoch}/{max_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Global Val Acc: {global_acc:.4f}")

            # -------------- EARLY STOPPING LOGIC --------------
            improved = global_acc > best_global_acc
            if improved:
                best_global_acc = global_acc
                patience_counter = 0

                # Store best weights
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            should_stop = (patience_counter >= patience)

        else:
            # Non-zero ranks don't compute or store anything
            global_acc = 0.0
            improved = False
            should_stop = False

        # Broadcast should_stop
        should_stop_tensor = torch.tensor(int(should_stop), device=device)
        dist.broadcast(should_stop_tensor, src=0)
        should_stop = bool(should_stop_tensor.item())

        # If we want to keep a consistent best_global_acc across ranks,
        # we can broadcast that too:
        best_global_acc_tensor = torch.tensor(best_global_acc, dtype=torch.float, device=device)
        dist.broadcast(best_global_acc_tensor, src=0)
        best_global_acc = best_global_acc_tensor.item()

        # Everyone checks if we should stop
        if should_stop:
            if rank == 0 and verbose:
                print(f"Early stopping at epoch {epoch}. Best Global Acc = {best_global_acc:.4f}")
            break

        # Barrier before the next epoch
        dist.barrier()

    # --------------- END OF TRAINING ----------------
    # Return best model state (only rank=0 has a real copy, others have None)
    return best_model_state, best_global_acc

def test_fn(model,test_loader,device):
    model.eval()
    local_preds, local_labels, local_score = [], [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data.x, data.edge_index, data.batch)
            preds = outputs.argmax(dim=1)
            score = F.softmax(outputs, dim=1)
            local_preds.extend(preds.cpu().numpy())
            local_labels.extend(data.y.cpu().numpy())
            local_score.extend(score.cpu().numpy())
            correct += (preds == data.y).sum().item()
            total += preds.size(0)
    local_val_acc = correct / total if total > 0 else 0.0
    return local_val_acc, local_preds, local_labels, local_score

def run(world_size: int, rank: int, local_rank: int, graph_dataset,sample_names,label_encoder):
    # Will query the runtime environment for `MASTER_ADDR` and `MASTER_PORT`.
    # Make sure, those are set!
    print(f"Initializing process group: Rank={rank}, World Size={world_size}, Local Rank={local_rank}")
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    print(f"Process group initialized: Rank={rank}")

    # Move to device for faster feature fetch.
    #data = data.to(local_rank, 'x', 'y')

    folds = stratified_kfold_split(graph_dataset, n_splits=5)
   
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        
        # Create a directory for this fold
        fold_dir = os.path.join(outdir, f"result_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)  # Create folder if it doesn't exist

        # Split dataset
        train_dataset = Subset(graph_dataset, train_idx)
        test_dataset = Subset(graph_dataset, val_idx)

        print(f"Rank={rank}, len(train_dataset) = {len(train_dataset)}")

        # Use DistributedSampler
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler,drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler,drop_last=True)
        if rank == 0:
            # Validate data loaders
            print("Checking train loader...")
            check_batches(train_loader)
        if rank == 0:
            print("Checking validation loader...")
            check_batches(test_loader)

        # Initialize model
        model = GNN(input_dim=1024, hidden_dim=512, num_classes=4,num_layers=3, dropout=0.3).to(local_rank)
        #model = DDP(model, device_ids=[local_rank])
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        # Calculate class weights
        class_weights = calculate_class_weights_from_graph(train_dataset, device=local_rank)

        # Use CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(local_rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # 5) Call the training loop function
        best_model_state, best_global_acc = train_with_early_stopping_ddp(
            model=model,train_loader=train_loader, test_loader=test_loader,
            device=local_rank, test_fn=test_fn, criterion=criterion, optimizer=optimizer, 
            rank=rank, world_size=world_size, max_epochs=100, patience=10)
        dist.barrier()
        #  Save best model (only on rank=0)
        if rank == 0:
            model_path = os.path.join(fold_dir, "GNN_ViT_model.pth")
            if best_model_state is not None:
                # Load the best model state into the model, then save
                model.load_state_dict(best_model_state)
                torch.save(model.state_dict(), model_path)
                print(f"Best model for Fold {fold_idx + 1} saved (acc={best_global_acc:.4f}) at {model_path}")
        dist.barrier()
        # Optionally run a final gather to save the final predictions
        # or reuse the last epoch gather if you prefer. For clarity:
        final_val_acc, final_preds, final_labels, final_score = test_fn(model, test_loader, device)
        local_data = (final_preds, final_labels, final_score)
        if rank == 0:
            gather_list = [None for _ in range(world_size)]
        else:
            gather_list = None
        dist.gather_object(local_data, gather_list, dst=0)

            
        if rank == 0:
            global_preds = []
            global_labels = []
            global_score = []
            for gathered in gather_list:
                preds, labels, score = gathered
                global_preds.extend(preds)
                global_labels.extend(labels)
                global_score.append(score)

            # Suppose we want a "final global accuracy"
            correct = sum(pr == lab for pr, lab in zip(global_preds, global_labels))
            final_global_acc = correct / len(global_labels) if len(global_labels) else 0.0

            # Save results for this fold in a text file
            save_results_to_text(fold_dir, sample_names, val_idx, global_labels, global_preds, global_score, label_encoder)

        dist.barrier()

    dist.destroy_process_group()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optuna Search")
    parser.add_argument('--column', type=str, required=True, help='Column name to select from NCI_CBTN_class')
    parser.add_argument('--outdir', type=str, required=False, default=None, help='Directory to save the outputs')
    #parser.add_argument('--SMOTE', type=str, choices=['TRUE', 'FALSE'], required=False, default='FALSE', help='SMOTE for imbalacnce classe')
    args = parser.parse_args()

    # Convert the string to a boolean
    #use_smote = args.SMOTE == 'TRUE'
    
    # Intiate environment
    # Get the world size from the WORLD_SIZE variable or directly from SLURM:
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    # Likewise for RANK and LOCAL_RANK:
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    print("WORLD_SIZE:",world_size)
    print("RANK:", rank)
    print("LOCAL_RANK:", local_rank)
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
    
    
    run(world_size, rank, local_rank, graph_dataset, sample_names,label_encoder)

    # Collect data
    true_labels, predicted_labels, predicted_scores = collect_and_combine_results(args.outdir)

    # Annalyze data
    analyze_results(true_labels, predicted_labels, predicted_scores, label_encoder,outdir=args.outdir)
   
    print("All GNN and ViT hybride modle 5 fold cross validation process is Done...!")
    
