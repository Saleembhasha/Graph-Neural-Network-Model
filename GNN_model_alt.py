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
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from MLP_classifier_Utils import *
import optuna

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

# Setup MASTER_ADDR and MASTER_PORT for SLURM environments
def setup_distributed_environment():
    '''if 'SLURM_NODELIST' in os.environ:
        master_node = os.popen(f"scontrol show hostnames {os.environ['SLURM_NODELIST']}").read().splitlines()[0]
        os.environ['MASTER_ADDR'] = master_node
    else:
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    # Dynamically set MASTER_PORT if not set
    os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_PROCID", 0))))'''

    # SLURM-specific rank and world size
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    rank =int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    #local_rank = int(os.environ.get('SLURM_LOCALID', rank % torch.cuda.device_count()))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    return world_size,rank,local_rank

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


# Training and evaluation function
def train_and_evaluate(rank,local_rank, train_loader, val_loader, model, criterion, optimizer, num_epochs):
    # Ensure correct device setup
    #device = torch.device(f"cuda:{local_rank}")
    #model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(local_rank)
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        # Synchronize all processes before moving to validation
        dist.barrier()
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_preds, val_labels, val_scores = [], [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(local_rank)
            outputs = model(data.x, data.edge_index, data.batch)
            preds = outputs.argmax(dim=1)
            scores = F.softmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(data.y.cpu().numpy())
            val_scores.extend(scores.cpu().numpy())      
    accuracy = accuracy_score(val_labels, val_preds)
    if rank == 0 :
        print(f"Validation Accuracy: {accuracy:.4f}")

    # Synchronize all processes before returning results
    dist.barrier()

    return accuracy, val_preds, val_labels, val_scores

# Calculate class weights
def calculate_class_weights_from_graph(graph_dataset, device='cpu'):
    """
    Calculate class weights from a graph dataset based on the distribution of node labels.

    Args:
        graph_dataset (Dataset): A PyTorch Geometric dataset containing graph data.
        num_classes (int): Total number of classes.
        device (str): Device to which the weights should be moved ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Class weights as a PyTorch tensor.
    """
    # Collect all labels from the graph dataset
    all_labels = []
    for data in graph_dataset:
        all_labels.extend(data.y.tolist())  # Assuming labels are stored in `data.y`

    # Convert to a NumPy array
    all_labels = np.array(all_labels)
    classes = np.unique(all_labels)
    # Calculate class weights using balanced mode
    class_weights = compute_class_weight(class_weight='balanced',classes=classes, y=all_labels)
    # Convert to a PyTorch tensor and move to the specified device
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights

def train_and_evaluate_alt(rank, local_rank, train_loader, val_loader, model, criterion, optimizer, num_epochs, patience,stopping_metric="precision"):
    assert stopping_metric in ["precision", "accuracy"], "stopping_metric must be 'precision' or 'accuracy'."
    best_metric = 0.0
    patience_counter = 0
    best_val_preds, best_val_labels, best_val_score, best_model_state = None, None, None, None
    scaler = GradScaler()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(local_rank)
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Synchronize all processes before moving to validation
        dist.barrier()

        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
        
        model.eval()
        val_preds, val_labels, val_score = [], [], []
        metric = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(local_rank)
                outputs = model(data.x, data.edge_index, data.batch)
                preds = outputs.argmax(dim=1)
                score = F.softmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())
                val_score.extend(score.cpu().numpy())

        # Calculate the chosen metric
        if stopping_metric == "precision":
            metric = precision_score(val_labels, val_preds, average="macro")
        elif stopping_metric == "accuracy":
            metric = accuracy_score(val_labels, val_preds)

        # Synchronize metric across ranks
        metric_tensor = torch.tensor(metric, device=local_rank)
        dist.reduce(metric_tensor, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            metric = metric_tensor.item() / dist.get_world_size() # Average across ranks
            print(f"Epoch {epoch+1}/{num_epochs}, Validation {stopping_metric}: {metric:.4f}")

        # Broadcast metric to all ranks
        #dist.broadcast(metric_tensor, src=0)
        #metric = metric_tensor.item()

        # Early Stopping Check
        if metric > best_metric:
            best_metric = metric
            patience_counter = 0  
            best_val_preds = val_preds
            best_val_labels = val_labels
            best_val_score = val_score
            best_model_state = model.state_dict()  # Save the best model state
                
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if rank == 0:
                print(f"Early stopping after {patience} epochs with no improvement.")
            break

        
        # Synchronize all processes before proceeding to the next epoch
        dist.barrier()

    # Load the best model state before returning
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_metric, best_val_preds, best_val_labels, best_val_score




def objective(trial, rank, world_size,local_rank, graph_dataset):

    
    # Initialize hyperparameters to None for all ranks
   
    # Hyperparameters to tune
    if rank == 0:
        hidden_dim = trial.suggest_int("hidden_dim", 16, 1024, step=16)
        num_layers = trial.suggest_int("hidden_dim", 2, 6)
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.1)
        learning_rate = trial.suggest_categorical("learning_rate", 1e-5, 1e-1, log=False)
        batch_size = trial.suggest_categorical("batch_size", 16,128, step=16)
        num_epochs = trial.suggest_int("num_epochs", 100, 1000, step=100)
        patience = trial.suggest_int("patience",10,50, step=10)
        
    else:
        hidden_dim = 512
        num_layers = 3
        dropout = 0.2
        learning_rate = 1e-3
        batch_size = 64
        num_epochs = 100
        patience = 10 

    # Broadcast hyperparameters from rank 0 to all other ranks
    hyperparams = [hidden_dim,num_layers, dropout, learning_rate, batch_size, num_epochs,patience]
    dist.broadcast_object_list(hyperparams, src=0)
    hidden_dim,num_layers, dropout, learning_rate, batch_size, num_epochs,patience = hyperparams
    
    # Load data and split into train and validation sets 
    train_dataset, val_dataset = split_graph_data(graph_dataset, train_percent=80, val_percent=20)
    
    # Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    model = GNN(input_dim=1024, hidden_dim=hidden_dim, num_classes=4,num_layers=num_layers,dropout=dropout).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Calculate class weights
    class_weights = calculate_class_weights_from_graph(train_dataset, device=local_rank)

    # Use CrossEntropyLoss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #accuracy, _, _, _ =train_and_evaluate(rank,local_rank, train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs)

    _, accuracy, _, _, _ =train_and_evaluate_alt(rank,local_rank, train_loader, val_loader, model, criterion, optimizer, num_epochs, patience, stopping_metric="accuracy")

    return accuracy



def distributed_hyperparameter_search(world_size: int, rank: int, local_rank: int, graph_dataset, outdir, n_trials=50):
    try:
        print(f"Rank {rank}: Initializing process group with MASTER_ADDR={os.environ.get('MASTER_ADDR')} and MASTER_PORT={os.environ.get('MASTER_PORT')}")

        dist.init_process_group(backend='nccl', init_method="env://", rank=rank, world_size=world_size, timeout=timedelta(seconds=1200))
        
        torch.cuda.set_device(local_rank)

        def optuna_objective(trial):
            return objective(trial, rank, world_size, local_rank, graph_dataset)

        user = os.getenv("MYSQL_USER", "Saleem")
        password = os.getenv("MYSQL_PASSWORD", "Aira123")
        host = os.popen(f"scontrol show hostnames {os.environ['SLURM_NODELIST']}").read().splitlines()[0]
        port = os.getenv("MYSQL_PORT", "55555")
        db = os.getenv("MYSQL_DB", "optuna_db")

        # Connect to MySQL database and create/load study
        storage_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        #storage_url = "mysql+pymysql://Saleem:Aira123@localhost:55555/optuna_db"
        study_name = f"study_{rank}"
        print(f"Connecting to Optuna database: {storage_url}")

        # Create or load the study
        study = optuna.create_study(study_name=study_name, storage=storage_url, load_if_exists=True)
        print(f"Study '{study_name}' connected. Starting optimization...")
        study.optimize(optuna_objective, n_trials=n_trials)

        if rank == 0:
            best_params = study.best_params
            print(f"Best hyperparameters: {best_params}")
            with open(os.path.join(outdir, "best_params.json"), "w") as f:
                json.dump(best_params, f)
        else:
            best_params = None

        # Broadcast best parameters to all ranks
        if rank == 0:
            best_params_list = [
                best_params['hidden_dim'],
                best_params['dropout'],
                best_params['learning_rate'],
                best_params['batch_size'],
                best_params['num_epochs'],
                best_params['patience']
            ]
        else:
            best_params_list = [None] * 6

        if dist.is_initialized():
            dist.broadcast_object_list(best_params_list, src=0)

        # Extract hyperparameters
        hidden_dim, dropout, learning_rate, batch_size, num_epochs, patience = best_params_list

        # Perform five-fold cross-validation
        mean_accuracy = perform_five_fold_cross_validation(graph_dataset, best_params, rank, local_rank, outdir)

        if rank == 0:
            print(f"Five-fold mean accuracy: {mean_accuracy}")

        dist.barrier()
    except Exception as e:
        print(f"Rank {rank} encountered an error: {e}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


## extract featutes and sample name

def extract_sample_names_and_features(slide_features):
    """
    Extract sample names and features from slide features.

    Args:
        slide_features (list): A list where each element is a tuple (sample_name, features_array),
                               where `features_array` is a 2D array of tile-level features 
                               (e.g., shape [n_tiles, n_features]).

    Returns:
        tuple: 
            - sample_names (list): List of sample names.
            - features (list): List of tile-level feature arrays for each sample.
    """
    # Extract sample names
    sample_names = [f[0] for f in slide_features]
    
    # Extract features
    features = [f[1] for f in slide_features]  # Assuming f[1] contains the tile-level features (e.g., [n_tiles, 1024])

    return sample_names, features


# Scaling the tile level Features
def scale_tile_features(features):
    """
    Scale the tile features for each sample.

    Args:
        features (list of np.ndarray): Each element is an array of shape [n_tiles, 1024].

    Returns:
        scaled_features (list of np.ndarray): Scaled tile features for each sample.
    """
    scaler = StandardScaler()
    scaled_features = [scaler.fit_transform(sample_features) for sample_features in features]
    return scaled_features

# Split dataset into training and val
def split_graph_data(dataset, train_percent=70, val_percent=30, test_percent=None, shuffle=True):
    """
    Split a dataset into training, validation, and optionally test sets based on percentages.

    Args:
        dataset (Dataset): A PyTorch Geometric dataset or similar dataset.
        train_percent (int): Percentage of data to use for training (0-100).
        val_percent (int): Percentage of data to use for validation (0-100).
        test_percent (int, optional): Percentage of data to use for testing (0-100). If None, no test split is created.
        shuffle (bool): Whether to shuffle the dataset before splitting.

    Returns:
        tuple:
            - train_dataset (Dataset): Training subset.
            - val_dataset (Dataset): Validation subset.
            - test_dataset (Dataset, optional): Test subset if test_percent is provided.
    """
    
    if shuffle:
        dataset = dataset.shuffle()

    # Validate percentages
    if test_percent is not None and (train_percent + val_percent + test_percent > 100):
        raise ValueError("Train, validation, and test percentages cannot exceed 100.")

    # Calculate the number of samples
    num_samples = len(dataset)
    num_train = int((train_percent / 100.0) * num_samples)
    num_val = int((val_percent / 100.0) * num_samples)

    # Split the dataset
    train_dataset = dataset[:num_train]
    val_dataset = dataset[num_train:num_train + num_val]

    if test_percent is not None:
        num_test = int((test_percent / 100.0) * num_samples)
        test_dataset = dataset[num_train + num_val:num_train + num_val + num_test]
        return train_dataset, val_dataset, test_dataset

    return train_dataset, val_dataset


def create_graph_data(tiles, labels, k=5):
    """
    Create graph data from tile features.
    Args:
        tiles (numpy.ndarray): Tile features [num_tiles, num_features].
        labels (int): Slide-level label.
        k (int): Number of neighbors for KNN graph.

    Returns:
        graph_data: PyTorch Geometric Data object.
    """

    # Generate KNN graph
    adjacency_matrix = kneighbors_graph(tiles, n_neighbors=k, mode="connectivity", include_self=False)
    # Convert the output of nonzero() to a NumPy array
    nonzero_indices = np.array(adjacency_matrix.nonzero())  # Shape: (2, num_nonzero)

    # Create the PyTorch tensor
    edge_index = torch.tensor(nonzero_indices, dtype=torch.long)
    #edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)

    # Validate edge_index shape
    if edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape: {edge_index.shape}")

    # Add self-loops to prevent empty edge_index
    #edge_index, _ = add_self_loops(edge_index, num_nodes=tiles.shape[0])

    # Convert to PyTorch tensors
    x = torch.tensor(tiles, dtype=torch.float32)  # Node features
    y = torch.tensor([labels], dtype=torch.long)  # Slide-level label

    # Create graph data
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    return graph_data

class TileGraphDataset(Dataset):
    def __init__(self, features, labels, k, transform=None):
        """
        Args:
            features (list of numpy.ndarray): List of tile feature arrays, one per sample.
            labels (list of int): List of slide-level labels.
            k (int): Number of neighbors for KNN graph.
            transform (callable, optional): Transformation to apply to each graph.
        """
        super(TileGraphDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.k = k
        self.transform = transform

        # Precompute the graphs
        self.graphs = [create_graph_data(tiles, label, k) for tiles, label in zip(self.features, self.labels)]

    def len(self):
        """Return the number of graphs."""
        return len(self.graphs)

    def get(self, idx):
        """
        Get the graph at the specified index.

        Args:
            idx (int): Index of the graph to retrieve.

        Returns:
            Data: A PyTorch Geometric Data object.
        """
        graph = self.graphs[idx]
        if self.transform:
            graph = self.transform(graph)
        return graph

    def shuffle(self, return_perm=False):
        """
        Shuffle the dataset.

        Args:
            return_perm (bool): If True, also return the permutation applied.

        Returns:
            TileGraphDataset: A shuffled dataset.
            Optional[torch.Tensor]: The permutation applied (if return_perm=True).
        """
        perm = torch.randperm(len(self.graphs)).tolist()
        self.graphs = [self.graphs[i] for i in perm]
        if return_perm:
            return self, perm
        return self




def stratified_kfold_split(dataset, num_folds=5, shuffle=True, random_state=42):
    """
    Perform Stratified K-Fold Cross Validation on a dataset.

    Args:
        dataset (Dataset): A PyTorch Geometric dataset or similar dataset.
        labels (array-like): Array of labels corresponding to the dataset.
        num_folds (int): Number of folds (default: 5).
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Random seed for reproducibility.

    Returns:
        list of tuples: List where each tuple contains (train_idx, val_idx).
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)

    # Collect all labels from the graph dataset
    labels = []
    for data in dataset:
        labels.extend(data.y.tolist())  # Assuming labels are stored in `data.y`
    # Convert to a NumPy array
    labels = np.array(labels)

    fold_indices = []
    for train_idx, val_idx in skf.split(range(len(dataset)), labels):
        fold_indices.append((train_idx, val_idx))

    return fold_indices

# Check batch loader
def check_batches(dataloader):
    """
    Function to iterate over a DataLoader and print information about each batch.

    Args:
        dataloader (DataLoader): PyTorch Geometric DataLoader object.
    """
    for step, data in enumerate(dataloader):
        print(f'Step {step + 1}:')
        print('=======')
        
        if hasattr(data, 'num_graphs'):
            print(f'Number of graphs in the current batch: {data.num_graphs}')
        else:
            print("num_graphs attribute not found in data object.")
        
        print('Data in Batch:')
        print(data)
        print()


def save_results_to_text(fold_dir, sample_names, val_idx, val_labels, val_preds, val_scores, label_encoder):
    """
    Save prediction results to a text file in the specified format.

    Args:
        fold_dir (str): Directory to save the results file.
        sample_names (list): List of sample names.
        val_idx (list): Validation indices corresponding to the sample names.
        val_labels (list): List of true labels.
        val_preds (list): List of predicted labels.
        val_scores (list): List of predicted probabilities.
        label_encoder: LabelEncoder object containing class names.

    Returns:
        None
    """

    # File path
    results_path = os.path.join(fold_dir, "predicted_results.txt")
    class_names = label_encoder.classes_  # Get class names

    # Write to file
    with open(results_path, "w") as f:
        # Write header
        header = "Sample_Name,True_Label,Predicted_Label," + ",".join(class_names) + "\n"
        f.write(header)

        # Write each result
        for idx, (true_label, pred_label, scores) in enumerate(zip(val_labels, val_preds, val_scores)):
            sample_name = sample_names[val_idx[idx]]  # Retrieve sample name using val_idx

            # Convert scores to a comma-separated string
            scores_str = ",".join(map(str, scores.tolist()))

            # Write result row
            f.write(f"{sample_name},{true_label},{pred_label},{scores_str}\n")

    print(f"Results saved to {results_path}")

# Function to collect results from subdirectories
def collect_and_combine_results(root_directory):
    """
    Collect and append results from multiple fold directories into a single file and return results.

    Args:
        fold_dirs (list): List of directories containing the results files (one per fold).
        outdir (str): Directory to save the combined results file.

    Returns:
        tuple: (true_labels, predicted_labels, predicted_scores)
               - true_labels: Array of true labels.
               - predicted_labels: Array of predicted labels.
               - predicted_scores: Array of predicted scores for each class.
    """
    output_file = os.path.join(root_directory, "Predicted_5FC_result.csv")

    # Find all fold directories
    fold_dirs = [os.path.join(root_directory, item) for item in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, item))]
    
    # List to store data from each fold
    combined_results = []

    for fold_dir in fold_dirs:
        # Path to the fold's results file
        results_file = os.path.join(fold_dir, "predicted_results.txt")

        if not os.path.exists(results_file):
            print(f"Results file not found in {fold_dir}. Skipping...")
            continue

        # Read fold results into a DataFrame
        fold_df = pd.read_csv(results_file)
        combined_results.append(fold_df)

    # Combine all fold dataframes row-wise
    combined_df = pd.concat(combined_results, ignore_index=True)

    # Save combined results to the output file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined results saved to {output_file}")

    # Extract true labels, predicted labels, and predicted scores
    true_labels = combined_df["True_Label"].values
    predicted_labels = combined_df["Predicted_Label"].values
    predicted_scores = combined_df.iloc[:, 3:].values  # Scores start from the 4th column

    return true_labels, predicted_labels, predicted_scores


def analyze_results(true_labels, predicted_labels, predicted_scores,label_encoder, outdir):
    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    conf_matrix_df = pd.DataFrame(conf_matrix)
    conf_matrix_df.to_csv(os.path.join(outdir, 'confusion_matrix.csv'))
    
    # Calculate overall performance metrics
    overall_accuracy = accuracy_score(true_labels, predicted_labels)

    # Plot confusion matrix heatmap
    conf_matrix_proportional = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize
    plot_confusion_matrix_heatmap(conf_matrix_proportional,
                                  conf_matrix,
                                  label_encoder.classes_,
                                  outdir=outdir,
                                  accuracy=overall_accuracy,
                                  filename='confusion_matrix_heatmap.png',
                                  title='Proportional Confusion Matrix (n=321)')

    # ROC Curve and AUC
    plot_roc_auc(true_labels, predicted_scores, label_encoder, outdir=outdir,filename='roc_auc_curve.png')
    
    # Generate and save classification report
    class_report = classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_, zero_division=0)
    with open(os.path.join(outdir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)
    print("Data Analysis Done!")


# five fold cross validation
def perform_five_fold_cross_validation(graph_dataset, sample_names, best_params, world_size, rank, local_rank, outdir):
    """
    Perform five-fold cross-validation with given parameters.

    Parameters:
        graph_dataset (Dataset): The dataset to be used for cross-validation.
        labels (list): The labels corresponding to the dataset.
        best_params (dict): Best hyperparameters for training.
        rank (int): Rank of the process for distributed training.
        local_rank (int): Local rank of the GPU.
        outdir (str): Directory to save fold results.
        label_encoder (object): Encoder to convert labels to/from their encoded forms.
        sample_names (list): List of sample names corresponding to the dataset.
        stratified_kfold_split (function): Function to perform stratified K-fold split.
        calculate_class_weights_from_graph (function): Function to calculate class weights from the graph dataset.
        train_and_evaluate_alt (function): Function to train and evaluate the model.
        check_batches (function): Function to validate data loaders.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        device (str, optional): Device for computation. Defaults to "cuda".

    Returns:
        float: Mean accuracy across all folds.
    """
    folds = stratified_kfold_split(graph_dataset, num_folds=5)
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        # Create a directory for this fold
        fold_dir = os.path.join(outdir, f"result_{fold_idx + 1}")
        os.makedirs(fold_dir, exist_ok=True)  # Create folder if it doesn't exist

        # Split dataset
        train_dataset = Subset(graph_dataset, train_idx)
        val_dataset = Subset(graph_dataset, val_idx)
        # Use DistributedSampler
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

        # Validate data loaders
        print("Checking train loader...")
        check_batches(train_loader)
        print("Checking validation loader...")
        check_batches(val_loader)

        
        hidden_dim, dropout, learning_rate, batch_size, num_epochs, patience = best_params

        # Initialize model
        model = GNN(input_dim=1024, hidden_dim=hidden_dim, num_classes=4, dropout=dropout).to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        # Calculate class weights
        class_weights = calculate_class_weights_from_graph(train_dataset, device=local_rank)

        # Use CrossEntropyLoss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(local_rank)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate
        model, accuracy, val_preds, val_labels, val_scores = train_and_evaluate_alt(
            rank,local_rank, train_loader, val_loader,model, criterion, optimizer, num_epochs=num_epochs, patience=patience, stopping_metric="accuracy"
        )
        fold_accuracies.append(accuracy)

        if rank == 0:
            # Save model for this fold
            model_path = os.path.join(fold_dir, "GNN_ViT_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model for Fold {fold_idx + 1} saved at {model_path}")

            # Save results for this fold in a text file
            save_results_to_text(fold_dir, sample_names, val_idx, val_labels, val_preds, val_scores, label_encoder)
            print(f"Fold {fold_idx + 1} Accuracy: {accuracy}")

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Mean Accuracy across folds: {mean_accuracy}")

    return mean_accuracy

