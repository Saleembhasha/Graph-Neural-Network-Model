
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.neighbors import kneighbors_graph
from torch_geometric.nn import GraphConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from MLP_classifier_Utils import *
import optuna

# Set random seed for reproducibility
def set_random_seed(seed=42):
    # Python RNG
    np.random.seed(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(seed=42)

# Setup MASTER_ADDR and MASTER_PORT for SLURM environments
def setup_master_address_and_port():
    """Setup distributed environment for multi-node, multi-GPU training."""
    if "SLURM_NODELIST" in os.environ:
        master_node = os.popen(f"scontrol show hostnames {os.environ['SLURM_NODELIST']}").read().splitlines()[0]
        os.environ["MASTER_ADDR"] = master_node
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    # Dynamically set MASTER_PORT
    #os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_PROCID", 0))))
    os.environ['MASTER_PORT'] = '12345'


def setup_slurm_environment():
    if 'SLURM_NODELIST' in os.environ:
        master_node = os.popen(f"scontrol show hostnames {os.environ['SLURM_NODELIST']}").read().splitlines()[0]
        os.environ['MASTER_ADDR'] = master_node
    else:
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    # Dynamically set MASTER_PORT if not set
    os.environ.setdefault("MASTER_PORT", str(29500 + int(os.environ.get("SLURM_PROCID", 0))))

    # SLURM-specific rank and world size
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
    #rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
    #local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
    return world_size

# GNN model
class GNN(torch.nn.Module):
    """
    A Graph Neural Network (GNN) for graph classification tasks.

    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Dimension of hidden layers.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability for regularization.
    """
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super(GNN, self).__init__()
        #torch.manual_seed(123)
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


# Training and evaluation function

def train_and_evaluate(rank, train_loader, val_loader, model, criterion, optimizer, num_epochs):
    
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            data = data.to(rank)
            optimizer.zero_grad()
            outputs = model(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()
        dist.barrier()
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Validation
    model.eval()
    val_preds, val_labels, val_scores = [], [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(rank)
            outputs = model(data.x, data.edge_index, data.batch)
            preds = outputs.argmax(dim=1)
            scores = F.softmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(data.y.cpu().numpy())
            val_scores.extend(scores.cpu().numpy())
    accuracy = accuracy_score(val_labels, val_preds)
    if rank == 0:
        print(f"Validation Accuracy: {accuracy:.4f}")
    dist.barrier()
    return accuracy, val_preds, val_labels, val_scores



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

def objective(trial, rank, world_size, graph_dataset):

    # Hyperparameters to tune
    hidden_dim = trial.suggest_int("hidden_dim", 32, 1024, step=16)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_epochs = trial.suggest_int("num_epochs", 10, 500, step=10)
    hyperparams = [hidden_dim, dropout, learning_rate, batch_size, num_epochs]

    # Broadcast hyperparameters from rank 0 to all other ranks
    dist.broadcast_object_list(hyperparams, src=0)
    hidden_dim, dropout, learning_rate, batch_size, num_epochs,patience = hyperparams

    # Load data and split into train and validation sets
    #graph_dataset = TileGraphDataset(features, slide_labels, k=k)
    #graph_dataset = graph_dataset.to(rank)
    train_dataset, val_dataset = split_graph_data(graph_dataset, train_percent=80, val_percent=20)

    # Use DistributedSampler
    train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    input_dim=1024
    num_classes=4

    set_random_seed(seed=42)
    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Calculate class weights
    class_weights = calculate_class_weights_from_graph(train_dataset, device=rank)

    # Use CrossEntropyLoss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    accuracy, _, _, _ =train_and_evaluate(rank, train_loader, val_loader, model, criterion, optimizer, num_epochs=num_epochs)

    return accuracy



def distributed_hyperparameter_search(rank:int, world_size:int,graph_dataset, n_trials=50):
    # Initialize distributed environment
    #setup_master_address_and_port()
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    def optuna_objective(trial):
       return objective(trial, rank, world_size, graph_dataset)
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=n_trials)

    if rank == 0:
        print("Best hyperparameters:", study.best_params)
        with open("best_params.json", "w") as f:
            json.dump(study.best_params, f)
        print("Best parameters saved to 'best_params.json'")

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
    labels = np.array(all_labels)

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

