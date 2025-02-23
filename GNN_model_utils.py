import torch
import numpy as np
import pandas as pd
import os, json, random
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import seaborn as sns
import pickle,joblib
import torch.distributed as dist
from torch_geometric.data import Data, Dataset
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_self_loops
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report,roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold

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

# Creating a graph Data  with coords
def make_graph_data_with_coords(tile_features, slide_label, tile_coords, k=5):
    """
    Create a PyG `Data` object for one slide/WSI, with tile coordinates stored as `data.coords`.
    
    Args:
        tile_features (numpy.ndarray or torch.Tensor): [n_tiles, feature_dim]
        tile_coords (numpy.ndarray or torch.Tensor): [n_tiles, 2], (x, y) coords for each tile
        slide_label (int or float): Slide-level label (classification or regression)
        edge_index (torch.Tensor, optional): [2, num_edges] adjacency. Can be None for no edges.
        
    Returns:
        data (torch_geometric.data.Data)
    """
    # Convert tile features to torch Tensor if needed
    if not isinstance(tile_features, torch.Tensor):
        x = torch.from_numpy(tile_features).float()
    else:
        x = tile_features.float()
        
    # Convert coords to torch Tensor
    if not isinstance(tile_coords, torch.Tensor):
        coords = torch.from_numpy(tile_coords).float()
    else:
        coords = tile_coords.float()
    
    # Convert slide label to 1D tensor (for classification)
    y = torch.tensor([slide_label], dtype=torch.long)
    
    # Edge index
    # Generate KNN graph
    adjacency_matrix = kneighbors_graph(tile_features, n_neighbors=k, mode="connectivity", include_self=False)
    # Convert the output of nonzero() to a NumPy array
    nonzero_indices = np.array(adjacency_matrix.nonzero())  # Shape: (2, num_nonzero)

    # Create the PyTorch tensor
    edge_index = torch.tensor(nonzero_indices, dtype=torch.long)
    #edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)

    # Validate edge_index shape
    if edge_index.shape[0] != 2:
        raise ValueError(f"Invalid edge_index shape: {edge_index.shape}")

    
    # Construct PyG Data
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    graph_data.coords = coords  # <--- Storing tile coords here
    
    return graph_data

## crate tile level graph data
class TileGraphDataset(Dataset):
    def __init__(self, features, labels, coords=None, k=5, transform=None):
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
        self.coords= coords
        self.k = k
        self.transform = transform

        # Precompute the graphs
        if self.coords == None:
            self.graphs = [create_graph_data(tiles, label, k) for tiles, label in zip(self.features, self.labels)]
        else:
            self.graphs = [make_graph_data_with_coords(features, labels, coords, k) for features, labels, coords,k in zip(self.features, self.labels, self.coords, self.k)]


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


# Split data into five fold
def stratified_kfold_split(dataset, n_splits=5, shuffle=True, random_state=42):
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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Collect all labels from the graph dataset
    labels = []
    for data in dataset:
        labels.extend(data.y.tolist())  # Assuming labels are stored in `data.y`
    # Convert to a NumPy array
    labels = np.array(labels)
    print(f"Number of labels: {len(labels)}:")
    fold_indices = []
    for train_idx, val_idx in skf.split(np.arange(len(dataset)), labels):
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")
        print("-" * 30)
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
        print(f'Step {step+1}:')
        print('=======')
        
        if hasattr(data, 'num_graphs'):
            print(f'Number of graphs in the current batch: {data.num_graphs}')
        else:
            print("num_graphs attribute not found in data object.")
        
        print('Data in Batch:')
        print(data)
        print()

def save_label_encoder(label_encoder, outdir='models', filename='label_encoder.pkl'):
    """
    Save the label encoder to a file using pickle.
    
    Parameters:
    - label_encoder: The trained label encoder object
    - outdir: Directory to save the encoder
    - filename: The name of the file to save the encoder
    """
    os.makedirs(outdir, exist_ok=True)  # Ensure the output directory exists
    encoder_path = os.path.join(outdir, filename)
    
    # Save the label encoder with pickle
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"Label encoder saved to {encoder_path}")

# Function to load the label encoder from a file
def load_label_encoder(filepath):
    """
    Load the label encoder from a pickle file.
    
    Parameters:
    - filepath: Path to the saved label encoder
    
    Returns:
    - label_encoder: The loaded label encoder object
    """
    with open(filepath, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder

def save_results_to_text(results_path, sample_names, val_idx, val_labels, val_preds, val_scores, label_encoder):
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
    #results_path = os.path.join(fold_dir, "predicted_results.txt")
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


def plot_confusion_matrix_heatmap(conf_matrix_proportional, conf_matrix, class_labels,accuracy=0.0, outdir='plots', filename='Proportional_confusion_matrix_heatmap.png', title='Proportional Confusion Matrix'):
    """
    Function to plot and save a proportional confusion matrix heatmap with annotations.

    Parameters:
    - conf_matrix_proportional: 2D array of proportions (confusion matrix values normalized by row)
    - conf_matrix: 2D array of raw counts (confusion matrix)
    - class_labels: List of class labels
    - outdir: Directory to save the plot
    - filename: Filename to save the plot as (default: 'Proportional_confusion_matrix_heatmap.png')
    - title: Title of the plot (default: 'Proportional Confusion Matrix')
    """
    # Ensure the output directory exists
    #os.makedirs(outdir, exist_ok=True)

    # Plot the proportional confusion matrix heatmap with raw numbers annotated
    plt.figure(figsize=(10, 8))
    sns_heatmap = sns.heatmap(conf_matrix_proportional, 
                              annot=conf_matrix, 
                              cmap='coolwarm', 
                              xticklabels=class_labels, 
                              yticklabels=class_labels, 
                              fmt='d', 
                              cbar_kws={'label': 'Proportion'}, 
                              annot_kws={"size": 14, "weight": "bold", "color": "white"})

    # Customize the plot
    plt.xlabel('Predicted Cell States', fontsize=18, fontweight='bold')
    plt.ylabel('Actual Cell States', fontsize=18, fontweight='bold')
    # Subtitle positioned just below the main title
    plt.text(0.5, 0.90, f'Overall Accuracy: {round(accuracy, 2):.2f}', ha='center', va='center', fontsize=10, transform=plt.gcf().transFigure)
    plt.suptitle(title, y=0.95,fontsize=20)
    #plt.title(title, fontsize=20)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')

    # Adjust colorbar properties
    cbar = sns_heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)

    # Save the plot as a high-resolution image
    plot_path = os.path.join(outdir, filename)
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix heatmap saved to {plot_path}")

def plot_roc_auc(y_true, prediction_scores, label_encoder, outdir='plots', filename='ROC_AUC_plot.png'):
    """
    Function to plot multi-class ROC AUC curves and save the plot.

    Parameters:
    - y_true: Ground truth labels (torch tensor or NumPy array)
    - prediction_scores: Predicted scores (output from model, NumPy array)
    - label_encoder: Label encoder to decode class labels
    - outdir: Directory to save the plot
    - filename: Name of the plot file to save
    """
    # Ensure the output directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Binarize the true labels for multi-class ROC AUC calculation
    y_bin = label_binarize(y_true, classes=np.arange(len(label_encoder.classes_)))
    
    # Calculate ROC AUC score for each class
    roc_auc_scores = {}
    fpr = {}
    tpr = {}
    
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], prediction_scores[:, i])
        roc_auc_scores[label_encoder.classes_[i]] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    if len(label_encoder.classes_) > 3:
       colors = ['#66A61E', '#FF7F00', '#1F78B4','#C71587']
    elif len(label_encoder.classes_)==3:
       colors = ["#FFA206","#2F4B7C","#E31A1C"]
    
    for i, color in zip(range(len(label_encoder.classes_)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=4,
                 label=f'ROC curve for {label_encoder.classes_[i]} (area = {roc_auc_scores[label_encoder.classes_[i]]:0.2f})')
    
    # Plot baseline for random guess
    plt.plot([0, 1], [0, 1], 'k--', lw=4)
    
    # Customize plot appearance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.title('ROC Curve', fontsize=18, fontweight='bold')
    plt.legend(loc="lower right", prop={'weight': 'bold', 'size': 14})
    plt.rcParams.update({'font.size': 16, 'font.family': 'Arial'})
    plt.grid(alpha=0.3)
    
    # Save the plot to the specified directory
    plot_path = os.path.join(outdir, filename)
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"ROC AUC plot saved to {plot_path}")


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




def create_probability_heatmap(tile_coords, tile_probs, slide_width, slide_height,
                               class_idx=1, tile_size=256, downsample=32):
    """
    Create a heatmap for a single slide from tile probabilities.

    Args:
        tile_coords (list or np.ndarray): shape [N, 2], each row = (x, y) top-left corner of tile in original resolution
        tile_probs (np.ndarray): shape [N, num_classes], probabilities for each tile
        slide_width (int): width of the WSI at original resolution
        slide_height (int): height of the WSI at original resolution
        class_idx (int): which class probability to visualize
        tile_size (int): original tile width/height in pixels
        downsample (int): factor to downsample the slide for heatmap

    Returns:
        heatmap (np.ndarray): shape [H_ds, W_ds], containing the class probabilities
    """
    # Compute the downsampled slide dimensions
    H_ds = slide_height // downsample
    W_ds = slide_width // downsample

    heatmap = np.zeros((H_ds, W_ds), dtype=np.float32)

    # Fill heatmap with tile probabilities
    for i in range(len(tile_coords)):
        x, y = tile_coords[i]  # top-left corner in original resolution
        prob = tile_probs[i, class_idx]  # Probability for the chosen class

        # Downsampled coordinates
        x_ds = x // downsample
        y_ds = y // downsample

        # Downsampled tile size in this heatmap
        tile_size_ds = tile_size // downsample

        # Fill the region [y_ds:y_ds+tile_size_ds, x_ds:x_ds+tile_size_ds]
        # Clip to make sure we don't go out of bounds
        x_min = max(0, x_ds)
        y_min = max(0, y_ds)
        x_max = min(W_ds, x_ds + tile_size_ds)
        y_max = min(H_ds, y_ds + tile_size_ds)

        heatmap[y_min:y_max, x_min:x_max] = prob

    return heatmap


