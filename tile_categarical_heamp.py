import numpy as np
import pandas as pd
import os
import openslide
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import Patch



def create_predicted_score_scatter_plot(tile_coords, pred_score,scale_method="zscore", title="path_level_score", outdir="patch_prediction.png"):
    """
    Generates a scatter plot of patch-level IDH mutation predictions.

    Args:
        x_coords (np.ndarray): X coordinates for each patch.
        y_coords (np.ndarray): Y coordinates for each patch.
        idh_predictions (np.ndarray): IDH mutation prediction values.
        output_path (str): File path to save the plot.

    Returns:
        None (Displays and saves the figure)
    """
    print(f"Tile score lenth:{len(pred_score)}, Tile Coord lenth:{len(tile_coords)}")

    # Apply scaling method
    if scale_method == "minmax":
        pred_score = min_max_scale(pred_score)
    elif scale_method == "log":
        pred_score = log_scale(pred_score)
    elif scale_method == "zscore":
        pred_score = z_score_scale(pred_score)
    else:
        raise ValueError("Invalid scale_method. Choose 'minmax', 'log', or 'zscore'.")
    
    # Extract X and Y coordinates
    y_coords, x_coords = tile_coords[:, 0], tile_coords[:, 1]
    
    # Create scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_coords, y_coords, c=pred_score, cmap="viridis", edgecolors="black", alpha=0.8)

    # Flip the Y-axis so lower values appear on top
    plt.gca().invert_yaxis()
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Prediction score", fontsize=12)

    # Titles and labels
    plt.title(title, fontsize=14)
    #plt.xlabel("UMAP Dimension 1", fontsize=12)
    #plt.ylabel("UMAP Dimension 2", fontsize=12)

    # Save and show the plot
    plt.savefig(os.path.join(outdir,"patch_prediction.png"), bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Patch-level IDH prediction plot saved to {outdir}")

# Min-Max Scaling
def min_max_scale(scores, new_min=0, new_max=1):
    """
    Scales the tile scores to a given range using Min-Max Scaling.

    Args:
        scores (np.ndarray): Array of raw scores.
        new_min (float): Desired minimum value after scaling.
        new_max (float): Desired maximum value after scaling.

    Returns:
        np.ndarray: Scaled scores.
    """
    min_val = np.min(scores)
    max_val = np.max(scores)

    if max_val - min_val == 0:  # Avoid division by zero
        return np.full_like(scores, (new_min + new_max) / 2)

    scaled_scores = new_min + ((scores - min_val) / (max_val - min_val)) * (new_max - new_min)
    return scaled_scores

# Log Scaling (Good for Skewed Data)
def log_scale(scores):
    """
    Applies log scaling to tile scores.

    Args:
        scores (np.ndarray): Raw scores.

    Returns:
        np.ndarray: Log-transformed scores.
    """
    return np.log1p(scores)  # log(x + 1) to prevent log(0) errors

# Standardization (Z-Score Scaling)
def z_score_scale(scores):
    """
    Standardizes tile scores using Z-score normalization.

    Args:
        scores (np.ndarray): Raw scores.

    Returns:
        np.ndarray: Standardized scores.
    """
    mean = np.mean(scores)
    std = np.std(scores)

    if std == 0:  # Avoid division by zero
        return np.zeros_like(scores)

    return (scores - mean) / std



def create_multi_panel_scatter_plot(tile_coords, tile_scores,scale_method="zscore", outdir="multi_panel_plot.png"):
    """
    Generates a multi-panel plot with multiple scatter plots for different cell states.

    Args:
        tile_coords (np.ndarray): shape [N, 2], each row = (x, y) coordinates for patches.
        tile_scores (pd.DataFrame): DataFrame where each column is a different cell state score.
        outdir (str): Output directory to save the figure.

    Returns:
        None (Displays and saves the multi-panel figure)
    """

    num_plots = len(tile_scores.columns)
    rows = (num_plots + 1) // 2  # Arrange in 2 columns
    cols = 2 if num_plots > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 5))  # Adjust figure size
    axes = axes.flatten()  # Flatten axes for easy iteration

    for i, cellstate in enumerate(tile_scores.columns):
        pred_score = tile_scores[cellstate]  # Extract prediction scores
        # Apply scaling method
        if scale_method == "none" :
            print("Skipping scaling methode...!")
        elif scale_method == "minmax":
            pred_score = min_max_scale(pred_score)
        elif scale_method == "log":
            pred_score = log_scale(pred_score)
        elif scale_method == "zscore":
            pred_score = z_score_scale(pred_score)
        else:
            raise ValueError("Invalid scale_method. Choose 'minmax', 'log', or 'zscore'.")

        x_coords, y_coords = tile_coords[:, 0], tile_coords[:, 1]

        scatter = axes[i].scatter(x_coords, y_coords, c=pred_score, cmap="viridis", edgecolors="black", alpha=0.8)
        axes[i].invert_yaxis()  # Flip Y-axis
        axes[i].set_title(cellstate, fontsize=14)
        #axes[i].set_xlabel("UMAP Dimension 1", fontsize=12)
        #axes[i].set_ylabel("UMAP Dimension 2", fontsize=12)

    # Add a single shared colorbar
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.7, orientation="vertical")
    cbar.set_label("Prediction Score", fontsize=12)

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Save the figure
    save_path = os.path.join(outdir, "All_cellstate_prediction_score.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()

    print(f"Multi-panel plot saved to {save_path}")



def create_categorical_scatter_plot(tile_coords, tile_labels, category_labels, custom_colors, outdir="categorical_plot/"):
    """
    Generates a scatter plot where colors represent different categories.

    Args:
        tile_coords (np.ndarray): shape [N, 2], each row = (x, y) coordinates.
        categories (np.ndarray): shape [N], categorical labels as numbers (0, 1, 2, ...).
        category_labels (dict): Mapping of category numbers to category names.
        custom_colors (dict): Custom color mapping for each category.
        output_path (str): Path to save the figure.

    Returns:
        None (Displays and saves the figure)
    """

    # Define unique categories
    unique_categories = np.unique(tile_labels)

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    for cat in unique_categories:
        mask = tile_labels == cat
        plt.scatter(tile_coords[mask, 1], tile_coords[mask, 0], 
                    color=custom_colors[cat], label=category_labels[cat], 
                    edgecolors="black", alpha=0.8)

    # Flip Y-axis for proper orientation
    plt.gca().invert_yaxis()

    # Add legend for categorical variable
    plt.legend(title="Cell states", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Titles and labels
    plt.title("Predicted Cell states", fontsize=14)
    #plt.xlabel("UMAP Dimension 1", fontsize=12)
    #plt.ylabel("UMAP Dimension 2", fontsize=12)

    # Save and show the plot
    save_path = os.path.join(outdir, "cellstate_Class_prediction.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=600)
    plt.close()

    print(f"Categorical scatter plot saved to {save_path}")



def overlay_tile_paths_on_wsi(
    slide_path: str,
    tile_coords: np.ndarray,   # shape (N,2) in level 0 coords
    tile_labels: np.ndarray,   # shape (N,) numeric
    tile_size: int = 512,
    level_for_display: int = 2,
    label_names=None,          # dict label->string
    label_colors=None,         # dict label->(r,g,b)
    outdir: str = "./output",
    output_prefix: str = "overlay_paths"
):
    """
    Overlays tile paths (outlines) with label colors on a downsampled WSI image.
    """

    # Open the WSI
    slide = openslide.OpenSlide(slide_path)
    if level_for_display >= slide.level_count:
        raise ValueError(f"Invalid level {level_for_display}, max is {slide.level_count - 1}")
    ## magnification max
    if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties:
        mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
        print("mag_max:", mag_max)
        mag_original = mag_max
    else:
        print("[WARNING] mag not found, assuming: {mag_assumed}")
        mag_max = 40
        mag_original = 0

    ## downsample_level
    downsampling = int(int(mag_max)/mag_selected)
    print(f"downsampling: {downsampling}")

    # Get dimensions and downsampling factor
    display_width, display_height = slide.level_dimensions[level_for_display]
    downsample_factor = slide.level_downsamples[level_for_display]  # Get the correct downsample factor

    print(f"Using level {level_for_display} => Dimension: {display_width}x{display_height}, Downsample={downsample_factor}")

    # Read the region at this level
    display_region = slide.read_region((0, 0), level_for_display, (display_width, display_height))
    display_image = display_region.convert('RGB')

    # Default label-to-name mapping if not provided
    if label_names is None:
        label_names = {
            0: "Neural-Crest",
            1: "Neuronal",
            2: "Photoreceptor",
            3: "Proliferative",
        }

    # Default label-to-RGB color mapping if not provided
    if label_colors is None:
        label_colors = {
            0: (1.0, 0.0, 0.0),  # Red
            1: (0.0, 1.0, 0.0),  # Green
            2: (0.0, 0.0, 1.0),  # Blue
            3: (1.0, 1.0, 0.0),  # Yellow
        }

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(display_image)
    ax.axis('off')

    # Draw colored outlines for each tile
    for i, (row_level0, col_level0) in enumerate(tile_coords):
        lbl = tile_labels[i]
        color = label_colors.get(lbl, (1.0, 1.0, 1.0))  # Default to white if label not found

        # FIXED: Apply proper downsampling to avoid (0,0) overlap
        row_ds = int(row_level0 / downsample_factor)
        col_ds = int(col_level0 / downsample_factor)
        tile_size_ds = int(tile_size / downsample_factor)

        # Create an outline rectangle (no filled color)
        rect = patches.Rectangle(
            (col_ds, row_ds),  # x (left), y (top)
            tile_size_ds,       # Width
            tile_size_ds,       # Height
            linewidth=2,
            edgecolor=color,    # Use category color
            facecolor='none'    # Transparent fill
        )
        ax.add_patch(rect)

        # Optional debug print for first few tiles
        if i < 5:
            name = label_names.get(lbl, f"Label {lbl}")
            print(f"Tile {i} => Label={lbl} ({name}), Region=({col_ds},{row_ds}), Size={tile_size_ds}")

    # Build a legend from label_names
    legend_patches = []
    for lbl, name in label_names.items():
        c = label_colors.get(lbl, (1.0, 1.0, 1.0))
        legend_patches.append(Patch(edgecolor=c, facecolor='none', label=name, linewidth=2))

    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Save the figure
    output_path = os.path.join(outdir, f"{output_prefix}_level{level_for_display}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved tile path overlay to {output_path}")

