# Medulloblastoma Celltype prdiction from H&E images using Graph-Neural-Network-Model
<img width="1536" height="1024" alt="GNN-Model -Archetecture" src="https://github.com/user-attachments/assets/8b6d0a4b-d334-4c42-aa1a-9cac13b08df2" />

#Tile-Based Feature Extraction from WSIs

Slide Selection: Load metadata and select whole-slide image (WSI) using slide index.

Tiling Strategy: WSI is divided into non-overlapping 512×512 px tiles at 20× magnification using OpenSlide.

Tile Filtering: Tiles are evaluated for edge sharpness and retained if above thresholds.

Color Normalization: Valid tiles undergo Macenko normalization to correct staining variation.

Visualization: Two composite masks are generated to visualize all and accepted tiles with grid overlays.

Feature Extraction:

Tiles resized to 224×224 px and normalized using ImageNet mean/std.

A pre-trained ViT (vit_large_patch16_224) model is applied to extract deep features in batches.

Output: Extracted features are saved as .npy files for downstream ML tasks.

# GNN + Attention-based MIL for Tile-Level Prediction

Graph Construction:
Each slide is represented as a graph, where nodes are tile-level image embeddings (e.g., 1024-D features), and edges define spatial or feature-based relationships between tiles.

Graph Convolutional Layers (GNN):
Multiple GraphConv layers (with ReLU, BatchNorm, and Dropout) are applied to model inter-tile interactions and learn context-aware node (tile) embeddings.

Instance-Level Classification:
A shared MLP head predicts class logits for each tile (instance), enabling localized predictions.

Gated Attention Mechanism:
Attention weights are computed per tile using a gating function (tanh × sigmoid), which selectively weighs tiles based on relevance to the slide-level task.

MIL Pooling & Slide-Level Prediction:
Attention-weighted tile logits are aggregated to form a slide-level prediction via weighted summation, enabling multi-instance learning (MIL).

End-to-End Output:
The model outputs:

Slide-level logits for overall classification

Tile-level logits for localized interpretation

Attention weights for spatial explainability
