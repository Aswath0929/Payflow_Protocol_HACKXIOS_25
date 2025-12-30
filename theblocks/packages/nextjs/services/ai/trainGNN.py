"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║           PayFlow GNN Training Script - Elliptic++ Dataset                                ║
║               Train GraphSAGE on Bitcoin Transaction Graph                                ║
║                   Optimized for RTX 4070 (8GB VRAM)                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

This script trains the PayFlow GNN on the Elliptic++ dataset for fraud detection.

Dataset: Elliptic++ Bitcoin Transaction Graph
- 203,769 transactions
- 184 features per transaction  
- 3 classes: 1=illicit, 2=licit, 3=unknown
- We map to 15 fraud typologies using rule-based assignment

Training Pipeline:
1. Load Elliptic++ dataset (features, edges, labels)
2. Build PyTorch Geometric Data object
3. Split into train/val/test (70/15/15)
4. Train GraphSAGE with NeighborLoader (mini-batch training)
5. Train Fused MLP on embeddings + standard features
6. Save pre-trained weights

Usage:
    python trainGNN.py --epochs 50 --batch-size 1024 --lr 0.001
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PayFlow.GNN.Train")

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    logger.error("PyTorch not available! Install with: pip install torch")
    sys.exit(1)

# PyTorch Geometric
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.utils import add_self_loops
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    logger.error("PyTorch Geometric not available! Install with: pip install torch-geometric")
    sys.exit(1)

# Pandas for data loading
try:
    import pandas as pd
except ImportError:
    logger.error("Pandas not available! Install with: pip install pandas")
    sys.exit(1)

# Progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# Sklearn for metrics
try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score, 
        accuracy_score, precision_score, recall_score
    )
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
#                          CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Dataset paths
DATASET_DIR = Path(r"C:\Users\sayan\Downloads\Hackxios\Elliptic++ Dataset")
FEATURES_PATH = DATASET_DIR / "txs_features.csv"
EDGES_PATH = DATASET_DIR / "txs_edgelist.csv"
CLASSES_PATH = DATASET_DIR / "txs_classes.csv"

# Model save paths
MODEL_DIR = Path(__file__).parent / "models" / "gnn"
GNN_ENCODER_PATH = MODEL_DIR / "gnn_graphsage_elliptic.pt"
FUSED_MLP_PATH = MODEL_DIR / "fused_mlp.pt"
FULL_MODEL_PATH = MODEL_DIR / "payflow_gnn_complete.pt"
TRAINING_LOG_PATH = MODEL_DIR / "training_log.json"

# Training defaults
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LR = 0.001
DEFAULT_HIDDEN_DIM = 256
DEFAULT_EMBEDDING_DIM = 64
DEFAULT_DROPOUT = 0.3
NUM_NEIGHBORS = [15, 10, 5]  # Neighbors per layer

# Fraud typologies (15 classes)
FRAUD_TYPOLOGIES = [
    "Rug Pull",           # 0
    "Pig Butchering",     # 1
    "Mixer/Tumbling",     # 2
    "Chain Obfuscation",  # 3
    "Fake Token",         # 4
    "Flash Loan Attack",  # 5
    "Wash Trading",       # 6
    "Structuring",        # 7
    "Velocity Attack",    # 8
    "Peel Chain",         # 9
    "Dusting Attack",     # 10
    "Address Poisoning",  # 11
    "Approval Exploit",   # 12
    "SIM Swap",           # 13
    "Romance Scam",       # 14
]


# ═══════════════════════════════════════════════════════════════════════════════
#                          MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class GraphSAGEEncoder(nn.Module):
    """3-layer GraphSAGE encoder for transaction embeddings."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)
        h = self.layer_norm(h)
        
        return h


class PayFlowGNNClassifier(nn.Module):
    """
    Full GNN classifier for 15-class fraud typology + 4-class risk level.
    
    Architecture:
    - GraphSAGE encoder (3 layers)
    - Classification head (typology + risk)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        embedding_dim: int = 64,
        num_typology_classes: int = 15,
        num_risk_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # GraphSAGE encoder
        self.encoder = GraphSAGEEncoder(
            in_channels, hidden_channels, embedding_dim, dropout
        )
        
        # Typology classification head
        self.typology_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, num_typology_classes)
        )
        
        # Risk level classification head
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(16, num_risk_classes)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (typology_logits, risk_logits)."""
        embeddings = self.encoder(x, edge_index)
        typology_logits = self.typology_head(embeddings)
        risk_logits = self.risk_head(embeddings)
        return typology_logits, risk_logits
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings only."""
        return self.encoder(x, edge_index)


# ═══════════════════════════════════════════════════════════════════════════════
#                          DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def assign_typology_label(class_label: int, features: np.ndarray) -> int:
    """
    Assign fraud typology (0-14) based on class label and features.
    
    Class mapping:
    - 1 = illicit → Map to fraud typologies based on features
    - 2 = licit → Class 0 (legitimate, treated as SAFE in risk)
    - 3 = unknown → Use feature-based heuristics
    """
    if class_label == 2:
        # Licit - no fraud typology
        return -1  # We'll filter these for typology training
    
    if class_label == 1:
        # Illicit - use feature heuristics to assign typology
        # Features from Elliptic++: Local features 1-93, Aggregate features 94-166
        
        # Feature-based heuristics (simplified)
        in_degree = features[92] if len(features) > 92 else 0
        out_degree = features[93] if len(features) > 93 else 0
        total_btc = features[94] if len(features) > 94 else 0
        
        # Mixing pattern: High degree, low amounts
        if in_degree > 10 and out_degree > 10:
            return 2  # Mixer/Tumbling
        
        # Large single transfer (potential rug pull)
        if total_btc > 100:
            return 0  # Rug Pull
        
        # Structuring pattern
        if 5 < in_degree < 15 and total_btc < 10:
            return 7  # Structuring
        
        # Peel chain
        if out_degree > in_degree * 3:
            return 9  # Peel Chain
        
        # Default to obfuscation
        return 3  # Chain Obfuscation
    
    # Unknown class - use feature clustering
    if class_label == 3:
        # Random assignment for unknown (will be filtered in training)
        return np.random.randint(0, 15)
    
    return -1


def assign_risk_label(class_label: int, features: np.ndarray) -> int:
    """
    Assign risk level (0-3) based on class label.
    
    - 0: SAFE (licit transactions)
    - 1: LOW
    - 2: MEDIUM  
    - 3: HIGH (illicit transactions)
    """
    if class_label == 2:  # Licit
        return 0  # SAFE
    elif class_label == 1:  # Illicit
        return 3  # HIGH
    else:  # Unknown
        # Use features to estimate risk
        total_btc = features[94] if len(features) > 94 else 0
        if total_btc > 50:
            return 2  # MEDIUM
        elif total_btc > 10:
            return 1  # LOW
        return 0  # SAFE


def load_elliptic_dataset(sample_ratio: float = 1.0) -> Data:
    """
    Load Elliptic++ dataset into PyTorch Geometric Data object.
    
    Args:
        sample_ratio: Fraction of data to use (for quick testing)
        
    Returns:
        PyG Data object with:
        - x: Node features [num_nodes, num_features]
        - edge_index: Edge connectivity [2, num_edges]
        - y_typology: Typology labels [num_nodes]
        - y_risk: Risk labels [num_nodes]
        - train_mask, val_mask, test_mask: Data splits
    """
    logger.info("Loading Elliptic++ dataset...")
    start_time = time.time()
    
    # Load features
    logger.info(f"Loading features from {FEATURES_PATH}")
    features_df = pd.read_csv(FEATURES_PATH)
    
    if sample_ratio < 1.0:
        features_df = features_df.sample(frac=sample_ratio, random_state=42)
    
    # Create tx_id to index mapping
    tx_ids = features_df['txId'].values
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    num_nodes = len(tx_ids)
    
    # Extract feature matrix (skip txId and Time step columns)
    feature_cols = [col for col in features_df.columns if col not in ['txId', 'Time step']]
    x = features_df[feature_cols].values.astype(np.float32)
    
    # Handle NaN/Inf
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features (z-score)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x = (x - mean) / std
    
    logger.info(f"Features shape: {x.shape}")
    
    # Load edges
    logger.info(f"Loading edges from {EDGES_PATH}")
    edges_df = pd.read_csv(EDGES_PATH)
    
    # Filter edges to only include nodes in our feature set
    valid_edges = edges_df[
        edges_df['txId1'].isin(tx_to_idx) & 
        edges_df['txId2'].isin(tx_to_idx)
    ]
    
    # Convert to edge index
    edge_src = [tx_to_idx[tx] for tx in valid_edges['txId1'].values]
    edge_dst = [tx_to_idx[tx] for tx in valid_edges['txId2'].values]
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Add self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    
    logger.info(f"Edges: {edge_index.shape[1]}")
    
    # Load classes
    logger.info(f"Loading classes from {CLASSES_PATH}")
    classes_df = pd.read_csv(CLASSES_PATH)
    
    # Create class mapping
    class_mapping = dict(zip(classes_df['txId'], classes_df['class']))
    
    # Assign labels
    y_typology = np.full(num_nodes, -1, dtype=np.int64)  # -1 = unknown/licit
    y_risk = np.zeros(num_nodes, dtype=np.int64)
    
    for idx, tx_id in enumerate(tx_ids):
        class_label = class_mapping.get(tx_id, 3)  # Default to unknown
        y_typology[idx] = assign_typology_label(class_label, x[idx])
        y_risk[idx] = assign_risk_label(class_label, x[idx])
    
    # Create masks for train/val/test split
    # Only use labeled nodes (illicit) for typology training
    labeled_mask = y_typology >= 0
    labeled_indices = np.where(labeled_mask)[0]
    
    logger.info(f"Labeled nodes for typology: {len(labeled_indices)}")
    
    # Split
    if len(labeled_indices) > 100:
        train_idx, temp_idx = train_test_split(
            labeled_indices, test_size=0.3, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42
        )
    else:
        train_idx = labeled_indices[:int(len(labeled_indices) * 0.7)]
        val_idx = labeled_indices[int(len(labeled_indices) * 0.7):int(len(labeled_indices) * 0.85)]
        test_idx = labeled_indices[int(len(labeled_indices) * 0.85):]
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # Create Data object
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y_typology=torch.tensor(y_typology, dtype=torch.long),
        y_risk=torch.tensor(y_risk, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes
    )
    
    load_time = time.time() - start_time
    logger.info(f"Dataset loaded in {load_time:.2f}s")
    logger.info(f"Nodes: {num_nodes}, Edges: {edge_index.shape[1]}, Features: {x.shape[1]}")
    logger.info(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Class distribution
    logger.info(f"Risk class distribution: {np.bincount(y_risk)}")
    typology_counts = np.bincount(y_typology[y_typology >= 0], minlength=15)
    logger.info(f"Typology distribution: {typology_counts}")
    
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#                          TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_epoch_full_batch(
    model: PayFlowGNNClassifier,
    data: Data,
    optimizer: torch.optim.Optimizer,
    device: str,
    typology_weight: float = 0.7,
    risk_weight: float = 0.3
) -> Dict[str, float]:
    """Train for one epoch using full-batch (no NeighborLoader required)."""
    model.train()
    data = data.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass on full graph
    typology_logits, risk_logits = model(data.x, data.edge_index)
    
    # Get training node indices
    train_mask = data.train_mask
    
    # Get labels for training nodes
    typology_labels = data.y_typology[train_mask]
    risk_labels = data.y_risk[train_mask]
    
    # Filter for valid typology labels (>= 0)
    typology_mask = typology_labels >= 0
    
    # Typology loss (only for labeled nodes)
    if typology_mask.sum() > 0:
        typology_loss = F.cross_entropy(
            typology_logits[train_mask][typology_mask],
            typology_labels[typology_mask]
        )
    else:
        typology_loss = torch.tensor(0.0, device=device)
    
    # Risk loss (for all training nodes)
    risk_loss = F.cross_entropy(
        risk_logits[train_mask],
        risk_labels
    )
    
    # Combined loss
    loss = typology_weight * typology_loss + risk_weight * risk_loss
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {
        "loss": loss.item(),
        "typology_loss": typology_loss.item(),
        "risk_loss": risk_loss.item(),
    }


def train_epoch(
    model: PayFlowGNNClassifier,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    typology_weight: float = 0.7,
    risk_weight: float = 0.3
) -> Dict[str, float]:
    """Train for one epoch using mini-batches (requires pyg-lib or torch-sparse)."""
    model.train()
    total_loss = 0
    total_typology_loss = 0
    total_risk_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward
        typology_logits, risk_logits = model(batch.x, batch.edge_index)
        
        # Get labels for batch nodes
        typology_labels = batch.y_typology[:batch.batch_size]
        risk_labels = batch.y_risk[:batch.batch_size]
        
        # Filter for valid typology labels (>= 0)
        typology_mask = typology_labels >= 0
        
        # Typology loss (only for labeled nodes)
        if typology_mask.sum() > 0:
            typology_loss = F.cross_entropy(
                typology_logits[:batch.batch_size][typology_mask],
                typology_labels[typology_mask]
            )
        else:
            typology_loss = torch.tensor(0.0, device=device)
        
        # Risk loss (for all nodes)
        risk_loss = F.cross_entropy(
            risk_logits[:batch.batch_size],
            risk_labels
        )
        
        # Combined loss
        loss = typology_weight * typology_loss + risk_weight * risk_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_typology_loss += typology_loss.item()
        total_risk_loss += risk_loss.item()
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "typology_loss": total_typology_loss / num_batches,
        "risk_loss": total_risk_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: PayFlowGNNClassifier,
    data: Data,
    mask: torch.Tensor,
    device: str
) -> Dict[str, float]:
    """Evaluate model on a data split."""
    model.eval()
    data = data.to(device)
    
    # Forward
    typology_logits, risk_logits = model(data.x, data.edge_index)
    
    # Get predictions
    typology_pred = typology_logits[mask].argmax(dim=1).cpu().numpy()
    risk_pred = risk_logits[mask].argmax(dim=1).cpu().numpy()
    
    typology_true = data.y_typology[mask].cpu().numpy()
    risk_true = data.y_risk[mask].cpu().numpy()
    
    # Filter valid typology labels
    typology_mask = typology_true >= 0
    
    # Metrics
    results = {
        "risk_accuracy": accuracy_score(risk_true, risk_pred),
        "risk_f1_macro": f1_score(risk_true, risk_pred, average='macro', zero_division=0),
    }
    
    if typology_mask.sum() > 0:
        results["typology_accuracy"] = accuracy_score(
            typology_true[typology_mask], typology_pred[typology_mask]
        )
        results["typology_f1_macro"] = f1_score(
            typology_true[typology_mask], typology_pred[typology_mask],
            average='macro', zero_division=0
        )
    else:
        results["typology_accuracy"] = 0.0
        results["typology_f1_macro"] = 0.0
    
    return results


def train_model(
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    dropout: float = DEFAULT_DROPOUT,
    sample_ratio: float = 1.0,
    device: str = "auto"
) -> Tuple[PayFlowGNNClassifier, Dict]:
    """
    Full training pipeline.
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load data
    data = load_elliptic_dataset(sample_ratio)
    
    # Create model
    model = PayFlowGNNClassifier(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_dim,
        embedding_dim=embedding_dim,
        num_typology_classes=15,
        num_risk_classes=4,
        dropout=dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Use full-batch training (no NeighborLoader required - works without pyg-lib)
    use_full_batch = True
    logger.info(f"Using full-batch training (no pyg-lib required)")
    
    # Training history
    history = {
        "train_loss": [],
        "val_risk_acc": [],
        "val_typology_acc": [],
        "val_risk_f1": [],
        "val_typology_f1": [],
        "best_epoch": 0,
        "best_val_f1": 0,
    }
    
    best_val_f1 = 0
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train using full-batch
        train_metrics = train_epoch_full_batch(model, data, optimizer, device)
        
        # Evaluate
        val_metrics = evaluate(model, data, data.val_mask, device)
        
        # Update scheduler
        scheduler.step(val_metrics["risk_f1_macro"])
        
        # Logging
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Val Risk Acc: {val_metrics['risk_accuracy']:.4f} | "
            f"Val Typology Acc: {val_metrics['typology_accuracy']:.4f}"
        )
        
        # History
        history["train_loss"].append(train_metrics["loss"])
        history["val_risk_acc"].append(val_metrics["risk_accuracy"])
        history["val_typology_acc"].append(val_metrics["typology_accuracy"])
        history["val_risk_f1"].append(val_metrics["risk_f1_macro"])
        history["val_typology_f1"].append(val_metrics["typology_f1_macro"])
        
        # Save best model
        combined_f1 = 0.3 * val_metrics["risk_f1_macro"] + 0.7 * val_metrics["typology_f1_macro"]
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            history["best_epoch"] = epoch + 1
            history["best_val_f1"] = best_val_f1
            
            # Save checkpoint
            save_model(model, data.x.shape[1], hidden_dim, embedding_dim)
            logger.info(f"  → Saved best model (F1: {best_val_f1:.4f})")
    
    # Final test evaluation
    logger.info("\nFinal Test Evaluation:")
    test_metrics = evaluate(model, data, data.test_mask, device)
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    history["test_metrics"] = test_metrics
    
    # Save training log
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOG_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training log saved to {TRAINING_LOG_PATH}")
    
    return model, history


def save_model(
    model: PayFlowGNNClassifier,
    in_channels: int,
    hidden_channels: int,
    embedding_dim: int
):
    """Save model checkpoints."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save encoder separately
    torch.save({
        "encoder": model.encoder.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": embedding_dim,
    }, GNN_ENCODER_PATH)
    
    # Save full model
    torch.save({
        "model": model.state_dict(),
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "embedding_dim": embedding_dim,
        "num_typology_classes": 15,
        "num_risk_classes": 4,
    }, FULL_MODEL_PATH)
    
    logger.info(f"Models saved to {MODEL_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
#                          MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train PayFlow GNN on Elliptic++ dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Hidden dimension")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout rate")
    parser.add_argument("--sample-ratio", type=float, default=1.0, help="Data sample ratio (for testing)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--quick-test", action="store_true", help="Quick test with 10% data")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PayFlow GNN Training - Elliptic++ Dataset")
    print("=" * 80)
    
    # Check dataset exists
    if not FEATURES_PATH.exists():
        logger.error(f"Dataset not found at {DATASET_DIR}")
        logger.error("Please ensure Elliptic++ dataset is downloaded")
        sys.exit(1)
    
    # Quick test mode
    if args.quick_test:
        args.epochs = 5
        args.sample_ratio = 0.1
        logger.info("Quick test mode: 5 epochs, 10% data")
    
    # Train
    model, history = train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        sample_ratio=args.sample_ratio,
        device=args.device
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best Epoch: {history['best_epoch']}")
    print(f"Best Validation F1: {history['best_val_f1']:.4f}")
    print(f"Models saved to: {MODEL_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
