"""
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║           PayFlow Graph Neural Network - Fused with Local Neural Network                  ║
║               GraphSAGE + MLP Fusion for Enhanced Fraud Detection                         ║
║                  100% Offline - No External API Dependencies                              ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝

This module FUSES Graph Neural Network (GraphSAGE) with the existing MLP/Autoencoder
architecture in localNeuralNetwork.py - NOT as a separate layer, but as enhanced 
feature extraction that enriches the existing 13-feature input.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PayFlow Fused Neural Architecture                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Transaction Input                                                              │
│        │                                                                         │
│        ├───────────────────┬────────────────────────────────────────────────────│
│        │                   │                                                     │
│   ┌────▼────┐         ┌────▼────┐                                               │
│   │  13     │         │  Graph  │                                               │
│   │Standard │         │ Context │ ← Neighbor transaction graph                   │
│   │Features │         │  Data   │                                               │
│   └────┬────┘         └────┬────┘                                               │
│        │                   │                                                     │
│        │              ┌────▼────┐                                               │
│        │              │GraphSAGE│                                               │
│        │              │3-Layer  │ ← SAGEConv(in→256→256→64)                     │
│        │              └────┬────┘                                               │
│        │                   │                                                     │
│        │              ┌────▼────┐                                               │
│        │              │  64-dim │ ← Graph Embeddings                            │
│        │              │Embedding│                                               │
│        │              └────┬────┘                                               │
│        │                   │                                                     │
│   ┌────▼───────────────────▼────┐                                               │
│   │      Fused Feature Vector    │ ← 13 + 64 = 77 features                      │
│   │        (Concatenation)       │                                               │
│   └──────────────┬───────────────┘                                               │
│                  │                                                               │
│           ┌──────▼──────┐                                                        │
│           │  FusedMLP   │ ← Enhanced MLP: 77→128→64→32→4                        │
│           └──────┬──────┘                                                        │
│                  │                                                               │
│    ┌─────────────┴─────────────┐                                                │
│    │                           │                                                 │
│  ┌─▼──┐                   ┌────▼─────┐                                          │
│  │Risk│                   │15-Class  │                                          │
│  │4cls│                   │Typology  │                                          │
│  └────┘                   └──────────┘                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Key Innovation:
- Graph embeddings ENRICH existing features, don't replace them
- Maintains backward compatibility with non-graph transactions
- Fallback to standard features when graph context unavailable
- Pre-trained on Elliptic++ dataset (203K Bitcoin transactions)
"""

import os
import sys
import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# PyTorch Geometric imports with fallback
try:
    from torch_geometric.nn import SAGEConv, GATConv, GCNConv
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import NeighborLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    SAGEConv = None
    Data = None

# Configure logging
logger = logging.getLogger("PayFlow.GNN")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# ═══════════════════════════════════════════════════════════════════════════════
#                          CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# 15 Fraud Typologies - Matching FraudTypologyDetector exactly
class GNNFraudTypology(Enum):
    """15 Fraud typologies with market impact for GNN classification."""
    RUG_PULL = (0, "Rug Pull", 8000)          # $8B market
    PIG_BUTCHERING = (1, "Pig Butchering", 4000)
    MIXER_TUMBLING = (2, "Mixer/Tumbling", 2500)
    CHAIN_OBFUSCATION = (3, "Chain Obfuscation", 1500)
    FAKE_TOKEN = (4, "Fake Token", 1200)
    FLASH_LOAN = (5, "Flash Loan Attack", 800)
    WASH_TRADING = (6, "Wash Trading", 600)
    STRUCTURING = (7, "Structuring/Smurfing", 500)
    VELOCITY_ATTACK = (8, "Velocity Attack", 400)
    PEEL_CHAIN = (9, "Peel Chain", 350)
    DUSTING = (10, "Dusting Attack", 300)
    ADDRESS_POISONING = (11, "Address Poisoning", 250)
    APPROVAL_EXPLOIT = (12, "Approval Exploit", 200)
    SIM_SWAP = (13, "SIM Swap Related", 200)
    ROMANCE_SCAM = (14, "Romance Scam Pattern", 200)
    
    def __init__(self, idx: int, name: str, market_impact_m: int):
        self.idx = idx
        self.display_name = name
        self.market_impact_m = market_impact_m

# Model paths
MODEL_DIR = Path(__file__).parent / "models" / "gnn"
PRETRAINED_GNN_PATH = MODEL_DIR / "gnn_graphsage_elliptic.pt"
FUSED_MLP_PATH = MODEL_DIR / "fused_mlp.pt"

# Feature dimensions
STANDARD_FEATURES = 13      # From localNeuralNetwork.py
GNN_EMBEDDING_DIM = 64      # GraphSAGE output dimension
FUSED_FEATURES = STANDARD_FEATURES + GNN_EMBEDDING_DIM  # 77 total

# ═══════════════════════════════════════════════════════════════════════════════
#                          GRAPH SAGE ENCODER
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE and TORCH_GEOMETRIC_AVAILABLE:
    
    class GraphSAGEEncoder(nn.Module):
        """
        GraphSAGE encoder for learning node embeddings from transaction graphs.
        
        Architecture:
        - 3 SAGEConv layers with BatchNorm and Dropout
        - Aggregates neighbor information using mean aggregation
        - Outputs 64-dimensional embeddings for fusion with MLP
        """
        
        def __init__(
            self,
            in_channels: int = 184,      # Elliptic++ has 184 features
            hidden_channels: int = 256,
            out_channels: int = 64,      # Embedding dimension for fusion
            num_layers: int = 3,
            dropout: float = 0.3
        ):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            
            # Layer 1: Input → Hidden
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            
            # Layer 2: Hidden → Hidden
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            
            # Layer 3: Hidden → Output (Embedding)
            self.conv3 = SAGEConv(hidden_channels, out_channels)
            self.bn3 = nn.BatchNorm1d(out_channels)
            
            # Layer normalization for stable training
            self.layer_norm = nn.LayerNorm(out_channels)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through GraphSAGE layers.
            
            Args:
                x: Node features [num_nodes, in_channels]
                edge_index: Edge connectivity [2, num_edges]
                
            Returns:
                Node embeddings [num_nodes, out_channels]
            """
            # Layer 1
            h = self.conv1(x, edge_index)
            h = self.bn1(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Layer 2
            h = self.conv2(h, edge_index)
            h = self.bn2(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # Layer 3 (Embedding layer)
            h = self.conv3(h, edge_index)
            h = self.bn3(h)
            h = F.relu(h)
            
            # Final layer normalization
            h = self.layer_norm(h)
            
            return h
        
        def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Get normalized embeddings for fusion."""
            with torch.no_grad():
                embeddings = self.forward(x, edge_index)
                # L2 normalize for stable fusion
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            return embeddings


    class FusedFraudMLP(nn.Module):
        """
        Fused MLP that takes CONCATENATED standard features + GNN embeddings.
        
        Input: [13 standard features] + [64 GNN embeddings] = 77 features
        Output: 
            - risk_level: 4-class (SAFE, LOW, MEDIUM, HIGH/CRITICAL)
            - typology: 15-class fraud typology
        """
        
        def __init__(
            self,
            in_features: int = FUSED_FEATURES,  # 77
            hidden_dims: List[int] = [128, 64, 32],
            num_risk_classes: int = 4,
            num_typology_classes: int = 15,
            dropout: float = 0.3
        ):
            super().__init__()
            
            # Shared feature extraction backbone
            self.backbone = nn.Sequential(
                nn.Linear(in_features, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.BatchNorm1d(hidden_dims[2]),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Less dropout in final layers
            )
            
            # Risk level classification head (4 classes)
            self.risk_head = nn.Sequential(
                nn.Linear(hidden_dims[2], 16),
                nn.ReLU(),
                nn.Linear(16, num_risk_classes)
            )
            
            # Typology classification head (15 classes)
            self.typology_head = nn.Sequential(
                nn.Linear(hidden_dims[2], 32),
                nn.ReLU(),
                nn.Linear(32, num_typology_classes)
            )
            
            # Initialize weights
            self._init_weights()
            
        def _init_weights(self):
            """Initialize weights with Xavier/He initialization."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm1d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass returning both risk level and typology predictions.
            
            Args:
                x: Fused features [batch, 77]
                
            Returns:
                risk_logits: [batch, 4]
                typology_logits: [batch, 15]
            """
            # Shared backbone
            features = self.backbone(x)
            
            # Multi-task heads
            risk_logits = self.risk_head(features)
            typology_logits = self.typology_head(features)
            
            return risk_logits, typology_logits
        
        def predict_risk(self, x: torch.Tensor) -> Tuple[int, float, np.ndarray]:
            """
            Predict risk level with confidence.
            
            Returns:
                risk_level: 0-3 (SAFE, LOW, MEDIUM, HIGH)
                confidence: float 0-1
                probabilities: array of class probabilities
            """
            with torch.no_grad():
                risk_logits, _ = self.forward(x)
                probs = F.softmax(risk_logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                confidence = probs.max(dim=-1).values
                
            return pred.item(), confidence.item(), probs.cpu().numpy().flatten()
        
        def predict_typology(self, x: torch.Tensor) -> Tuple[int, float, np.ndarray]:
            """
            Predict fraud typology with confidence.
            
            Returns:
                typology_idx: 0-14 (fraud typology index)
                confidence: float 0-1
                probabilities: array of class probabilities
            """
            with torch.no_grad():
                _, typology_logits = self.forward(x)
                probs = F.softmax(typology_logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                confidence = probs.max(dim=-1).values
                
            return pred.item(), confidence.item(), probs.cpu().numpy().flatten()


    class PayFlowGNNClassifier(nn.Module):
        """
        Full GNN classifier for 15-class fraud typology detection.
        
        This is trained on Elliptic++ data and provides typology predictions
        that are fused with the local neural network.
        """
        
        def __init__(
            self,
            in_channels: int = 184,
            hidden_channels: int = 256,
            num_classes: int = 15
        ):
            super().__init__()
            
            # GraphSAGE layers
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)
            
            self.conv3 = SAGEConv(hidden_channels, num_classes)
            
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Forward pass returning class logits."""
            h = self.conv1(x, edge_index)
            h = self.bn1(h)
            h = F.relu(h)
            h = self.dropout(h)
            
            h = self.conv2(h, edge_index)
            h = self.bn2(h)
            h = F.relu(h)
            h = self.dropout(h)
            
            h = self.conv3(h, edge_index)
            
            return h
        
        def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[int, float, np.ndarray]:
            """Predict fraud typology for single transaction."""
            with torch.no_grad():
                logits = self.forward(x, edge_index)
                probs = F.softmax(logits, dim=-1)
                pred = torch.argmax(probs, dim=-1)
                confidence = probs.max(dim=-1).values
                
            return pred[0].item(), confidence[0].item(), probs[0].cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
#                          GRAPH CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionNode:
    """Represents a transaction in the graph."""
    tx_id: str
    address: str
    amount: float
    timestamp: float
    features: np.ndarray = field(default_factory=lambda: np.zeros(13))
    neighbors: List[str] = field(default_factory=list)
    edge_weights: List[float] = field(default_factory=list)


class TransactionGraph:
    """
    Dynamic transaction graph that builds neighbor relationships.
    
    This graph is built incrementally as transactions are processed,
    enabling GNN inference on new transactions by connecting them
    to existing transaction history.
    """
    
    def __init__(self, max_nodes: int = 100000):
        self.max_nodes = max_nodes
        self.nodes: Dict[str, TransactionNode] = {}
        self.address_to_txs: Dict[str, List[str]] = {}  # Maps address → tx_ids
        self.edge_index: List[Tuple[int, int]] = []
        self.node_id_map: Dict[str, int] = {}  # tx_id → numerical index
        self.next_node_id = 0
        
    def add_transaction(
        self,
        tx_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: float,
        features: np.ndarray
    ) -> int:
        """
        Add transaction to graph and create edges to related transactions.
        
        Returns the node index for GNN inference.
        """
        # Create node
        node = TransactionNode(
            tx_id=tx_id,
            address=sender,
            amount=amount,
            timestamp=timestamp,
            features=features
        )
        
        # Assign numerical index
        node_idx = self.next_node_id
        self.node_id_map[tx_id] = node_idx
        self.next_node_id += 1
        
        # Store node
        self.nodes[tx_id] = node
        
        # Update address mapping
        sender_lower = sender.lower()
        recipient_lower = recipient.lower()
        
        if sender_lower not in self.address_to_txs:
            self.address_to_txs[sender_lower] = []
        if recipient_lower not in self.address_to_txs:
            self.address_to_txs[recipient_lower] = []
        
        # Create edges to other transactions involving same addresses
        # This captures the transaction flow pattern
        for related_tx in self.address_to_txs.get(sender_lower, [])[-20:]:  # Last 20 txs
            if related_tx in self.node_id_map:
                related_idx = self.node_id_map[related_tx]
                self.edge_index.append((node_idx, related_idx))
                self.edge_index.append((related_idx, node_idx))  # Undirected
                node.neighbors.append(related_tx)
        
        for related_tx in self.address_to_txs.get(recipient_lower, [])[-20:]:
            if related_tx in self.node_id_map:
                related_idx = self.node_id_map[related_tx]
                self.edge_index.append((node_idx, related_idx))
                self.edge_index.append((related_idx, node_idx))
                node.neighbors.append(related_tx)
        
        # Add to address index
        self.address_to_txs[sender_lower].append(tx_id)
        self.address_to_txs[recipient_lower].append(tx_id)
        
        # Prune if too large
        if len(self.nodes) > self.max_nodes:
            self._prune_old_nodes()
        
        return node_idx
    
    def _prune_old_nodes(self):
        """Remove oldest nodes to stay under max_nodes limit."""
        # Sort by timestamp and remove oldest 10%
        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1].timestamp
        )
        remove_count = len(sorted_nodes) // 10
        
        for tx_id, _ in sorted_nodes[:remove_count]:
            del self.nodes[tx_id]
            # Note: We don't remove from edge_index for simplicity
            # Edges to removed nodes will be ignored during inference
    
    def get_neighborhood_subgraph(
        self,
        tx_id: str,
        num_hops: int = 2,
        max_neighbors: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Extract k-hop neighborhood subgraph around a transaction.
        
        Returns:
            features: [num_nodes, feature_dim] node feature matrix
            edge_index: [2, num_edges] edge connectivity
            center_idx: index of center node in the subgraph
        """
        if tx_id not in self.nodes:
            # Return empty/default if transaction not in graph
            return np.zeros((1, 13)), np.zeros((2, 0), dtype=np.int64), 0
        
        # BFS to find neighborhood
        visited = {tx_id: 0}  # tx_id → new local index
        frontier = [tx_id]
        local_idx = 1
        
        for hop in range(num_hops):
            next_frontier = []
            for current_tx in frontier:
                if current_tx not in self.nodes:
                    continue
                for neighbor_tx in self.nodes[current_tx].neighbors[:max_neighbors // num_hops]:
                    if neighbor_tx not in visited and neighbor_tx in self.nodes:
                        visited[neighbor_tx] = local_idx
                        next_frontier.append(neighbor_tx)
                        local_idx += 1
                        if local_idx >= max_neighbors:
                            break
                if local_idx >= max_neighbors:
                    break
            frontier = next_frontier
            if local_idx >= max_neighbors:
                break
        
        # Build feature matrix
        features = np.zeros((len(visited), 13))
        for tx, idx in visited.items():
            if tx in self.nodes:
                features[idx] = self.nodes[tx].features
        
        # Build local edge index
        edges = []
        for src_tx, src_idx in visited.items():
            if src_tx not in self.nodes:
                continue
            for neighbor_tx in self.nodes[src_tx].neighbors:
                if neighbor_tx in visited:
                    dst_idx = visited[neighbor_tx]
                    edges.append([src_idx, dst_idx])
        
        edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
        
        return features, edge_index, 0  # Center node is always index 0


# ═══════════════════════════════════════════════════════════════════════════════
#                          GNN PREDICTION RESULT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GNNPrediction:
    """Result from GNN-fused prediction."""
    # Risk prediction
    risk_level: int              # 0-3
    risk_level_name: str         # SAFE, LOW, MEDIUM, HIGH
    risk_confidence: float       # 0-1
    risk_probabilities: np.ndarray
    
    # Typology prediction
    primary_typology: int        # 0-14
    typology_name: str
    typology_confidence: float
    typology_probabilities: np.ndarray
    top_3_typologies: List[Tuple[str, float]]
    
    # Graph context
    num_neighbors: int
    neighborhood_density: float  # Edge density in subgraph
    has_graph_context: bool      # True if meaningful graph exists
    
    # Performance
    inference_time_ms: float
    
    # Graph embeddings (optional - may be None)
    graph_embedding: np.ndarray = None
    embedding_dimension: int = 64


# ═══════════════════════════════════════════════════════════════════════════════
#                          MAIN GNN ENGINE (FUSED)
# ═══════════════════════════════════════════════════════════════════════════════

class FusedGNNEngine:
    """
    Main engine that FUSES GNN embeddings with the existing neural network.
    
    This is NOT a separate layer - it ENRICHES the 13 standard features
    with 64 graph-based features, creating a 77-dimensional fused representation.
    
    The fused representation is then processed by FusedFraudMLP which
    outputs both risk levels and fraud typologies.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the fused GNN engine.
        
        Args:
            device: 'cuda', 'cpu', or 'auto'
        """
        self.enabled = False
        self.device = self._get_device(device)
        
        # Components
        self.graph_encoder: Optional[GraphSAGEEncoder] = None
        self.fused_mlp: Optional[FusedFraudMLP] = None
        self.gnn_classifier: Optional[PayFlowGNNClassifier] = None
        self.transaction_graph: Optional[TransactionGraph] = None
        self.pretrained_encoder: Optional[GraphSAGEEncoder] = None  # Elliptic++ trained
        self.has_pretrained_encoder = False
        
        # Statistics
        self.stats = {
            "total_predictions": 0,
            "graph_enriched_predictions": 0,
            "fallback_predictions": 0,
            "avg_inference_ms": 0,
            "model_loaded": False,
            "pretrained_loaded": False
        }
        
        # Initialize
        self._initialize()
        
    def _get_device(self, device: str) -> str:
        """Determine computation device."""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device
    
    def _initialize(self):
        """Initialize GNN components."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. GNN features disabled.")
            return
            
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.warning("PyTorch Geometric not available. GNN features disabled.")
            return
        
        try:
            # Initialize graph encoder (for embeddings)
            self.graph_encoder = GraphSAGEEncoder(
                in_channels=13,  # Match our standard features
                hidden_channels=256,
                out_channels=GNN_EMBEDDING_DIM,  # 64
            ).to(self.device)
            
            # Initialize fused MLP
            self.fused_mlp = FusedFraudMLP(
                in_features=FUSED_FEATURES,  # 77
                hidden_dims=[128, 64, 32],
                num_risk_classes=4,
                num_typology_classes=15
            ).to(self.device)
            
            # Initialize transaction graph
            self.transaction_graph = TransactionGraph(max_nodes=100000)
            
            # Try to load pre-trained weights
            self._load_pretrained_weights()
            
            self.enabled = True
            self.stats["model_loaded"] = True
            
            logger.info(f"FusedGNNEngine initialized on {self.device}")
            logger.info(f"Graph Encoder: {sum(p.numel() for p in self.graph_encoder.parameters())} params")
            logger.info(f"Fused MLP: {sum(p.numel() for p in self.fused_mlp.parameters())} params")
            
        except Exception as e:
            logger.error(f"Failed to initialize GNN engine: {e}")
            self.enabled = False
    
    def _load_pretrained_weights(self):
        """Load pre-trained model weights if available."""
        try:
            # Ensure model directory exists
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            
            if PRETRAINED_GNN_PATH.exists():
                checkpoint = torch.load(PRETRAINED_GNN_PATH, map_location=self.device, weights_only=False)
                # Check if model dimensions match
                if checkpoint.get("in_channels") == 13:
                    self.graph_encoder.load_state_dict(checkpoint["encoder"])
                    logger.info("Loaded pre-trained GraphSAGE encoder (13-dim input)")
                else:
                    # Model was trained on Elliptic++ (182 features) - create matching encoder
                    in_ch = checkpoint.get("in_channels", 182)
                    hidden_ch = checkpoint.get("hidden_channels", 256)
                    out_ch = checkpoint.get("out_channels", GNN_EMBEDDING_DIM)
                    
                    # Create encoder matching the checkpoint
                    self.pretrained_encoder = GraphSAGEEncoder(
                        in_channels=in_ch,
                        hidden_channels=hidden_ch,
                        out_channels=out_ch,
                    ).to(self.device)
                    self.pretrained_encoder.load_state_dict(checkpoint["encoder"])
                    logger.info(f"Loaded pre-trained GraphSAGE encoder ({in_ch}-dim input, Elliptic++ trained)")
                    
                    # Flag that we have the pretrained model available
                    self.has_pretrained_encoder = True
                
            if FUSED_MLP_PATH.exists():
                checkpoint = torch.load(FUSED_MLP_PATH, map_location=self.device, weights_only=False)
                self.fused_mlp.load_state_dict(checkpoint["fused_mlp"])
                logger.info("Loaded pre-trained Fused MLP")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {e}")
            logger.info("Using randomly initialized weights (train with Elliptic++ dataset)")
    
    def get_graph_embedding(
        self,
        features: np.ndarray,
        tx_id: str
    ) -> np.ndarray:
        """
        Get 64-dimensional graph embedding for a transaction.
        
        If the transaction has graph context (neighbors), returns
        meaningful embedding. Otherwise returns zero embedding.
        """
        if not self.enabled or self.transaction_graph is None:
            return np.zeros(GNN_EMBEDDING_DIM)
        
        try:
            # Get neighborhood subgraph
            node_features, edge_index, center_idx = \
                self.transaction_graph.get_neighborhood_subgraph(tx_id)
            
            # Check if we have meaningful graph context
            if edge_index.shape[1] < 2:
                # No edges - return zero embedding (fallback to standard features)
                return np.zeros(GNN_EMBEDDING_DIM)
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            edges = torch.tensor(edge_index, dtype=torch.long).to(self.device)
            
            # Get embedding
            self.graph_encoder.eval()
            embeddings = self.graph_encoder.get_embedding(x, edges)
            
            # Return center node embedding
            return embeddings[center_idx].cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Error computing graph embedding: {e}")
            return np.zeros(GNN_EMBEDDING_DIM)
    
    def predict(
        self,
        tx_id: str,
        sender: str,
        recipient: str,
        amount: float,
        timestamp: float,
        standard_features: np.ndarray  # 13 features from LocalNeuralNetworkEngine
    ) -> GNNPrediction:
        """
        Make fused GNN prediction by enriching standard features with graph embeddings.
        
        This is the main inference method that:
        1. Adds transaction to graph
        2. Computes graph embedding (if neighbors exist)
        3. Concatenates [13 standard] + [64 graph] = 77 features
        4. Runs through FusedMLP for risk + typology prediction
        """
        start_time = time.time()
        
        # Fallback if not enabled
        if not self.enabled:
            return self._create_fallback_prediction(standard_features, start_time)
        
        try:
            # Add transaction to graph
            self.transaction_graph.add_transaction(
                tx_id=tx_id,
                sender=sender,
                recipient=recipient,
                amount=amount,
                timestamp=timestamp,
                features=standard_features
            )
            
            # Get graph embedding
            graph_embedding = self.get_graph_embedding(standard_features, tx_id)
            has_graph_context = np.any(graph_embedding != 0)
            
            # FUSION: Concatenate standard features + graph embedding
            fused_features = np.concatenate([standard_features, graph_embedding])
            
            # Convert to tensor
            x = torch.tensor(fused_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Predict
            self.fused_mlp.eval()
            risk_level, risk_conf, risk_probs = self.fused_mlp.predict_risk(x)
            typology_idx, typology_conf, typology_probs = self.fused_mlp.predict_typology(x)
            
            # Get typology name
            typology_name = list(GNNFraudTypology)[typology_idx].display_name if typology_idx < 15 else "Unknown"
            
            # Get top 3 typologies
            top_indices = np.argsort(typology_probs)[-3:][::-1]
            top_3 = [
                (list(GNNFraudTypology)[i].display_name if i < 15 else "Unknown", float(typology_probs[i]))
                for i in top_indices
            ]
            
            # Risk level name
            risk_names = ["SAFE", "LOW", "MEDIUM", "HIGH"]
            risk_name = risk_names[risk_level] if risk_level < 4 else "UNKNOWN"
            
            # Graph stats
            num_neighbors = len(self.transaction_graph.nodes.get(tx_id, TransactionNode("", "", 0, 0)).neighbors)
            
            # Update stats
            inference_time = (time.time() - start_time) * 1000
            self.stats["total_predictions"] += 1
            if has_graph_context:
                self.stats["graph_enriched_predictions"] += 1
            else:
                self.stats["fallback_predictions"] += 1
            self.stats["avg_inference_ms"] = (
                self.stats["avg_inference_ms"] * (self.stats["total_predictions"] - 1) + inference_time
            ) / self.stats["total_predictions"]
            
            return GNNPrediction(
                graph_embedding=graph_embedding,
                embedding_dimension=GNN_EMBEDDING_DIM,
                risk_level=risk_level,
                risk_level_name=risk_name,
                risk_confidence=risk_conf,
                risk_probabilities=risk_probs,
                primary_typology=typology_idx,
                typology_name=typology_name,
                typology_confidence=typology_conf,
                typology_probabilities=typology_probs,
                top_3_typologies=top_3,
                num_neighbors=num_neighbors,
                neighborhood_density=num_neighbors / 50 if num_neighbors > 0 else 0,  # Normalized
                has_graph_context=has_graph_context,
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            logger.error(f"GNN prediction error: {e}")
            return self._create_fallback_prediction(standard_features, start_time)
    
    def _create_fallback_prediction(
        self,
        standard_features: np.ndarray,
        start_time: float
    ) -> GNNPrediction:
        """Create fallback prediction when GNN is not available."""
        inference_time = (time.time() - start_time) * 1000
        
        return GNNPrediction(
            graph_embedding=np.zeros(GNN_EMBEDDING_DIM),
            embedding_dimension=GNN_EMBEDDING_DIM,
            risk_level=0,
            risk_level_name="SAFE",
            risk_confidence=0.5,
            risk_probabilities=np.array([0.7, 0.2, 0.08, 0.02]),
            primary_typology=0,
            typology_name="Unknown",
            typology_confidence=0.0,
            typology_probabilities=np.ones(15) / 15,
            top_3_typologies=[("Unknown", 0.0)] * 3,
            num_neighbors=0,
            neighborhood_density=0.0,
            has_graph_context=False,
            inference_time_ms=inference_time
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self.stats,
            "enabled": self.enabled,
            "device": self.device,
            "graph_nodes": len(self.transaction_graph.nodes) if self.transaction_graph else 0,
            "torch_available": TORCH_AVAILABLE,
            "torch_geometric_available": TORCH_GEOMETRIC_AVAILABLE,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                          GLOBAL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

# Singleton instance
_gnn_engine: Optional[FusedGNNEngine] = None

def get_gnn_engine() -> FusedGNNEngine:
    """Get or create the global GNN engine instance."""
    global _gnn_engine
    if _gnn_engine is None:
        _gnn_engine = FusedGNNEngine()
    return _gnn_engine


# ═══════════════════════════════════════════════════════════════════════════════
#                          TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("PayFlow Graph Neural Network - Fused Architecture Test")
    print("=" * 80)
    
    # Initialize engine
    engine = get_gnn_engine()
    print(f"\nGNN Engine Status: {'ENABLED' if engine.enabled else 'DISABLED'}")
    print(f"Device: {engine.device}")
    print(f"Stats: {json.dumps(engine.get_stats(), indent=2)}")
    
    if engine.enabled:
        # Test prediction
        test_features = np.random.randn(13)  # Random 13 features
        
        for i in range(5):
            tx_id = f"test_tx_{i}"
            prediction = engine.predict(
                tx_id=tx_id,
                sender="0x1234567890abcdef",
                recipient="0xfedcba0987654321",
                amount=1000 + i * 100,
                timestamp=time.time() + i,
                standard_features=test_features
            )
            
            print(f"\n--- Prediction {i+1} ---")
            print(f"Risk Level: {prediction.risk_level_name} (conf: {prediction.risk_confidence:.2f})")
            print(f"Primary Typology: {prediction.typology_name} (conf: {prediction.typology_confidence:.2f})")
            print(f"Top 3 Typologies: {prediction.top_3_typologies}")
            print(f"Graph Context: {prediction.has_graph_context}")
            print(f"Neighbors: {prediction.num_neighbors}")
            print(f"Inference Time: {prediction.inference_time_ms:.2f}ms")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
