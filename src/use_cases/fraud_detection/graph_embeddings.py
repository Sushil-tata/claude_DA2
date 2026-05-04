"""
Graph Embeddings for Fraud Detection

Implements node embedding methods for fraud network analysis:
- Node2Vec random walk embeddings
- DeepWalk embeddings
- GraphSAGE embeddings (when available)
- Community detection integration
- Embedding quality metrics
- Temporal graph embeddings for evolving networks

Example:
    >>> from src.use_cases.fraud_detection.graph_embeddings import Node2VecEmbeddings
    >>> embedder = Node2VecEmbeddings(dimensions=64)
    >>> embeddings = embedder.fit_transform(graph)
    >>> quality = embedder.evaluate_quality(graph, embeddings)
"""

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict

from loguru import logger
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    logger.warning("node2vec not available. Install with: pip install node2vec")
    NODE2VEC_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")
    TORCH_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


class BaseGraphEmbedding(ABC):
    """Base class for graph embedding methods."""
    
    def __init__(self, dimensions: int = 64):
        """
        Initialize graph embedding.
        
        Args:
            dimensions: Embedding dimension
        """
        self.dimensions = dimensions
        self.embeddings_: Optional[np.ndarray] = None
        self.node_to_idx_: Optional[Dict[Any, int]] = None
        self.idx_to_node_: Optional[Dict[int, Any]] = None
    
    @abstractmethod
    def fit(self, graph: nx.Graph) -> 'BaseGraphEmbedding':
        """Fit embedding model."""
        pass
    
    @abstractmethod
    def transform(self, nodes: Optional[List[Any]] = None) -> np.ndarray:
        """Transform nodes to embeddings."""
        pass
    
    def fit_transform(self, graph: nx.Graph) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(graph)
        return self.transform()
    
    def get_embedding(self, node: Any) -> Optional[np.ndarray]:
        """
        Get embedding for a single node.
        
        Args:
            node: Node identifier
            
        Returns:
            Node embedding or None if not found
        """
        if self.node_to_idx_ is None or self.embeddings_ is None:
            return None
        
        idx = self.node_to_idx_.get(node)
        if idx is None:
            return None
        
        return self.embeddings_[idx]
    
    def save(self, path: Union[str, Path]):
        """Save embeddings to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'dimensions': self.dimensions,
            'embeddings': self.embeddings_,
            'node_to_idx': self.node_to_idx_,
            'idx_to_node': self.idx_to_node_
        }
        
        joblib.dump(data, path)
        logger.info(f"Embeddings saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseGraphEmbedding':
        """Load embeddings from disk."""
        data = joblib.load(path)
        
        embedder = cls(dimensions=data['dimensions'])
        embedder.embeddings_ = data['embeddings']
        embedder.node_to_idx_ = data['node_to_idx']
        embedder.idx_to_node_ = data['idx_to_node']
        
        logger.info(f"Embeddings loaded from {path}")
        return embedder


class Node2VecEmbeddings(BaseGraphEmbedding):
    """
    Node2Vec embeddings using biased random walks.
    
    Uses random walks with parameterized sampling to learn node embeddings
    that preserve network structure.
    
    Example:
        >>> embedder = Node2VecEmbeddings(
        ...     dimensions=64,
        ...     walk_length=30,
        ...     num_walks=200,
        ...     p=1,  # Return parameter
        ...     q=1   # In-out parameter
        ... )
        >>> embeddings = embedder.fit_transform(graph)
    """
    
    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 30,
        num_walks: int = 200,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 4,
        window: int = 10,
        min_count: int = 1,
        batch_words: int = 4
    ):
        """
        Initialize Node2Vec embeddings.
        
        Args:
            dimensions: Embedding dimension
            walk_length: Length of random walk
            num_walks: Number of walks per node
            p: Return parameter (probability of returning to previous node)
            q: In-out parameter (BFS vs DFS)
            workers: Number of parallel workers
            window: Context window size
            min_count: Minimum word count
            batch_words: Batch size for word2vec
        """
        super().__init__(dimensions)
        
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec is required. Install with: pip install node2vec")
        
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        self.model = None
        
        logger.info("Initialized Node2VecEmbeddings")
    
    def fit(self, graph: nx.Graph) -> 'Node2VecEmbeddings':
        """
        Fit Node2Vec model.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Self
        """
        logger.info(f"Fitting Node2Vec on graph with {graph.number_of_nodes()} nodes")
        
        # Create Node2Vec model
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
            quiet=True
        )
        
        # Train model
        self.model = node2vec.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=self.batch_words
        )
        
        # Extract embeddings
        nodes = list(graph.nodes())
        self.node_to_idx_ = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node_ = {i: node for i, node in enumerate(nodes)}
        
        self.embeddings_ = np.array([
            self.model.wv[str(node)] for node in nodes
        ])
        
        logger.info("Node2Vec fitting completed")
        return self
    
    def transform(self, nodes: Optional[List[Any]] = None) -> np.ndarray:
        """
        Transform nodes to embeddings.
        
        Args:
            nodes: List of nodes (if None, returns all embeddings)
            
        Returns:
            Node embeddings
        """
        if self.embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if nodes is None:
            return self.embeddings_
        
        embeddings = []
        for node in nodes:
            emb = self.get_embedding(node)
            if emb is not None:
                embeddings.append(emb)
            else:
                # Return zero vector for unknown nodes
                embeddings.append(np.zeros(self.dimensions))
        
        return np.array(embeddings)


class DeepWalkEmbeddings(BaseGraphEmbedding):
    """
    DeepWalk embeddings using uniform random walks.
    
    Simpler version of Node2Vec with uniform random walks (p=1, q=1).
    
    Example:
        >>> embedder = DeepWalkEmbeddings(
        ...     dimensions=64,
        ...     walk_length=40,
        ...     num_walks=80
        ... )
        >>> embeddings = embedder.fit_transform(graph)
    """
    
    def __init__(
        self,
        dimensions: int = 64,
        walk_length: int = 40,
        num_walks: int = 80,
        workers: int = 4,
        window: int = 5,
        min_count: int = 1
    ):
        """
        Initialize DeepWalk embeddings.
        
        Args:
            dimensions: Embedding dimension
            walk_length: Length of random walk
            num_walks: Number of walks per node
            workers: Number of parallel workers
            window: Context window size
            min_count: Minimum word count
        """
        super().__init__(dimensions)
        
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec is required. Install with: pip install node2vec")
        
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.model = None
        
        logger.info("Initialized DeepWalkEmbeddings")
    
    def fit(self, graph: nx.Graph) -> 'DeepWalkEmbeddings':
        """
        Fit DeepWalk model.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Self
        """
        logger.info(f"Fitting DeepWalk on graph with {graph.number_of_nodes()} nodes")
        
        # DeepWalk is Node2Vec with p=1, q=1
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=1.0,
            q=1.0,
            workers=self.workers,
            quiet=True
        )
        
        self.model = node2vec.fit(
            window=self.window,
            min_count=self.min_count
        )
        
        # Extract embeddings
        nodes = list(graph.nodes())
        self.node_to_idx_ = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node_ = {i: node for i, node in enumerate(nodes)}
        
        self.embeddings_ = np.array([
            self.model.wv[str(node)] for node in nodes
        ])
        
        logger.info("DeepWalk fitting completed")
        return self
    
    def transform(self, nodes: Optional[List[Any]] = None) -> np.ndarray:
        """
        Transform nodes to embeddings.
        
        Args:
            nodes: List of nodes (if None, returns all embeddings)
            
        Returns:
            Node embeddings
        """
        if self.embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if nodes is None:
            return self.embeddings_
        
        embeddings = []
        for node in nodes:
            emb = self.get_embedding(node)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.dimensions))
        
        return np.array(embeddings)


class GraphSAGEEmbeddings(BaseGraphEmbedding):
    """
    GraphSAGE embeddings using neighborhood aggregation.
    
    Neural network-based approach that aggregates features from node neighborhoods.
    Useful when node features are available.
    
    Example:
        >>> embedder = GraphSAGEEmbeddings(
        ...     dimensions=64,
        ...     num_layers=2,
        ...     aggregator='mean'
        ... )
        >>> embeddings = embedder.fit_transform(graph, node_features)
    """
    
    def __init__(
        self,
        dimensions: int = 64,
        num_layers: int = 2,
        aggregator: str = 'mean',
        learning_rate: float = 0.01,
        batch_size: int = 256,
        epochs: int = 50,
        dropout: float = 0.5
    ):
        """
        Initialize GraphSAGE embeddings.
        
        Args:
            dimensions: Embedding dimension
            num_layers: Number of GraphSAGE layers
            aggregator: Aggregation function ('mean', 'pool', 'lstm')
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            dropout: Dropout rate
        """
        super().__init__(dimensions)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GraphSAGE")
        
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info("Initialized GraphSAGEEmbeddings")
    
    def fit(
        self,
        graph: nx.Graph,
        node_features: Optional[np.ndarray] = None
    ) -> 'GraphSAGEEmbeddings':
        """
        Fit GraphSAGE model.
        
        Args:
            graph: NetworkX graph
            node_features: Optional node features (if None, uses one-hot encoding)
            
        Returns:
            Self
        """
        logger.info(f"Fitting GraphSAGE on graph with {graph.number_of_nodes()} nodes")
        
        nodes = list(graph.nodes())
        self.node_to_idx_ = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node_ = {i: node for i, node in enumerate(nodes)}
        
        # Create node features if not provided
        if node_features is None:
            # Use one-hot encoding of node degree as features
            degrees = dict(graph.degree())
            max_degree = max(degrees.values())
            node_features = np.zeros((len(nodes), max_degree + 1))
            for i, node in enumerate(nodes):
                degree = degrees[node]
                node_features[i, degree] = 1
        
        # Create adjacency list
        adj_list = defaultdict(list)
        for node in nodes:
            neighbors = list(graph.neighbors(node))
            adj_list[self.node_to_idx_[node]] = [
                self.node_to_idx_[n] for n in neighbors
            ]
        
        # Create and train model
        input_dim = node_features.shape[1]
        self.model = GraphSAGE(
            input_dim=input_dim,
            hidden_dim=self.dimensions,
            output_dim=self.dimensions,
            num_layers=self.num_layers,
            dropout=self.dropout,
            aggregator=self.aggregator
        ).to(self.device)
        
        # Train using unsupervised loss
        self._train_unsupervised(
            node_features,
            adj_list,
            graph
        )
        
        # Extract embeddings
        self.model.eval()
        with torch.no_grad():
            features_t = torch.FloatTensor(node_features).to(self.device)
            self.embeddings_ = self.model(features_t, adj_list).cpu().numpy()
        
        logger.info("GraphSAGE fitting completed")
        return self
    
    def _train_unsupervised(
        self,
        node_features: np.ndarray,
        adj_list: Dict[int, List[int]],
        graph: nx.Graph
    ):
        """Train GraphSAGE with unsupervised loss."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        features_t = torch.FloatTensor(node_features).to(self.device)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Sample node pairs
            edges = list(graph.edges())
            np.random.shuffle(edges)
            
            for i in range(0, len(edges), self.batch_size):
                batch_edges = edges[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                # Get embeddings
                embeddings = self.model(features_t, adj_list)
                
                # Compute loss for edge pairs
                loss = 0
                for u, v in batch_edges:
                    u_idx = self.node_to_idx_[u]
                    v_idx = self.node_to_idx_[v]
                    
                    # Positive pair
                    pos_score = torch.sigmoid(
                        torch.dot(embeddings[u_idx], embeddings[v_idx])
                    )
                    
                    # Negative sampling
                    neg_v = np.random.randint(0, len(node_features))
                    neg_score = torch.sigmoid(
                        torch.dot(embeddings[u_idx], embeddings[neg_v])
                    )
                    
                    # BCE loss
                    loss += -torch.log(pos_score + 1e-10) - torch.log(1 - neg_score + 1e-10)
                
                loss = loss / len(batch_edges)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: loss={total_loss:.4f}")
    
    def transform(self, nodes: Optional[List[Any]] = None) -> np.ndarray:
        """
        Transform nodes to embeddings.
        
        Args:
            nodes: List of nodes (if None, returns all embeddings)
            
        Returns:
            Node embeddings
        """
        if self.embeddings_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if nodes is None:
            return self.embeddings_
        
        embeddings = []
        for node in nodes:
            emb = self.get_embedding(node)
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(self.dimensions))
        
        return np.array(embeddings)


class GraphSAGE(nn.Module):
    """GraphSAGE neural network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = 'mean'
    ):
        """Initialize GraphSAGE."""
        super().__init__()
        
        self.num_layers = num_layers
        self.aggregator = aggregator
        
        # Layers
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim, aggregator))
        
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator))
        
        if num_layers > 1:
            self.layers.append(GraphSAGELayer(hidden_dim, output_dim, aggregator))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_list: Dict[int, List[int]]
    ) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_list)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer."""
    
    def __init__(self, input_dim: int, output_dim: int, aggregator: str = 'mean'):
        """Initialize GraphSAGE layer."""
        super().__init__()
        
        self.aggregator = aggregator
        self.linear = nn.Linear(input_dim * 2, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        adj_list: Dict[int, List[int]]
    ) -> torch.Tensor:
        """Forward pass."""
        num_nodes = x.shape[0]
        aggregated = torch.zeros_like(x)
        
        # Aggregate neighbor features
        for node_idx in range(num_nodes):
            neighbors = adj_list.get(node_idx, [])
            if neighbors:
                neighbor_features = x[neighbors]
                if self.aggregator == 'mean':
                    aggregated[node_idx] = neighbor_features.mean(dim=0)
                elif self.aggregator == 'pool':
                    aggregated[node_idx] = neighbor_features.max(dim=0)[0]
                else:
                    aggregated[node_idx] = neighbor_features.mean(dim=0)
        
        # Concatenate self and aggregated features
        combined = torch.cat([x, aggregated], dim=1)
        
        return self.linear(combined)


class CommunityEmbeddings:
    """
    Integrate community detection with embeddings.
    
    Detects communities in the graph and uses them to enhance embeddings
    or as additional features.
    
    Example:
        >>> community_emb = CommunityEmbeddings(method='louvain')
        >>> communities = community_emb.detect_communities(graph)
        >>> features = community_emb.create_community_features(graph, embeddings)
    """
    
    def __init__(self, method: str = 'louvain'):
        """
        Initialize community embeddings.
        
        Args:
            method: Community detection method ('louvain', 'label_propagation')
        """
        self.method = method
        self.communities_: Optional[Dict[Any, int]] = None
        
        logger.info(f"Initialized CommunityEmbeddings with {method}")
    
    def detect_communities(self, graph: nx.Graph) -> Dict[Any, int]:
        """
        Detect communities in graph.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping nodes to community IDs
        """
        logger.info(f"Detecting communities with {self.method}")
        
        if self.method == 'louvain':
            try:
                import community as community_louvain
                self.communities_ = community_louvain.best_partition(graph)
            except ImportError:
                logger.warning("python-louvain not available, using label propagation")
                self.method = 'label_propagation'
        
        if self.method == 'label_propagation':
            communities = nx.algorithms.community.label_propagation_communities(graph)
            self.communities_ = {}
            for i, community in enumerate(communities):
                for node in community:
                    self.communities_[node] = i
        
        n_communities = len(set(self.communities_.values()))
        logger.info(f"Detected {n_communities} communities")
        
        return self.communities_
    
    def create_community_features(
        self,
        graph: nx.Graph,
        embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create community-based features.
        
        Args:
            graph: NetworkX graph
            embeddings: Optional node embeddings to enhance
            
        Returns:
            Community features
        """
        if self.communities_ is None:
            self.detect_communities(graph)
        
        nodes = list(graph.nodes())
        n_communities = len(set(self.communities_.values()))
        
        # One-hot encode communities
        community_features = np.zeros((len(nodes), n_communities))
        for i, node in enumerate(nodes):
            community_id = self.communities_[node]
            community_features[i, community_id] = 1
        
        # Optionally concatenate with embeddings
        if embeddings is not None:
            community_features = np.hstack([embeddings, community_features])
        
        logger.info(f"Created community features with shape {community_features.shape}")
        return community_features


class EmbeddingQualityMetrics:
    """
    Evaluate quality of graph embeddings.
    
    Example:
        >>> metrics = EmbeddingQualityMetrics()
        >>> quality = metrics.evaluate(graph, embeddings)
    """
    
    @staticmethod
    def evaluate(
        graph: nx.Graph,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate embedding quality.
        
        Args:
            graph: NetworkX graph
            embeddings: Node embeddings
            labels: Optional ground truth labels
            
        Returns:
            Dictionary of quality metrics
        """
        logger.info("Evaluating embedding quality")
        
        metrics = {}
        
        # Link prediction quality (how well embeddings preserve edges)
        if graph.number_of_edges() > 0:
            edge_scores = []
            non_edge_scores = []
            
            edges = list(graph.edges())
            nodes = list(graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Sample edges
            sample_size = min(1000, len(edges))
            sampled_edges = np.random.choice(len(edges), sample_size, replace=False)
            
            for idx in sampled_edges:
                u, v = edges[idx]
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                
                # Cosine similarity
                score = np.dot(embeddings[u_idx], embeddings[v_idx]) / (
                    np.linalg.norm(embeddings[u_idx]) * np.linalg.norm(embeddings[v_idx]) + 1e-10
                )
                edge_scores.append(score)
            
            # Sample non-edges
            for _ in range(sample_size):
                u = np.random.choice(nodes)
                v = np.random.choice(nodes)
                if not graph.has_edge(u, v) and u != v:
                    u_idx = node_to_idx[u]
                    v_idx = node_to_idx[v]
                    score = np.dot(embeddings[u_idx], embeddings[v_idx]) / (
                        np.linalg.norm(embeddings[u_idx]) * np.linalg.norm(embeddings[v_idx]) + 1e-10
                    )
                    non_edge_scores.append(score)
            
            metrics['link_pred_edge_mean'] = float(np.mean(edge_scores))
            metrics['link_pred_non_edge_mean'] = float(np.mean(non_edge_scores))
            metrics['link_pred_separation'] = metrics['link_pred_edge_mean'] - metrics['link_pred_non_edge_mean']
        
        # Clustering quality
        if labels is not None and len(np.unique(labels)) > 1:
            try:
                metrics['silhouette_score'] = float(silhouette_score(embeddings, labels))
            except Exception as e:
                logger.warning(f"Could not compute silhouette score: {e}")
        
        logger.info(f"Embedding quality metrics: {metrics}")
        return metrics


class TemporalGraphEmbeddings:
    """
    Embeddings for temporal/evolving graphs.
    
    Handles graphs that change over time by maintaining and updating embeddings
    as the graph evolves.
    
    Example:
        >>> temporal_emb = TemporalGraphEmbeddings(
        ...     base_embedder=Node2VecEmbeddings(dimensions=64)
        ... )
        >>> temporal_emb.add_snapshot(graph_t0, timestamp=0)
        >>> temporal_emb.add_snapshot(graph_t1, timestamp=1)
        >>> embeddings = temporal_emb.get_current_embeddings()
    """
    
    def __init__(
        self,
        base_embedder: BaseGraphEmbedding,
        memory_decay: float = 0.9
    ):
        """
        Initialize temporal graph embeddings.
        
        Args:
            base_embedder: Base embedding method
            memory_decay: Decay factor for temporal smoothing
        """
        self.base_embedder = base_embedder
        self.memory_decay = memory_decay
        self.snapshots: List[Tuple[int, nx.Graph]] = []
        self.embeddings_history: List[Tuple[int, np.ndarray]] = []
        self.current_embeddings: Optional[np.ndarray] = None
        
        logger.info("Initialized TemporalGraphEmbeddings")
    
    def add_snapshot(
        self,
        graph: nx.Graph,
        timestamp: int
    ):
        """
        Add graph snapshot at timestamp.
        
        Args:
            graph: Graph snapshot
            timestamp: Timestamp
        """
        logger.info(f"Adding graph snapshot at t={timestamp}")
        
        self.snapshots.append((timestamp, graph))
        
        # Compute embeddings for this snapshot
        embeddings = self.base_embedder.fit_transform(graph)
        
        # Apply temporal smoothing with previous embeddings
        if self.current_embeddings is not None:
            embeddings = (
                self.memory_decay * self.current_embeddings +
                (1 - self.memory_decay) * embeddings
            )
        
        self.current_embeddings = embeddings
        self.embeddings_history.append((timestamp, embeddings))
        
        logger.info(f"Updated embeddings at t={timestamp}")
    
    def get_current_embeddings(self) -> Optional[np.ndarray]:
        """Get current embeddings."""
        return self.current_embeddings
    
    def get_embeddings_at_time(self, timestamp: int) -> Optional[np.ndarray]:
        """
        Get embeddings at specific timestamp.
        
        Args:
            timestamp: Timestamp
            
        Returns:
            Embeddings at timestamp or None
        """
        for t, embeddings in self.embeddings_history:
            if t == timestamp:
                return embeddings
        return None
