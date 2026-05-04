"""
Unsupervised learning methods for clustering and dimensionality reduction.

This module provides clustering algorithms, dimensionality reduction techniques,
and autoencoder-based clustering with visualization utilities.

Example:
    >>> from src.models.unsupervised import ClusteringEngine
    >>> clustering = ClusteringEngine(method='kmeans')
    >>> labels = clustering.fit_predict(X, optimal_k=True)
    >>> clustering.visualize_clusters(X, labels)
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from loguru import logger
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    logger.warning("HDBSCAN not available. Install with: pip install hdbscan")
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    logger.warning("UMAP not available. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization libraries not available")
    VISUALIZATION_AVAILABLE = False


class ClusteringEngine:
    """
    Unified clustering engine supporting multiple algorithms.
    
    Supports:
    - KMeans with optimal k selection
    - HDBSCAN (density-based)
    - Gaussian Mixture Models
    - Spectral Clustering
    - Hierarchical Clustering
    
    Example:
        >>> engine = ClusteringEngine(method='kmeans')
        >>> labels = engine.fit_predict(X, optimal_k=True, k_range=(2, 10))
        >>> engine.plot_elbow_curve()
    """
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: int = 3,
        random_state: int = 42
    ):
        """
        Initialize clustering engine.
        
        Args:
            method: Clustering method ('kmeans', 'hdbscan', 'gmm', 'spectral', 'hierarchical')
            n_clusters: Number of clusters (not used for HDBSCAN)
            random_state: Random seed
        """
        self.method = method.lower()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.scaler = StandardScaler()
        self.elbow_scores: Optional[Dict[int, float]] = None
        self.silhouette_scores: Optional[Dict[int, float]] = None
    
    def fit_predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        optimal_k: bool = False,
        k_range: Tuple[int, int] = (2, 10),
        scale: bool = True
    ) -> np.ndarray:
        """
        Fit clustering model and predict cluster labels.
        
        Args:
            X: Input features
            optimal_k: Whether to find optimal number of clusters
            k_range: Range of k values to try if optimal_k=True
            scale: Whether to scale features
            
        Returns:
            Cluster labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        if scale:
            X = self.scaler.fit_transform(X)
        
        # Find optimal k if requested
        if optimal_k and self.method in ['kmeans', 'gmm', 'spectral', 'hierarchical']:
            self.n_clusters = self._find_optimal_k(X, k_range)
            logger.info(f"Optimal number of clusters: {self.n_clusters}")
        
        # Fit clustering model
        logger.info(f"Fitting {self.method} clustering...")
        
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.labels_ = self.model.fit_predict(X)
        
        elif self.method == 'hdbscan':
            if not HDBSCAN_AVAILABLE:
                raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
            self.model = hdbscan.HDBSCAN(
                min_cluster_size=max(2, len(X) // 100),
                min_samples=5
            )
            self.labels_ = self.model.fit_predict(X)
        
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                covariance_type='full'
            )
            self.labels_ = self.model.fit_predict(X)
        
        elif self.method == 'spectral':
            self.model = SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                affinity='rbf'
            )
            self.labels_ = self.model.fit_predict(X)
        
        elif self.method == 'hierarchical':
            self.model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
            self.labels_ = self.model.fit_predict(X)
        
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        logger.info(f"Clustering complete. Found {len(np.unique(self.labels_))} clusters")
        return self.labels_
    
    def _find_optimal_k(
        self,
        X: np.ndarray,
        k_range: Tuple[int, int]
    ) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score.
        
        Args:
            X: Input features
            k_range: Range of k values to try
            
        Returns:
            Optimal k
        """
        logger.info(f"Finding optimal k in range {k_range}...")
        
        self.elbow_scores = {}
        self.silhouette_scores = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            if self.method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = model.fit_predict(X)
                self.elbow_scores[k] = model.inertia_
            elif self.method == 'gmm':
                model = GaussianMixture(n_components=k, random_state=self.random_state)
                labels = model.fit_predict(X)
                self.elbow_scores[k] = -model.bic(X)  # Negative BIC for consistency
            else:
                # For other methods, use simple clustering
                if self.method == 'spectral':
                    model = SpectralClustering(n_clusters=k, random_state=self.random_state)
                else:  # hierarchical
                    model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X)
                self.elbow_scores[k] = -calinski_harabasz_score(X, labels)
            
            # Silhouette score
            if len(np.unique(labels)) > 1:
                self.silhouette_scores[k] = silhouette_score(X, labels)
            else:
                self.silhouette_scores[k] = -1
        
        # Find k with best silhouette score
        optimal_k = max(self.silhouette_scores.items(), key=lambda x: x[1])[0]
        return optimal_k
    
    def plot_elbow_curve(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Plot elbow curve for optimal k selection.
        
        Args:
            filepath: Optional path to save plot
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return
        
        if self.elbow_scores is None:
            logger.warning("No elbow scores available. Run fit_predict with optimal_k=True first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Elbow curve
        k_values = sorted(self.elbow_scores.keys())
        scores = [self.elbow_scores[k] for k in k_values]
        ax1.plot(k_values, scores, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia / BIC')
        ax1.set_title('Elbow Curve')
        ax1.grid(True)
        
        # Silhouette scores
        silhouette_values = [self.silhouette_scores[k] for k in k_values]
        ax2.plot(k_values, silhouette_values, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score by k')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        else:
            plt.show()
    
    def visualize_clusters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
        method: str = 'pca',
        filepath: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Visualize clusters in 2D using dimensionality reduction.
        
        Args:
            X: Input features
            labels: Cluster labels (uses self.labels_ if not provided)
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            filepath: Optional path to save plot
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return
        
        if labels is None:
            labels = self.labels_
        
        if labels is None:
            logger.warning("No labels available. Run fit_predict first.")
            return
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Reduce to 2D
        reducer = DimensionalityReduction(method=method, n_components=2)
        X_2d = reducer.fit_transform(X)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Cluster Visualization ({self.method})')
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        else:
            plt.show()
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save clustering model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'method': self.method,
            'n_clusters': self.n_clusters,
            'model': self.model,
            'labels': self.labels_,
            'scaler': self.scaler
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> 'ClusteringEngine':
        """Load clustering model."""
        model_data = joblib.load(filepath)
        self.method = model_data['method']
        self.n_clusters = model_data['n_clusters']
        self.model = model_data['model']
        self.labels_ = model_data['labels']
        self.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {filepath}")
        return self


class DimensionalityReduction:
    """
    Dimensionality reduction using PCA, t-SNE, or UMAP.
    
    Example:
        >>> reducer = DimensionalityReduction(method='umap', n_components=2)
        >>> X_reduced = reducer.fit_transform(X)
        >>> reducer.plot_explained_variance()
    """
    
    def __init__(
        self,
        method: str = 'pca',
        n_components: int = 2,
        random_state: int = 42
    ):
        """
        Initialize dimensionality reduction.
        
        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components
            random_state: Random seed
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        scale: bool = True
    ) -> np.ndarray:
        """
        Fit reduction model and transform data.
        
        Args:
            X: Input features
            scale: Whether to scale features
            
        Returns:
            Reduced features
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if scale:
            X = self.scaler.fit_transform(X)
        
        logger.info(f"Applying {self.method} dimensionality reduction...")
        
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components, random_state=self.random_state)
            X_reduced = self.model.fit_transform(X)
        
        elif self.method == 'tsne':
            if not TSNE_AVAILABLE:
                raise ImportError("t-SNE not available")
            self.model = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=min(30, len(X) - 1)
            )
            X_reduced = self.model.fit_transform(X)
        
        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            self.model = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=min(15, len(X) - 1)
            )
            X_reduced = self.model.fit_transform(X)
        
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
        
        logger.info(f"Reduction complete: {X.shape} -> {X_reduced.shape}")
        return X_reduced
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform new data using fitted model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit_transform first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self.scaler.transform(X)
        
        if self.method in ['tsne', 'umap']:
            logger.warning(f"{self.method} doesn't support transform. Use fit_transform instead.")
            return self.fit_transform(X, scale=False)
        
        return self.model.transform(X)
    
    def plot_explained_variance(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """
        Plot explained variance (PCA only).
        
        Args:
            filepath: Optional path to save plot
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization libraries not available")
            return
        
        if self.method != 'pca':
            logger.warning("Explained variance only available for PCA")
            return
        
        if self.model is None:
            logger.warning("Model not fitted. Call fit_transform first.")
            return
        
        plt.figure(figsize=(10, 4))
        
        # Explained variance
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.model.explained_variance_ratio_) + 1),
                self.model.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')
        
        # Cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.model.explained_variance_ratio_) + 1),
                 np.cumsum(self.model.explained_variance_ratio_), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        
        if filepath:
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")
        else:
            plt.show()


class AutoencoderClustering:
    """
    Autoencoder-based clustering for representation learning.
    
    Trains an autoencoder to learn compressed representations, then applies
    clustering on the learned embeddings.
    
    Example:
        >>> model = AutoencoderClustering(encoding_dim=10, n_clusters=5)
        >>> model.fit(X, epochs=100)
        >>> labels = model.predict(X)
        >>> embeddings = model.get_embeddings(X)
    """
    
    def __init__(
        self,
        encoding_dim: int = 10,
        n_clusters: int = 3,
        hidden_dims: Optional[List[int]] = None,
        clustering_method: str = 'kmeans',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize autoencoder clustering.
        
        Args:
            encoding_dim: Dimension of encoded representation
            n_clusters: Number of clusters
            hidden_dims: List of hidden layer dimensions
            clustering_method: Clustering algorithm to use
            device: 'cuda' or 'cpu'
        """
        self.encoding_dim = encoding_dim
        self.n_clusters = n_clusters
        self.hidden_dims = hidden_dims or [64, 32]
        self.clustering_method = clustering_method
        self.device = device
        self.autoencoder = None
        self.clustering_model = None
        self.scaler = StandardScaler()
        
        logger.info(f"Using device: {device}")
    
    def _build_autoencoder(self, input_dim: int) -> nn.Module:
        """Build autoencoder architecture."""
        
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim, hidden_dims):
                super().__init__()
                
                # Encoder
                encoder_layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    encoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Decoder
                decoder_layers = []
                prev_dim = encoding_dim
                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
            
            def encode(self, x):
                return self.encoder(x)
        
        return Autoencoder(input_dim, self.encoding_dim, self.hidden_dims)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> 'AutoencoderClustering':
        """
        Fit autoencoder and clustering model.
        
        Args:
            X: Input features
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Whether to print training progress
            
        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Build autoencoder
        input_dim = X.shape[1]
        self.autoencoder = self._build_autoencoder(input_dim).to(self.device)
        
        # Train autoencoder
        logger.info("Training autoencoder...")
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                batch_X = batch[0]
                
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")
        
        # Get embeddings
        self.autoencoder.eval()
        with torch.no_grad():
            embeddings = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        # Fit clustering on embeddings
        logger.info(f"Fitting {self.clustering_method} clustering on embeddings...")
        self.clustering_model = ClusteringEngine(
            method=self.clustering_method,
            n_clusters=self.n_clusters
        )
        self.clustering_model.fit_predict(embeddings, scale=False)
        
        logger.info("Autoencoder clustering complete")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict cluster labels."""
        embeddings = self.get_embeddings(X)
        
        # Use clustering model to predict
        if self.clustering_model.method == 'kmeans':
            return self.clustering_model.model.predict(embeddings)
        else:
            # For other methods, refit (may not be ideal)
            logger.warning(f"{self.clustering_model.method} doesn't support predict. Refitting...")
            return self.clustering_model.fit_predict(embeddings, scale=False)
    
    def get_embeddings(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get autoencoder embeddings."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.autoencoder.eval()
        with torch.no_grad():
            embeddings = self.autoencoder.encode(X_tensor).cpu().numpy()
        
        return embeddings
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save autoencoder
        torch.save(self.autoencoder.state_dict(), filepath.parent / f"{filepath.stem}_autoencoder.pt")
        
        # Save clustering model
        self.clustering_model.save(filepath.parent / f"{filepath.stem}_clustering.pkl")
        
        # Save metadata
        metadata = {
            'encoding_dim': self.encoding_dim,
            'n_clusters': self.n_clusters,
            'hidden_dims': self.hidden_dims,
            'clustering_method': self.clustering_method,
            'scaler': self.scaler
        }
        joblib.dump(metadata, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: Union[str, Path]) -> 'AutoencoderClustering':
        """Load model."""
        filepath = Path(filepath)
        
        # Load metadata
        metadata = joblib.load(filepath)
        self.encoding_dim = metadata['encoding_dim']
        self.n_clusters = metadata['n_clusters']
        self.hidden_dims = metadata['hidden_dims']
        self.clustering_method = metadata['clustering_method']
        self.scaler = metadata['scaler']
        
        # Load clustering model
        self.clustering_model = ClusteringEngine()
        self.clustering_model.load(filepath.parent / f"{filepath.stem}_clustering.pkl")
        
        # Rebuild and load autoencoder (need to know input_dim)
        # This is a limitation - we need to store input_dim in metadata
        logger.warning("Autoencoder loading requires input_dim. Please call fit first or store input_dim.")
        
        logger.info(f"Model loaded from {filepath}")
        return self
