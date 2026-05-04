"""
Risk Propagation Through Fraud Networks

Implements risk propagation algorithms to identify fraud rings and connected fraud:
- PageRank-based risk propagation
- Label propagation for risk scoring
- Network influence scoring
- Fraud ring detection
- Iterative risk scoring algorithms
- Damping factor optimization

Example:
    >>> from src.use_cases.fraud_detection.risk_propagation import PageRankRiskPropagation
    >>> propagator = PageRankRiskPropagation(damping=0.85)
    >>> risk_scores = propagator.propagate(graph, initial_fraud_nodes)
    >>> fraud_rings = propagator.detect_fraud_rings(graph, risk_scores, threshold=0.7)
"""

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from loguru import logger
from sklearn.metrics import roc_auc_score


class BaseRiskPropagation(ABC):
    """Base class for risk propagation algorithms."""
    
    def __init__(self):
        """Initialize risk propagation."""
        self.risk_scores_: Optional[Dict[Any, float]] = None
    
    @abstractmethod
    def propagate(
        self,
        graph: nx.Graph,
        initial_scores: Dict[Any, float],
        **kwargs
    ) -> Dict[Any, float]:
        """Propagate risk through graph."""
        pass
    
    def get_top_risk_nodes(
        self,
        top_k: int = 100
    ) -> List[Tuple[Any, float]]:
        """
        Get top-k nodes by risk score.
        
        Args:
            top_k: Number of top nodes to return
            
        Returns:
            List of (node, risk_score) tuples
        """
        if self.risk_scores_ is None:
            raise ValueError("No risk scores available. Call propagate() first.")
        
        sorted_nodes = sorted(
            self.risk_scores_.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:top_k]
    
    def save(self, path: Union[str, Path]):
        """Save risk scores to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.risk_scores_, path)
        logger.info(f"Risk scores saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseRiskPropagation':
        """Load risk scores from disk."""
        instance = cls()
        instance.risk_scores_ = joblib.load(path)
        logger.info(f"Risk scores loaded from {path}")
        return instance


class PageRankRiskPropagation(BaseRiskPropagation):
    """
    PageRank-based risk propagation.
    
    Uses modified PageRank algorithm where fraud nodes act as sources
    of risk that propagates through the network.
    
    Example:
        >>> propagator = PageRankRiskPropagation(damping=0.85, max_iter=100)
        >>> initial_scores = {fraud_node: 1.0 for fraud_node in known_fraud}
        >>> risk_scores = propagator.propagate(graph, initial_scores)
    """
    
    def __init__(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6,
        personalization_weight: float = 0.5
    ):
        """
        Initialize PageRank risk propagation.
        
        Args:
            damping: Damping factor (probability of continuing random walk)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            personalization_weight: Weight for personalized PageRank
        """
        super().__init__()
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.personalization_weight = personalization_weight
        
        logger.info("Initialized PageRankRiskPropagation")
    
    def propagate(
        self,
        graph: nx.Graph,
        initial_scores: Dict[Any, float],
        edge_weights: Optional[Dict[Tuple[Any, Any], float]] = None
    ) -> Dict[Any, float]:
        """
        Propagate risk using PageRank.
        
        Args:
            graph: NetworkX graph
            initial_scores: Initial risk scores (1.0 for known fraud, 0.0 otherwise)
            edge_weights: Optional edge weights for weighted propagation
            
        Returns:
            Dictionary of node risk scores
        """
        logger.info(f"Propagating risk through {graph.number_of_nodes()} nodes")
        
        nodes = list(graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Initialize personalization vector from initial scores
        personalization = np.zeros(n)
        for node, score in initial_scores.items():
            if node in node_to_idx:
                personalization[node_to_idx[node]] = score
        
        # Normalize personalization
        if personalization.sum() > 0:
            personalization = personalization / personalization.sum()
        else:
            personalization = np.ones(n) / n
        
        # Build adjacency matrix
        adj_matrix = np.zeros((n, n))
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                
                # Get edge weight
                if edge_weights and (u, v) in edge_weights:
                    weight = edge_weights[(u, v)]
                elif edge_weights and (v, u) in edge_weights:
                    weight = edge_weights[(v, u)]
                else:
                    weight = 1.0
                
                adj_matrix[u_idx, v_idx] = weight
                adj_matrix[v_idx, u_idx] = weight  # Undirected
        
        # Normalize by out-degree
        out_degree = adj_matrix.sum(axis=1)
        for i in range(n):
            if out_degree[i] > 0:
                adj_matrix[i] = adj_matrix[i] / out_degree[i]
        
        # Initialize scores
        scores = np.ones(n) / n
        
        # Power iteration
        for iteration in range(self.max_iter):
            prev_scores = scores.copy()
            
            # PageRank update with personalization
            scores = (
                self.damping * adj_matrix.T @ scores +
                (1 - self.damping) * personalization
            )
            
            # Check convergence
            diff = np.abs(scores - prev_scores).sum()
            if diff < self.tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        # Convert to dictionary
        self.risk_scores_ = {
            node: float(scores[node_to_idx[node]])
            for node in nodes
        }
        
        logger.info("Risk propagation completed")
        return self.risk_scores_
    
    def optimize_damping(
        self,
        graph: nx.Graph,
        initial_scores: Dict[Any, float],
        ground_truth: Dict[Any, int],
        damping_range: Tuple[float, float] = (0.5, 0.95),
        n_trials: int = 10
    ) -> float:
        """
        Optimize damping factor using ground truth labels.
        
        Args:
            graph: NetworkX graph
            initial_scores: Initial risk scores
            ground_truth: Ground truth fraud labels
            damping_range: Range of damping values to try
            n_trials: Number of trials
            
        Returns:
            Optimal damping factor
        """
        logger.info("Optimizing damping factor")
        
        best_damping = self.damping
        best_auc = 0
        
        damping_values = np.linspace(damping_range[0], damping_range[1], n_trials)
        
        for damping in damping_values:
            self.damping = damping
            risk_scores = self.propagate(graph, initial_scores)
            
            # Evaluate using AUC
            y_true = []
            y_score = []
            
            for node in graph.nodes():
                if node in ground_truth and node in risk_scores:
                    y_true.append(ground_truth[node])
                    y_score.append(risk_scores[node])
            
            if len(set(y_true)) > 1:  # Need both classes
                auc = roc_auc_score(y_true, y_score)
                logger.info(f"Damping={damping:.3f}, AUC={auc:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_damping = damping
        
        self.damping = best_damping
        logger.info(f"Optimal damping: {best_damping:.3f} (AUC={best_auc:.4f})")
        
        return best_damping
    
    def detect_fraud_rings(
        self,
        graph: nx.Graph,
        risk_scores: Optional[Dict[Any, float]] = None,
        threshold: float = 0.7,
        min_ring_size: int = 3
    ) -> List[Set[Any]]:
        """
        Detect fraud rings (densely connected high-risk subgraphs).
        
        Args:
            graph: NetworkX graph
            risk_scores: Risk scores (uses self.risk_scores_ if None)
            threshold: Risk threshold for inclusion
            min_ring_size: Minimum ring size
            
        Returns:
            List of fraud rings (sets of nodes)
        """
        logger.info("Detecting fraud rings")
        
        if risk_scores is None:
            risk_scores = self.risk_scores_
        
        if risk_scores is None:
            raise ValueError("No risk scores available")
        
        # Filter high-risk nodes
        high_risk_nodes = {
            node for node, score in risk_scores.items()
            if score >= threshold
        }
        
        # Extract subgraph of high-risk nodes
        high_risk_subgraph = graph.subgraph(high_risk_nodes)
        
        # Find connected components
        fraud_rings = []
        for component in nx.connected_components(high_risk_subgraph):
            if len(component) >= min_ring_size:
                fraud_rings.append(component)
        
        logger.info(f"Detected {len(fraud_rings)} fraud rings")
        return fraud_rings


class LabelPropagationRisk(BaseRiskPropagation):
    """
    Label propagation for risk scoring.
    
    Iteratively propagates risk labels through the network based on
    neighborhood consensus.
    
    Example:
        >>> propagator = LabelPropagationRisk(max_iter=30)
        >>> risk_scores = propagator.propagate(graph, initial_scores)
    """
    
    def __init__(
        self,
        max_iter: int = 30,
        tol: float = 1e-3,
        propagation_weight: float = 0.5
    ):
        """
        Initialize label propagation.
        
        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance
            propagation_weight: Weight for neighbor influence
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.propagation_weight = propagation_weight
        
        logger.info("Initialized LabelPropagationRisk")
    
    def propagate(
        self,
        graph: nx.Graph,
        initial_scores: Dict[Any, float],
        fixed_nodes: Optional[Set[Any]] = None
    ) -> Dict[Any, float]:
        """
        Propagate risk using label propagation.
        
        Args:
            graph: NetworkX graph
            initial_scores: Initial risk scores
            fixed_nodes: Nodes with fixed scores (not updated)
            
        Returns:
            Dictionary of node risk scores
        """
        logger.info("Propagating risk with label propagation")
        
        # Initialize scores
        scores = {node: 0.0 for node in graph.nodes()}
        scores.update(initial_scores)
        
        if fixed_nodes is None:
            fixed_nodes = set(initial_scores.keys())
        
        # Iterative propagation
        for iteration in range(self.max_iter):
            new_scores = scores.copy()
            
            for node in graph.nodes():
                if node in fixed_nodes:
                    continue
                
                # Get neighbor scores
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    continue
                
                neighbor_scores = [scores[n] for n in neighbors]
                avg_neighbor_score = np.mean(neighbor_scores)
                
                # Update score as weighted average
                new_scores[node] = (
                    self.propagation_weight * avg_neighbor_score +
                    (1 - self.propagation_weight) * scores[node]
                )
            
            # Check convergence
            diff = sum(
                abs(new_scores[node] - scores[node])
                for node in graph.nodes()
            )
            
            scores = new_scores
            
            if diff < self.tol:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
        
        self.risk_scores_ = scores
        logger.info("Label propagation completed")
        return self.risk_scores_


class NetworkInfluenceScoring(BaseRiskPropagation):
    """
    Network influence-based risk scoring.
    
    Computes risk scores based on network structure and influence metrics
    like betweenness centrality and eigenvector centrality.
    
    Example:
        >>> scorer = NetworkInfluenceScoring()
        >>> risk_scores = scorer.propagate(
        ...     graph,
        ...     initial_scores,
        ...     use_betweenness=True,
        ...     use_eigenvector=True
        ... )
    """
    
    def __init__(self):
        """Initialize network influence scoring."""
        super().__init__()
        logger.info("Initialized NetworkInfluenceScoring")
    
    def propagate(
        self,
        graph: nx.Graph,
        initial_scores: Dict[Any, float],
        use_betweenness: bool = True,
        use_eigenvector: bool = True,
        use_closeness: bool = False,
        betweenness_weight: float = 0.3,
        eigenvector_weight: float = 0.3,
        closeness_weight: float = 0.2,
        initial_weight: float = 0.2
    ) -> Dict[Any, float]:
        """
        Compute risk scores using network influence.
        
        Args:
            graph: NetworkX graph
            initial_scores: Initial risk scores
            use_betweenness: Use betweenness centrality
            use_eigenvector: Use eigenvector centrality
            use_closeness: Use closeness centrality
            betweenness_weight: Weight for betweenness
            eigenvector_weight: Weight for eigenvector
            closeness_weight: Weight for closeness
            initial_weight: Weight for initial scores
            
        Returns:
            Dictionary of node risk scores
        """
        logger.info("Computing network influence scores")
        
        nodes = list(graph.nodes())
        scores = {node: 0.0 for node in nodes}
        
        # Normalize initial scores
        initial_total = sum(initial_scores.values())
        if initial_total > 0:
            normalized_initial = {
                node: score / initial_total
                for node, score in initial_scores.items()
            }
        else:
            normalized_initial = initial_scores
        
        # Compute centrality measures
        centralities = {}
        
        if use_betweenness:
            logger.info("Computing betweenness centrality")
            centralities['betweenness'] = nx.betweenness_centrality(graph)
        
        if use_eigenvector:
            logger.info("Computing eigenvector centrality")
            try:
                centralities['eigenvector'] = nx.eigenvector_centrality(
                    graph, max_iter=1000
                )
            except:
                logger.warning("Eigenvector centrality failed, using degree centrality")
                centralities['eigenvector'] = nx.degree_centrality(graph)
        
        if use_closeness:
            logger.info("Computing closeness centrality")
            centralities['closeness'] = nx.closeness_centrality(graph)
        
        # Normalize centralities
        for name, cent in centralities.items():
            max_cent = max(cent.values()) if cent else 1.0
            if max_cent > 0:
                centralities[name] = {
                    node: val / max_cent for node, val in cent.items()
                }
        
        # Combine scores
        weights = {
            'betweenness': betweenness_weight,
            'eigenvector': eigenvector_weight,
            'closeness': closeness_weight
        }
        
        for node in nodes:
            score = initial_weight * normalized_initial.get(node, 0.0)
            
            for name, cent in centralities.items():
                score += weights[name] * cent.get(node, 0.0)
            
            scores[node] = score
        
        # Boost scores for nodes connected to high-risk nodes
        for node in nodes:
            if node in initial_scores and initial_scores[node] > 0:
                for neighbor in graph.neighbors(node):
                    scores[neighbor] *= 1.5  # Amplify neighbor scores
        
        self.risk_scores_ = scores
        logger.info("Network influence scoring completed")
        return self.risk_scores_


class IterativeRiskScoring:
    """
    Iterative risk scoring with multiple rounds.
    
    Combines multiple propagation methods and iteratively refines scores
    based on discovered patterns.
    
    Example:
        >>> scorer = IterativeRiskScoring(
        ...     methods=['pagerank', 'label_propagation']
        ... )
        >>> risk_scores = scorer.score(graph, initial_fraud_nodes)
    """
    
    def __init__(
        self,
        methods: List[str] = ['pagerank', 'label_propagation', 'influence'],
        n_iterations: int = 3,
        top_k_threshold: int = 100,
        score_threshold: float = 0.5
    ):
        """
        Initialize iterative risk scoring.
        
        Args:
            methods: List of propagation methods to use
            n_iterations: Number of iterations
            top_k_threshold: Number of top nodes to add per iteration
            score_threshold: Score threshold for adding nodes
        """
        self.methods = methods
        self.n_iterations = n_iterations
        self.top_k_threshold = top_k_threshold
        self.score_threshold = score_threshold
        
        self.propagators = {
            'pagerank': PageRankRiskPropagation(),
            'label_propagation': LabelPropagationRisk(),
            'influence': NetworkInfluenceScoring()
        }
        
        logger.info("Initialized IterativeRiskScoring")
    
    def score(
        self,
        graph: nx.Graph,
        initial_fraud_nodes: Set[Any]
    ) -> Dict[Any, float]:
        """
        Score nodes iteratively.
        
        Args:
            graph: NetworkX graph
            initial_fraud_nodes: Set of known fraud nodes
            
        Returns:
            Dictionary of final risk scores
        """
        logger.info(f"Starting iterative scoring with {len(initial_fraud_nodes)} initial fraud nodes")
        
        current_fraud_nodes = initial_fraud_nodes.copy()
        final_scores = {node: 0.0 for node in graph.nodes()}
        
        for iteration in range(self.n_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}")
            
            # Create initial scores
            initial_scores = {
                node: 1.0 for node in current_fraud_nodes
            }
            
            # Run all propagation methods
            iteration_scores = []
            
            for method_name in self.methods:
                if method_name in self.propagators:
                    propagator = self.propagators[method_name]
                    scores = propagator.propagate(graph, initial_scores)
                    iteration_scores.append(scores)
            
            # Average scores across methods
            avg_scores = {node: 0.0 for node in graph.nodes()}
            for scores in iteration_scores:
                for node, score in scores.items():
                    avg_scores[node] += score / len(iteration_scores)
            
            # Update final scores (cumulative)
            for node, score in avg_scores.items():
                final_scores[node] = max(final_scores[node], score)
            
            # Add top-k high-risk nodes for next iteration
            sorted_nodes = sorted(
                avg_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            new_fraud_nodes = set()
            for node, score in sorted_nodes[:self.top_k_threshold]:
                if score >= self.score_threshold and node not in current_fraud_nodes:
                    new_fraud_nodes.add(node)
            
            if not new_fraud_nodes:
                logger.info("No new fraud nodes found, stopping")
                break
            
            logger.info(f"Added {len(new_fraud_nodes)} new suspected fraud nodes")
            current_fraud_nodes.update(new_fraud_nodes)
        
        logger.info("Iterative scoring completed")
        return final_scores
    
    def detect_fraud_communities(
        self,
        graph: nx.Graph,
        risk_scores: Dict[Any, float],
        threshold: float = 0.6
    ) -> List[Set[Any]]:
        """
        Detect fraud communities using risk scores.
        
        Args:
            graph: NetworkX graph
            risk_scores: Node risk scores
            threshold: Risk threshold
            
        Returns:
            List of fraud communities
        """
        logger.info("Detecting fraud communities")
        
        # Filter high-risk nodes
        high_risk_nodes = {
            node for node, score in risk_scores.items()
            if score >= threshold
        }
        
        # Create subgraph
        subgraph = graph.subgraph(high_risk_nodes)
        
        # Detect communities using Louvain or label propagation
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(subgraph)
            
            communities = defaultdict(set)
            for node, comm_id in partition.items():
                communities[comm_id].add(node)
            
            fraud_communities = list(communities.values())
        except ImportError:
            # Fall back to connected components
            fraud_communities = [
                comp for comp in nx.connected_components(subgraph)
                if len(comp) >= 2
            ]
        
        logger.info(f"Detected {len(fraud_communities)} fraud communities")
        return fraud_communities


class FraudRingDetector:
    """
    Specialized fraud ring detection.
    
    Combines risk propagation with graph pattern matching to identify
    organized fraud rings.
    
    Example:
        >>> detector = FraudRingDetector()
        >>> rings = detector.detect(graph, known_fraud_nodes)
    """
    
    def __init__(
        self,
        min_ring_size: int = 3,
        max_ring_size: int = 20,
        density_threshold: float = 0.5,
        risk_threshold: float = 0.6
    ):
        """
        Initialize fraud ring detector.
        
        Args:
            min_ring_size: Minimum ring size
            max_ring_size: Maximum ring size
            density_threshold: Minimum edge density
            risk_threshold: Minimum average risk score
        """
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.density_threshold = density_threshold
        self.risk_threshold = risk_threshold
        
        logger.info("Initialized FraudRingDetector")
    
    def detect(
        self,
        graph: nx.Graph,
        known_fraud_nodes: Set[Any],
        risk_scores: Optional[Dict[Any, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect fraud rings.
        
        Args:
            graph: NetworkX graph
            known_fraud_nodes: Known fraud nodes
            risk_scores: Optional risk scores
            
        Returns:
            List of fraud rings with metadata
        """
        logger.info("Detecting fraud rings")
        
        # If no risk scores provided, use PageRank
        if risk_scores is None:
            propagator = PageRankRiskPropagation()
            initial_scores = {node: 1.0 for node in known_fraud_nodes}
            risk_scores = propagator.propagate(graph, initial_scores)
        
        # Find densely connected high-risk subgraphs
        rings = []
        
        # Use k-clique communities for finding dense regions
        try:
            for k in range(self.min_ring_size, min(self.max_ring_size, 10)):
                cliques = list(nx.algorithms.community.k_clique_communities(graph, k))
                
                for clique in cliques:
                    if len(clique) < self.min_ring_size or len(clique) > self.max_ring_size:
                        continue
                    
                    # Check density
                    subgraph = graph.subgraph(clique)
                    density = nx.density(subgraph)
                    
                    if density < self.density_threshold:
                        continue
                    
                    # Check average risk score
                    avg_risk = np.mean([risk_scores.get(node, 0.0) for node in clique])
                    
                    if avg_risk < self.risk_threshold:
                        continue
                    
                    # Check if contains known fraud
                    has_fraud = any(node in known_fraud_nodes for node in clique)
                    
                    rings.append({
                        'nodes': clique,
                        'size': len(clique),
                        'density': density,
                        'avg_risk': avg_risk,
                        'has_known_fraud': has_fraud,
                        'fraud_count': sum(1 for node in clique if node in known_fraud_nodes)
                    })
        except Exception as e:
            logger.warning(f"k-clique detection failed: {e}, using connected components")
            
            # Fall back to connected components of high-risk nodes
            high_risk_nodes = {
                node for node, score in risk_scores.items()
                if score >= self.risk_threshold
            }
            
            subgraph = graph.subgraph(high_risk_nodes)
            
            for component in nx.connected_components(subgraph):
                if len(component) < self.min_ring_size or len(component) > self.max_ring_size:
                    continue
                
                comp_subgraph = graph.subgraph(component)
                density = nx.density(comp_subgraph)
                avg_risk = np.mean([risk_scores[node] for node in component])
                
                rings.append({
                    'nodes': component,
                    'size': len(component),
                    'density': density,
                    'avg_risk': avg_risk,
                    'has_known_fraud': any(node in known_fraud_nodes for node in component),
                    'fraud_count': sum(1 for node in component if node in known_fraud_nodes)
                })
        
        # Sort by average risk
        rings = sorted(rings, key=lambda x: x['avg_risk'], reverse=True)
        
        logger.info(f"Detected {len(rings)} fraud rings")
        return rings
