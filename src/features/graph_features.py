"""
Graph Features Engineering Module

Implements graph-based features for network analysis:
- Node embeddings (Node2Vec, GraphSAGE, DeepWalk wrappers)
- Community detection features (Louvain, label propagation)
- Centrality measures (degree, betweenness, closeness, PageRank, eigenvector)
- Graph-based risk propagation scores
- Structural features (clustering coefficient, local density)
- Network statistics

Author: Principal Data Science Decision Agent
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from pathlib import Path

import networkx as nx
from networkx.algorithms import community

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False
    logger.warning("Node2Vec not available. Install with: pip install node2vec")

try:
    from karateclub import DeepWalk, GraphWave
    KARATECLUB_AVAILABLE = True
except ImportError:
    KARATECLUB_AVAILABLE = False
    logger.warning("KarateClub not available. Install with: pip install karateclub")


class GraphFeatureConfig:
    """Configuration for graph feature engineering."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize graph feature configuration.

        Args:
            config_path: Path to feature_config.yaml. If None, uses defaults.

        Example:
            >>> config = GraphFeatureConfig("config/feature_config.yaml")
            >>> config.node_embedding_methods
            ['node2vec', 'graphsage', 'deepwalk']
        """
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
                graph_config = full_config.get("graph", {})
        else:
            graph_config = {}

        node_embeddings_config = graph_config.get("node_embeddings", {})
        self.node_embedding_methods = node_embeddings_config.get(
            "methods", ["node2vec", "deepwalk"]
        )

        self.node2vec_params = node_embeddings_config.get(
            "node2vec",
            {
                "dimensions": 128,
                "walk_length": 80,
                "num_walks": 10,
                "p": 1.0,
                "q": 1.0,
                "workers": 4,
            },
        )

        community_config = graph_config.get("community_detection", {})
        self.community_methods = community_config.get(
            "methods", ["louvain", "label_propagation"]
        )

        self.centrality_measures = graph_config.get(
            "centrality_measures",
            [
                "degree_centrality",
                "betweenness_centrality",
                "closeness_centrality",
                "pagerank",
                "eigenvector_centrality",
            ],
        )

        risk_propagation_config = graph_config.get("risk_propagation", {})
        self.risk_propagation_enabled = risk_propagation_config.get("enabled", True)
        self.risk_iterations = risk_propagation_config.get("iterations", 10)
        self.risk_damping = risk_propagation_config.get("damping_factor", 0.85)


class GraphFeatureEngine:
    """
    Engine for computing graph-based features from network data.

    Provides node embeddings, centrality measures, community detection,
    and risk propagation features for network analysis.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        entity_col: str = "user_id",
    ):
        """
        Initialize graph feature engine.

        Args:
            config_path: Path to feature_config.yaml
            entity_col: Column name for entity identifier

        Example:
            >>> engine = GraphFeatureEngine(
            ...     config_path="config/feature_config.yaml",
            ...     entity_col="customer_id"
            ... )
        """
        self.config = GraphFeatureConfig(config_path)
        self.entity_col = entity_col
        logger.info("Initialized GraphFeatureEngine")

    def build_graph_from_edges(
        self,
        edges_df: pd.DataFrame,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: Optional[str] = None,
        directed: bool = False,
    ) -> nx.Graph:
        """
        Build NetworkX graph from edge list.

        Args:
            edges_df: DataFrame with edge information
            source_col: Column name for source nodes
            target_col: Column name for target nodes
            weight_col: Optional column name for edge weights
            directed: Whether to create directed graph

        Returns:
            NetworkX graph object

        Example:
            >>> edges_df = pd.DataFrame({
            ...     'source': [1, 2, 3, 1],
            ...     'target': [2, 3, 4, 4],
            ...     'weight': [1.0, 2.0, 1.5, 0.5]
            ... })
            >>> engine = GraphFeatureEngine()
            >>> G = engine.build_graph_from_edges(edges_df)
        """
        logger.info(
            f"Building {'directed' if directed else 'undirected'} graph from edges"
        )

        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        # Add edges
        if weight_col and weight_col in edges_df.columns:
            edges = [
                (row[source_col], row[target_col], row[weight_col])
                for _, row in edges_df.iterrows()
            ]
            G.add_weighted_edges_from(edges)
        else:
            edges = [
                (row[source_col], row[target_col])
                for _, row in edges_df.iterrows()
            ]
            G.add_edges_from(edges)

        logger.info(
            f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )

        return G

    def compute_node_embeddings(
        self,
        G: nx.Graph,
        method: str = "node2vec",
        dimensions: int = 128,
        prefix: str = "embed",
    ) -> pd.DataFrame:
        """
        Compute node embeddings using various methods.

        Args:
            G: NetworkX graph
            method: Embedding method ('node2vec', 'deepwalk', 'graphwave')
            dimensions: Embedding dimensionality
            prefix: Prefix for feature names

        Returns:
            DataFrame with node embeddings

        Example:
            >>> G = nx.karate_club_graph()
            >>> engine = GraphFeatureEngine()
            >>> embeddings_df = engine.compute_node_embeddings(G, method='node2vec')
        """
        logger.info(f"Computing {method} node embeddings with {dimensions} dimensions")

        embeddings = {}

        if method == "node2vec" and NODE2VEC_AVAILABLE:
            try:
                node2vec = Node2Vec(
                    G,
                    dimensions=dimensions,
                    walk_length=self.config.node2vec_params.get("walk_length", 80),
                    num_walks=self.config.node2vec_params.get("num_walks", 10),
                    p=self.config.node2vec_params.get("p", 1.0),
                    q=self.config.node2vec_params.get("q", 1.0),
                    workers=self.config.node2vec_params.get("workers", 4),
                    quiet=True,
                )

                model = node2vec.fit(window=10, min_count=1, batch_words=4)

                for node in G.nodes():
                    embeddings[node] = model.wv[str(node)]

            except Exception as e:
                logger.error(f"Node2Vec failed: {str(e)}")
                return pd.DataFrame()

        elif method == "deepwalk" and KARATECLUB_AVAILABLE:
            try:
                # Convert to undirected if needed
                G_undirected = G.to_undirected() if G.is_directed() else G

                # Relabel nodes to consecutive integers
                mapping = {node: i for i, node in enumerate(G_undirected.nodes())}
                reverse_mapping = {i: node for node, i in mapping.items()}
                G_relabeled = nx.relabel_nodes(G_undirected, mapping)

                deepwalk = DeepWalk(dimensions=dimensions)
                deepwalk.fit(G_relabeled)

                embedding_matrix = deepwalk.get_embedding()

                for i, node_id in reverse_mapping.items():
                    embeddings[node_id] = embedding_matrix[i]

            except Exception as e:
                logger.error(f"DeepWalk failed: {str(e)}")
                return pd.DataFrame()

        else:
            logger.warning(
                f"Method {method} not available or not supported. "
                f"Available: {self.config.node_embedding_methods}"
            )
            return pd.DataFrame()

        # Convert to DataFrame
        if embeddings:
            embedding_df = pd.DataFrame.from_dict(embeddings, orient="index")
            embedding_df.columns = [
                f"{prefix}_{method}_{i}" for i in range(dimensions)
            ]
            embedding_df.index.name = self.entity_col
            embedding_df = embedding_df.reset_index()

            logger.info(f"Generated {dimensions} {method} embeddings for {len(embedding_df)} nodes")

            return embedding_df
        else:
            return pd.DataFrame()

    def compute_centrality_features(
        self,
        G: nx.Graph,
        measures: Optional[List[str]] = None,
        prefix: str = "centrality",
    ) -> pd.DataFrame:
        """
        Compute various centrality measures.

        Args:
            G: NetworkX graph
            measures: List of centrality measures to compute. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with centrality features

        Example:
            >>> G = nx.karate_club_graph()
            >>> engine = GraphFeatureEngine()
            >>> centrality_df = engine.compute_centrality_features(G)
        """
        measures = measures or self.config.centrality_measures

        logger.info(f"Computing centrality measures: {measures}")

        centrality_dict = {}

        if "degree_centrality" in measures:
            try:
                centrality_dict[f"{prefix}_degree"] = nx.degree_centrality(G)
            except Exception as e:
                logger.warning(f"Degree centrality failed: {str(e)}")

        if "betweenness_centrality" in measures:
            try:
                centrality_dict[f"{prefix}_betweenness"] = nx.betweenness_centrality(G)
            except Exception as e:
                logger.warning(f"Betweenness centrality failed: {str(e)}")

        if "closeness_centrality" in measures:
            try:
                # Only works on connected graphs
                if nx.is_connected(G) or nx.is_weakly_connected(G):
                    centrality_dict[f"{prefix}_closeness"] = nx.closeness_centrality(G)
                else:
                    logger.warning("Graph not connected, skipping closeness centrality")
            except Exception as e:
                logger.warning(f"Closeness centrality failed: {str(e)}")

        if "pagerank" in measures:
            try:
                centrality_dict[f"{prefix}_pagerank"] = nx.pagerank(G)
            except Exception as e:
                logger.warning(f"PageRank failed: {str(e)}")

        if "eigenvector_centrality" in measures:
            try:
                centrality_dict[f"{prefix}_eigenvector"] = nx.eigenvector_centrality(
                    G, max_iter=1000
                )
            except Exception as e:
                logger.warning(f"Eigenvector centrality failed: {str(e)}")

        # Convert to DataFrame
        if centrality_dict:
            centrality_df = pd.DataFrame(centrality_dict)
            centrality_df.index.name = self.entity_col
            centrality_df = centrality_df.reset_index()

            logger.info(f"Generated {len(centrality_dict)} centrality features")

            return centrality_df
        else:
            return pd.DataFrame()

    def compute_community_features(
        self,
        G: nx.Graph,
        methods: Optional[List[str]] = None,
        prefix: str = "community",
    ) -> pd.DataFrame:
        """
        Detect communities and create community-based features.

        Args:
            G: NetworkX graph
            methods: Community detection methods. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with community features

        Example:
            >>> G = nx.karate_club_graph()
            >>> engine = GraphFeatureEngine()
            >>> community_df = engine.compute_community_features(G)
        """
        methods = methods or self.config.community_methods

        logger.info(f"Computing community features using: {methods}")

        community_dict = {}

        # Convert to undirected for community detection
        G_undirected = G.to_undirected() if G.is_directed() else G

        if "louvain" in methods:
            try:
                communities_generator = community.greedy_modularity_communities(
                    G_undirected
                )
                communities = list(communities_generator)

                # Assign community IDs to nodes
                node_to_community = {}
                for comm_id, comm_nodes in enumerate(communities):
                    for node in comm_nodes:
                        node_to_community[node] = comm_id

                community_dict[f"{prefix}_louvain"] = node_to_community

                logger.info(f"Louvain detected {len(communities)} communities")

            except Exception as e:
                logger.warning(f"Louvain community detection failed: {str(e)}")

        if "label_propagation" in methods:
            try:
                communities_generator = community.label_propagation_communities(
                    G_undirected
                )
                communities = list(communities_generator)

                node_to_community = {}
                for comm_id, comm_nodes in enumerate(communities):
                    for node in comm_nodes:
                        node_to_community[node] = comm_id

                community_dict[f"{prefix}_label_prop"] = node_to_community

                logger.info(
                    f"Label propagation detected {len(communities)} communities"
                )

            except Exception as e:
                logger.warning(f"Label propagation failed: {str(e)}")

        # Convert to DataFrame
        if community_dict:
            community_df = pd.DataFrame(community_dict)
            community_df.index.name = self.entity_col
            community_df = community_df.reset_index()

            # Add community size features
            for col in community_df.columns:
                if col != self.entity_col:
                    community_sizes = community_df.groupby(col).size()
                    community_df[f"{col}_size"] = community_df[col].map(community_sizes)

            logger.info(f"Generated {len(community_dict) * 2} community features")

            return community_df
        else:
            return pd.DataFrame()

    def compute_structural_features(
        self,
        G: nx.Graph,
        prefix: str = "struct",
    ) -> pd.DataFrame:
        """
        Compute structural features (clustering coefficient, local density, etc.).

        Args:
            G: NetworkX graph
            prefix: Prefix for feature names

        Returns:
            DataFrame with structural features

        Example:
            >>> G = nx.karate_club_graph()
            >>> engine = GraphFeatureEngine()
            >>> struct_df = engine.compute_structural_features(G)
        """
        logger.info("Computing structural features")

        structural_dict = {}

        # Clustering coefficient
        try:
            structural_dict[f"{prefix}_clustering"] = nx.clustering(G)
        except Exception as e:
            logger.warning(f"Clustering coefficient failed: {str(e)}")

        # Node degree
        try:
            structural_dict[f"{prefix}_degree"] = dict(G.degree())
        except Exception as e:
            logger.warning(f"Degree failed: {str(e)}")

        # Triangles
        try:
            structural_dict[f"{prefix}_triangles"] = nx.triangles(G)
        except Exception as e:
            logger.warning(f"Triangles failed: {str(e)}")

        # Core number (k-core)
        try:
            structural_dict[f"{prefix}_core_number"] = nx.core_number(G)
        except Exception as e:
            logger.warning(f"Core number failed: {str(e)}")

        # Convert to DataFrame
        if structural_dict:
            structural_df = pd.DataFrame(structural_dict)
            structural_df.index.name = self.entity_col
            structural_df = structural_df.reset_index()

            # Add derived features
            if f"{prefix}_degree" in structural_df.columns:
                # Degree statistics in neighborhood
                avg_neighbor_degree = nx.average_neighbor_degree(G)
                structural_df[f"{prefix}_avg_neighbor_degree"] = structural_df[
                    self.entity_col
                ].map(avg_neighbor_degree)

            logger.info(f"Generated {len(structural_dict)} structural features")

            return structural_df
        else:
            return pd.DataFrame()

    def compute_risk_propagation_features(
        self,
        G: nx.Graph,
        risk_scores: Dict[any, float],
        iterations: Optional[int] = None,
        damping: Optional[float] = None,
        prefix: str = "risk_prop",
    ) -> pd.DataFrame:
        """
        Compute risk propagation scores through the network.

        Uses PageRank-like algorithm to propagate risk scores.

        Args:
            G: NetworkX graph
            risk_scores: Dictionary of initial risk scores for nodes
            iterations: Number of propagation iterations. If None, uses config.
            damping: Damping factor for propagation. If None, uses config.
            prefix: Prefix for feature names

        Returns:
            DataFrame with risk propagation features

        Example:
            >>> G = nx.karate_club_graph()
            >>> risk_scores = {0: 1.0, 1: 0.8}  # High risk nodes
            >>> engine = GraphFeatureEngine()
            >>> risk_df = engine.compute_risk_propagation_features(G, risk_scores)
        """
        if not self.config.risk_propagation_enabled:
            logger.info("Risk propagation disabled in config")
            return pd.DataFrame()

        iterations = iterations or self.config.risk_iterations
        damping = damping or self.config.risk_damping

        logger.info(
            f"Computing risk propagation with {iterations} iterations and "
            f"damping={damping}"
        )

        # Initialize risk scores for all nodes
        propagated_risk = {node: risk_scores.get(node, 0.0) for node in G.nodes()}

        # Iterative propagation
        for iteration in range(iterations):
            new_risk = {}

            for node in G.nodes():
                # Get neighbors
                neighbors = list(G.neighbors(node))

                if not neighbors:
                    new_risk[node] = propagated_risk[node]
                    continue

                # Aggregate risk from neighbors
                neighbor_risk = sum(propagated_risk[n] for n in neighbors) / len(
                    neighbors
                )

                # Damped update
                new_risk[node] = (
                    damping * neighbor_risk + (1 - damping) * risk_scores.get(node, 0.0)
                )

            propagated_risk = new_risk

        # Convert to DataFrame
        risk_df = pd.DataFrame(
            {
                self.entity_col: list(propagated_risk.keys()),
                f"{prefix}_score": list(propagated_risk.values()),
            }
        )

        # Add risk change from initial
        risk_df[f"{prefix}_change"] = risk_df.apply(
            lambda row: row[f"{prefix}_score"]
            - risk_scores.get(row[self.entity_col], 0.0),
            axis=1,
        )

        logger.info("Generated 2 risk propagation features")

        return risk_df

    def compute_network_statistics(
        self,
        G: nx.Graph,
        prefix: str = "network",
    ) -> Dict[str, float]:
        """
        Compute global network statistics.

        Args:
            G: NetworkX graph
            prefix: Prefix for feature names

        Returns:
            Dictionary with network statistics

        Example:
            >>> G = nx.karate_club_graph()
            >>> engine = GraphFeatureEngine()
            >>> stats = engine.compute_network_statistics(G)
        """
        logger.info("Computing network statistics")

        stats = {}

        # Basic stats
        stats[f"{prefix}_num_nodes"] = G.number_of_nodes()
        stats[f"{prefix}_num_edges"] = G.number_of_edges()
        stats[f"{prefix}_density"] = nx.density(G)

        # Connected components
        if G.is_directed():
            stats[f"{prefix}_num_components"] = nx.number_weakly_connected_components(G)
        else:
            stats[f"{prefix}_num_components"] = nx.number_connected_components(G)

        # Average clustering
        try:
            stats[f"{prefix}_avg_clustering"] = nx.average_clustering(G)
        except Exception as e:
            logger.warning(f"Average clustering failed: {str(e)}")

        # Transitivity
        try:
            stats[f"{prefix}_transitivity"] = nx.transitivity(G)
        except Exception as e:
            logger.warning(f"Transitivity failed: {str(e)}")

        # Assortativity
        try:
            stats[f"{prefix}_degree_assortativity"] = nx.degree_assortativity_coefficient(
                G
            )
        except Exception as e:
            logger.warning(f"Degree assortativity failed: {str(e)}")

        logger.info(f"Computed {len(stats)} network statistics")

        return stats

    def compute_all_features(
        self,
        G: nx.Graph,
        include_embeddings: bool = True,
        include_centrality: bool = True,
        include_community: bool = True,
        include_structural: bool = True,
        embedding_method: str = "node2vec",
        embedding_dims: int = 128,
    ) -> pd.DataFrame:
        """
        Compute all graph features at once.

        Args:
            G: NetworkX graph
            include_embeddings: Whether to include node embeddings
            include_centrality: Whether to include centrality features
            include_community: Whether to include community features
            include_structural: Whether to include structural features
            embedding_method: Method for node embeddings
            embedding_dims: Dimensionality of embeddings

        Returns:
            DataFrame with all graph features

        Example:
            >>> G = nx.karate_club_graph()
            >>> engine = GraphFeatureEngine()
            >>> all_features = engine.compute_all_features(G)
        """
        logger.info("Computing all graph features")

        # Start with node list
        nodes_df = pd.DataFrame({self.entity_col: list(G.nodes())})

        feature_dfs = [nodes_df]

        if include_centrality:
            centrality_df = self.compute_centrality_features(G)
            if not centrality_df.empty:
                feature_dfs.append(centrality_df)

        if include_community:
            community_df = self.compute_community_features(G)
            if not community_df.empty:
                feature_dfs.append(community_df)

        if include_structural:
            structural_df = self.compute_structural_features(G)
            if not structural_df.empty:
                feature_dfs.append(structural_df)

        if include_embeddings:
            embeddings_df = self.compute_node_embeddings(
                G, method=embedding_method, dimensions=embedding_dims
            )
            if not embeddings_df.empty:
                feature_dfs.append(embeddings_df)

        # Merge all features
        result_df = feature_dfs[0]
        for df in feature_dfs[1:]:
            result_df = result_df.merge(df, on=self.entity_col, how="left")

        logger.info(
            f"Generated {len(result_df.columns) - 1} graph features for "
            f"{len(result_df)} nodes"
        )

        return result_df


if __name__ == "__main__":
    # Example usage
    logger.info("Running graph features example")

    # Create sample graph
    G = nx.karate_club_graph()

    logger.info(
        f"Example graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )

    # Initialize engine
    engine = GraphFeatureEngine()

    # Compute all features
    features_df = engine.compute_all_features(
        G, include_embeddings=False  # Skip embeddings for quick demo
    )

    logger.info(f"Generated features shape: {features_df.shape}")
    logger.info(f"Feature columns: {features_df.columns.tolist()[:10]}")

    # Network statistics
    network_stats = engine.compute_network_statistics(G)
    logger.info(f"Network statistics: {network_stats}")
