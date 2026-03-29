"""
Fraud Detection Graph Builder

Constructs transaction/merchant/device/account graphs with node and edge feature
engineering for graph-based fraud detection.

Example:
    >>> from src.use_cases.fraud_detection.graph_builder import GraphBuilder
    >>> builder = GraphBuilder()
    >>> graph = builder.build_from_transactions(transactions_df)
    >>> builder.save_graph(graph, "fraud_graph.pkl")
"""

import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from datetime import datetime, timedelta

from loguru import logger


class GraphBuilderConfig:
    """Configuration for graph builder."""
    
    def __init__(self):
        """Initialize graph builder configuration."""
        self.node_types = ['account', 'merchant', 'device', 'ip_address']
        self.edge_types = [
            'account_merchant', 'account_device', 'device_ip',
            'account_ip', 'merchant_device'
        ]
        self.time_window_days = 30
        self.min_edge_weight = 1
        self.max_nodes = 1000000


class GraphBuilder:
    """
    Build and maintain fraud detection graphs.
    
    Constructs heterogeneous graphs from transaction data with:
    - Multiple node types (accounts, merchants, devices, IPs)
    - Weighted edges (transaction frequency/amount)
    - Temporal edge features
    - Node features from transaction patterns
    """
    
    def __init__(self, config: Optional[GraphBuilderConfig] = None):
        """
        Initialize graph builder.
        
        Args:
            config: GraphBuilderConfig instance
            
        Example:
            >>> builder = GraphBuilder()
        """
        self.config = config or GraphBuilderConfig()
        self.graph: Optional[nx.Graph] = None
        
        logger.info("Initialized GraphBuilder")
    
    def build_from_transactions(
        self,
        transactions: pd.DataFrame,
        account_col: str = 'account_id',
        merchant_col: str = 'merchant_id',
        device_col: str = 'device_id',
        ip_col: str = 'ip_address',
        amount_col: str = 'amount',
        timestamp_col: str = 'timestamp',
        fraud_col: Optional[str] = 'is_fraud'
    ) -> nx.Graph:
        """
        Build graph from transaction data.
        
        Args:
            transactions: Transaction DataFrame
            account_col: Account ID column
            merchant_col: Merchant ID column
            device_col: Device ID column
            ip_col: IP address column
            amount_col: Transaction amount column
            timestamp_col: Timestamp column
            fraud_col: Fraud label column (optional)
            
        Returns:
            NetworkX graph
            
        Example:
            >>> graph = builder.build_from_transactions(transactions_df)
        """
        logger.info(f"Building graph from {len(transactions)} transactions")
        
        # Filter by time window if needed
        if timestamp_col in transactions.columns:
            transactions = transactions.copy()
            transactions[timestamp_col] = pd.to_datetime(transactions[timestamp_col])
            cutoff_date = transactions[timestamp_col].max() - timedelta(days=self.config.time_window_days)
            transactions = transactions[transactions[timestamp_col] >= cutoff_date]
            logger.info(f"Filtered to {len(transactions)} transactions in last {self.config.time_window_days} days")
        
        # Create graph
        self.graph = nx.Graph()
        
        # Add nodes with attributes
        self._add_nodes(
            transactions,
            account_col, merchant_col, device_col, ip_col,
            amount_col, timestamp_col, fraud_col
        )
        
        # Add edges
        self._add_edges(
            transactions,
            account_col, merchant_col, device_col, ip_col,
            amount_col, timestamp_col
        )
        
        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def _add_nodes(
        self,
        transactions: pd.DataFrame,
        account_col: str,
        merchant_col: str,
        device_col: str,
        ip_col: str,
        amount_col: str,
        timestamp_col: str,
        fraud_col: Optional[str]
    ) -> None:
        """Add nodes with features to graph."""
        # Account nodes
        account_stats = transactions.groupby(account_col).agg({
            amount_col: ['count', 'sum', 'mean', 'std'],
            timestamp_col: ['min', 'max']
        }).reset_index()
        
        account_stats.columns = ['_'.join(col).strip('_') for col in account_stats.columns.values]
        
        if fraud_col and fraud_col in transactions.columns:
            fraud_stats = transactions.groupby(account_col)[fraud_col].agg(['sum', 'mean']).reset_index()
            fraud_stats.columns = [account_col, 'fraud_count', 'fraud_rate']
            account_stats = account_stats.merge(fraud_stats, on=account_col, how='left')
        
        for _, row in account_stats.iterrows():
            node_id = f"account_{row[account_col]}"
            self.graph.add_node(
                node_id,
                node_type='account',
                transaction_count=row.get(f'{amount_col}_count', 0),
                total_amount=row.get(f'{amount_col}_sum', 0),
                avg_amount=row.get(f'{amount_col}_mean', 0),
                std_amount=row.get(f'{amount_col}_std', 0),
                first_seen=row.get(f'{timestamp_col}_min'),
                last_seen=row.get(f'{timestamp_col}_max'),
                fraud_count=row.get('fraud_count', 0),
                fraud_rate=row.get('fraud_rate', 0)
            )
        
        # Merchant nodes
        merchant_stats = transactions.groupby(merchant_col).agg({
            amount_col: ['count', 'sum', 'mean', 'std'],
            account_col: 'nunique'
        }).reset_index()
        
        merchant_stats.columns = ['_'.join(col).strip('_') for col in merchant_stats.columns.values]
        
        if fraud_col and fraud_col in transactions.columns:
            merchant_fraud = transactions.groupby(merchant_col)[fraud_col].agg(['sum', 'mean']).reset_index()
            merchant_fraud.columns = [merchant_col, 'fraud_count', 'fraud_rate']
            merchant_stats = merchant_stats.merge(merchant_fraud, on=merchant_col, how='left')
        
        for _, row in merchant_stats.iterrows():
            node_id = f"merchant_{row[merchant_col]}"
            self.graph.add_node(
                node_id,
                node_type='merchant',
                transaction_count=row.get(f'{amount_col}_count', 0),
                total_amount=row.get(f'{amount_col}_sum', 0),
                avg_amount=row.get(f'{amount_col}_mean', 0),
                unique_accounts=row.get(f'{account_col}_nunique', 0),
                fraud_count=row.get('fraud_count', 0),
                fraud_rate=row.get('fraud_rate', 0)
            )
        
        # Device nodes
        if device_col in transactions.columns:
            device_stats = transactions.groupby(device_col).agg({
                account_col: 'nunique',
                amount_col: ['count', 'sum']
            }).reset_index()
            
            device_stats.columns = ['_'.join(col).strip('_') for col in device_stats.columns.values]
            
            for _, row in device_stats.iterrows():
                node_id = f"device_{row[device_col]}"
                self.graph.add_node(
                    node_id,
                    node_type='device',
                    unique_accounts=row.get(f'{account_col}_nunique', 0),
                    transaction_count=row.get(f'{amount_col}_count', 0),
                    total_amount=row.get(f'{amount_col}_sum', 0)
                )
        
        # IP address nodes
        if ip_col in transactions.columns:
            ip_stats = transactions.groupby(ip_col).agg({
                account_col: 'nunique',
                amount_col: 'count'
            }).reset_index()
            
            ip_stats.columns = ['_'.join(col).strip('_') for col in ip_stats.columns.values]
            
            for _, row in ip_stats.iterrows():
                node_id = f"ip_{row[ip_col]}"
                self.graph.add_node(
                    node_id,
                    node_type='ip_address',
                    unique_accounts=row.get(f'{account_col}_nunique', 0),
                    transaction_count=row.get(f'{amount_col}_count', 0)
                )
    
    def _add_edges(
        self,
        transactions: pd.DataFrame,
        account_col: str,
        merchant_col: str,
        device_col: str,
        ip_col: str,
        amount_col: str,
        timestamp_col: str
    ) -> None:
        """Add edges to graph."""
        # Account-Merchant edges
        am_edges = transactions.groupby([account_col, merchant_col]).agg({
            amount_col: ['count', 'sum', 'mean'],
            timestamp_col: ['min', 'max']
        }).reset_index()
        
        am_edges.columns = ['_'.join(col).strip('_') for col in am_edges.columns.values]
        
        for _, row in am_edges.iterrows():
            account_node = f"account_{row[account_col]}"
            merchant_node = f"merchant_{row[merchant_col]}"
            
            if self.graph.has_node(account_node) and self.graph.has_node(merchant_node):
                weight = row.get(f'{amount_col}_count', 1)
                if weight >= self.config.min_edge_weight:
                    self.graph.add_edge(
                        account_node,
                        merchant_node,
                        edge_type='account_merchant',
                        weight=weight,
                        total_amount=row.get(f'{amount_col}_sum', 0),
                        avg_amount=row.get(f'{amount_col}_mean', 0),
                        first_transaction=row.get(f'{timestamp_col}_min'),
                        last_transaction=row.get(f'{timestamp_col}_max')
                    )
        
        # Account-Device edges
        if device_col in transactions.columns:
            ad_edges = transactions.groupby([account_col, device_col]).agg({
                amount_col: 'count'
            }).reset_index()
            
            ad_edges.columns = [account_col, device_col, 'count']
            
            for _, row in ad_edges.iterrows():
                account_node = f"account_{row[account_col]}"
                device_node = f"device_{row[device_col]}"
                
                if self.graph.has_node(account_node) and self.graph.has_node(device_node):
                    self.graph.add_edge(
                        account_node,
                        device_node,
                        edge_type='account_device',
                        weight=row['count']
                    )
        
        # Account-IP edges
        if ip_col in transactions.columns:
            ai_edges = transactions.groupby([account_col, ip_col]).agg({
                amount_col: 'count'
            }).reset_index()
            
            ai_edges.columns = [account_col, ip_col, 'count']
            
            for _, row in ai_edges.iterrows():
                account_node = f"account_{row[account_col]}"
                ip_node = f"ip_{row[ip_col]}"
                
                if self.graph.has_node(account_node) and self.graph.has_node(ip_node):
                    self.graph.add_edge(
                        account_node,
                        ip_node,
                        edge_type='account_ip',
                        weight=row['count']
                    )
        
        # Device-IP edges
        if device_col in transactions.columns and ip_col in transactions.columns:
            di_edges = transactions.groupby([device_col, ip_col]).agg({
                amount_col: 'count'
            }).reset_index()
            
            di_edges.columns = [device_col, ip_col, 'count']
            
            for _, row in di_edges.iterrows():
                device_node = f"device_{row[device_col]}"
                ip_node = f"ip_{row[ip_col]}"
                
                if self.graph.has_node(device_node) and self.graph.has_node(ip_node):
                    self.graph.add_edge(
                        device_node,
                        ip_node,
                        edge_type='device_ip',
                        weight=row['count']
                    )
    
    def update_graph(
        self,
        new_transactions: pd.DataFrame,
        **kwargs
    ) -> nx.Graph:
        """
        Update existing graph with new transactions.
        
        Args:
            new_transactions: New transaction data
            **kwargs: Column name arguments (same as build_from_transactions)
            
        Returns:
            Updated graph
            
        Example:
            >>> builder.update_graph(new_transactions_df)
        """
        if self.graph is None:
            logger.warning("No existing graph, building from scratch")
            return self.build_from_transactions(new_transactions, **kwargs)
        
        logger.info(f"Updating graph with {len(new_transactions)} new transactions")
        
        # Build temporary graph from new transactions
        temp_builder = GraphBuilder(self.config)
        temp_graph = temp_builder.build_from_transactions(new_transactions, **kwargs)
        
        # Merge graphs
        for node, attrs in temp_graph.nodes(data=True):
            if self.graph.has_node(node):
                # Update existing node attributes
                for key, value in attrs.items():
                    if key in ['transaction_count', 'total_amount', 'fraud_count']:
                        self.graph.nodes[node][key] = self.graph.nodes[node].get(key, 0) + value
                    elif key == 'last_seen':
                        if value > self.graph.nodes[node].get(key, datetime.min):
                            self.graph.nodes[node][key] = value
            else:
                # Add new node
                self.graph.add_node(node, **attrs)
        
        # Merge edges
        for u, v, attrs in temp_graph.edges(data=True):
            if self.graph.has_edge(u, v):
                # Update edge weight
                self.graph[u][v]['weight'] = self.graph[u][v].get('weight', 0) + attrs.get('weight', 1)
            else:
                # Add new edge
                self.graph.add_edge(u, v, **attrs)
        
        logger.info(
            f"Graph updated: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def get_node_features(self, node_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract node features as DataFrame.
        
        Args:
            node_ids: Specific node IDs (optional, defaults to all)
            
        Returns:
            DataFrame with node features
            
        Example:
            >>> features = builder.get_node_features(['account_123', 'account_456'])
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_from_transactions() first.")
        
        if node_ids is None:
            node_ids = list(self.graph.nodes())
        
        features = []
        for node_id in node_ids:
            if not self.graph.has_node(node_id):
                logger.warning(f"Node {node_id} not in graph")
                continue
            
            attrs = self.graph.nodes[node_id]
            feature = {'node_id': node_id, **attrs}
            
            # Add degree features
            feature['degree'] = self.graph.degree(node_id)
            feature['weighted_degree'] = sum(
                self.graph[node_id][neighbor].get('weight', 1)
                for neighbor in self.graph.neighbors(node_id)
            )
            
            features.append(feature)
        
        return pd.DataFrame(features)
    
    def get_subgraph(
        self,
        center_nodes: List[str],
        hops: int = 2
    ) -> nx.Graph:
        """
        Extract k-hop subgraph around center nodes.
        
        Args:
            center_nodes: Center node IDs
            hops: Number of hops
            
        Returns:
            Subgraph
            
        Example:
            >>> subgraph = builder.get_subgraph(['account_123'], hops=2)
        """
        if self.graph is None:
            raise ValueError("Graph not built.")
        
        # Get all nodes within k hops
        nodes_to_include = set(center_nodes)
        
        for _ in range(hops):
            new_nodes = set()
            for node in nodes_to_include:
                if self.graph.has_node(node):
                    new_nodes.update(self.graph.neighbors(node))
            nodes_to_include.update(new_nodes)
        
        return self.graph.subgraph(nodes_to_include).copy()
    
    def save_graph(self, filepath: Union[str, Path]) -> None:
        """
        Save graph to disk.
        
        Args:
            filepath: Path to save graph
            
        Example:
            >>> builder.save_graph("graphs/fraud_graph.pkl")
        """
        if self.graph is None:
            raise ValueError("No graph to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'graph': self.graph,
            'config': self.config
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"Graph saved to {filepath}")
    
    @classmethod
    def load_graph(cls, filepath: Union[str, Path]) -> 'GraphBuilder':
        """
        Load graph from disk.
        
        Args:
            filepath: Path to saved graph
            
        Returns:
            GraphBuilder instance with loaded graph
            
        Example:
            >>> builder = GraphBuilder.load_graph("graphs/fraud_graph.pkl")
        """
        save_dict = joblib.load(filepath)
        
        instance = cls(config=save_dict['config'])
        instance.graph = save_dict['graph']
        
        logger.info(f"Graph loaded from {filepath}")
        return instance
