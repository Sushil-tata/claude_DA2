"""
Payment Network Intelligence for Income Estimation

Implements graph-based analysis for income inference:
- Employer network detection from common payment sources
- Peer comparison for income inference
- Network-based income validation
- Industry/sector inference from payment patterns
- Graph-based income stability indicators
- Cross-validation with deposit patterns
- Network anomaly detection
- Trust propagation through payment networks

Example:
    >>> from src.use_cases.income_estimation.graph_payment import PaymentNetworkAnalyzer
    >>> 
    >>> analyzer = PaymentNetworkAnalyzer()
    >>> graph = analyzer.build_payment_network(transactions_df)
    >>> employers = analyzer.detect_employer_networks(graph)
    >>> peer_income = analyzer.infer_from_peer_comparison(graph, user_id)
    >>> validation = analyzer.validate_income_with_network(deposits, graph, user_id)

Author: Principal Data Science Decision Agent
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
import networkx as nx
from scipy import stats


@dataclass
class EmployerNode:
    """Container for employer node information."""
    
    employer_id: str
    name: str
    employee_count: int
    avg_salary: float
    median_salary: float
    salary_std: float
    payment_frequency: str  # 'monthly', 'bi-weekly', 'weekly'
    industry: Optional[str] = None
    confidence: float = 0.0
    is_verified: bool = False


@dataclass
class PeerComparison:
    """Container for peer comparison results."""
    
    user_id: str
    peer_group_size: int
    peer_median_income: float
    peer_p25_income: float
    peer_p75_income: float
    user_income_percentile: float
    similarity_score: float
    confidence: float


@dataclass
class NetworkValidation:
    """Container for network-based validation results."""
    
    is_consistent: bool
    consistency_score: float
    employer_match: bool
    peer_deviation: float  # Standard deviations from peer median
    anomaly_score: float
    validation_confidence: float
    flags: List[str]


class PaymentNetworkAnalyzer:
    """
    Analyze payment networks for income intelligence.
    
    Uses graph analysis to detect employer networks, compare with peers,
    and validate income estimates through network patterns.
    """
    
    def __init__(
        self,
        min_employees: int = 3,
        similarity_threshold: float = 0.6,
        anomaly_threshold: float = 3.0,
        entity_col: str = "user_id",
        timestamp_col: str = "timestamp",
        amount_col: str = "amount",
        source_col: str = "source",
        target_col: str = "target",
        description_col: str = "description",
    ):
        """
        Initialize payment network analyzer.
        
        Args:
            min_employees: Minimum employees to classify as employer
            similarity_threshold: Minimum similarity for peer grouping
            anomaly_threshold: Threshold (std devs) for anomaly detection
            entity_col: Column name for entity/user ID
            timestamp_col: Column name for timestamp
            amount_col: Column name for amount
            source_col: Column name for payment source
            target_col: Column name for payment target
            description_col: Column name for description
        """
        self.min_employees = min_employees
        self.similarity_threshold = similarity_threshold
        self.anomaly_threshold = anomaly_threshold
        
        self.entity_col = entity_col
        self.timestamp_col = timestamp_col
        self.amount_col = amount_col
        self.source_col = source_col
        self.target_col = target_col
        self.description_col = description_col
        
        logger.info("PaymentNetworkAnalyzer initialized")
    
    def build_payment_network(
        self,
        transactions: pd.DataFrame,
        include_attributes: bool = True
    ) -> nx.DiGraph:
        """
        Build directed payment network from transactions.
        
        Args:
            transactions: Transaction data
            include_attributes: Whether to compute node attributes
            
        Returns:
            NetworkX directed graph
            
        Example:
            >>> graph = analyzer.build_payment_network(transactions_df)
            >>> print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        """
        logger.info("Building payment network...")
        
        df = transactions.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.timestamp_col]):
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges from transactions
        for _, row in df.iterrows():
            source = row.get(self.source_col, row.get(self.entity_col))
            target = row.get(self.target_col, row.get(self.entity_col))
            amount = row[self.amount_col]
            timestamp = row[self.timestamp_col]
            
            if pd.isna(source) or pd.isna(target):
                continue
            
            # Add or update edge
            if G.has_edge(source, target):
                G[source][target]['weight'] += abs(amount)
                G[source][target]['count'] += 1
                G[source][target]['transactions'].append({
                    'amount': amount,
                    'timestamp': timestamp
                })
            else:
                G.add_edge(
                    source,
                    target,
                    weight=abs(amount),
                    count=1,
                    transactions=[{
                        'amount': amount,
                        'timestamp': timestamp
                    }]
                )
        
        if include_attributes:
            # Compute node attributes
            self._compute_node_attributes(G)
        
        logger.info(
            f"Built payment network: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        
        return G
    
    def detect_employer_networks(
        self,
        graph: nx.DiGraph,
        deposits_df: Optional[pd.DataFrame] = None
    ) -> List[EmployerNode]:
        """
        Detect employer nodes in payment network.
        
        Args:
            graph: Payment network graph
            deposits_df: Optional deposit data for validation
            
        Returns:
            List of detected employer nodes
            
        Example:
            >>> employers = analyzer.detect_employer_networks(graph)
            >>> for emp in employers:
            ...     print(f"{emp.name}: {emp.employee_count} employees, "
            ...           f"avg ${emp.avg_salary:.2f}")
        """
        logger.info("Detecting employer networks...")
        
        employers = []
        
        # Find nodes with many outgoing edges (potential employers)
        for node in graph.nodes():
            # Get employees (nodes this node pays)
            employees = list(graph.successors(node))
            
            if len(employees) < self.min_employees:
                continue
            
            # Analyze payment patterns to employees
            salaries = []
            payment_intervals = []
            
            for employee in employees:
                edge_data = graph[node][employee]
                transactions = edge_data.get('transactions', [])
                
                if len(transactions) < 2:
                    continue
                
                # Calculate average payment
                amounts = [t['amount'] for t in transactions]
                avg_payment = np.mean(amounts)
                salaries.append(avg_payment)
                
                # Calculate payment frequency
                timestamps = sorted([t['timestamp'] for t in transactions])
                if len(timestamps) >= 2:
                    intervals = [
                        (timestamps[i+1] - timestamps[i]).days
                        for i in range(len(timestamps) - 1)
                    ]
                    if intervals:
                        payment_intervals.extend(intervals)
            
            if len(salaries) < self.min_employees:
                continue
            
            # Determine payment frequency
            if payment_intervals:
                median_interval = np.median(payment_intervals)
                if 27 <= median_interval <= 33:
                    frequency = 'monthly'
                elif 13 <= median_interval <= 17:
                    frequency = 'bi-weekly'
                elif 6 <= median_interval <= 8:
                    frequency = 'weekly'
                else:
                    frequency = 'irregular'
            else:
                frequency = 'unknown'
            
            # Calculate confidence
            # Higher confidence for more employees and regular payments
            confidence = min(
                (len(employees) / (self.min_employees * 3)) * 0.5 +
                (1.0 if frequency in ['monthly', 'bi-weekly', 'weekly'] else 0.3) * 0.5,
                1.0
            )
            
            # Create employer node
            employer = EmployerNode(
                employer_id=str(node),
                name=str(node),
                employee_count=len(employees),
                avg_salary=float(np.mean(salaries)),
                median_salary=float(np.median(salaries)),
                salary_std=float(np.std(salaries)),
                payment_frequency=frequency,
                confidence=confidence
            )
            
            employers.append(employer)
        
        # Sort by employee count
        employers.sort(key=lambda x: x.employee_count, reverse=True)
        
        logger.info(f"Detected {len(employers)} potential employers")
        
        return employers
    
    def infer_from_peer_comparison(
        self,
        graph: nx.DiGraph,
        user_id: str,
        user_income: Optional[float] = None,
        k_neighbors: int = 10
    ) -> PeerComparison:
        """
        Infer income from peer comparison using network similarity.
        
        Args:
            graph: Payment network graph
            user_id: User ID to analyze
            user_income: Known income (optional, for validation)
            k_neighbors: Number of similar peers to compare
            
        Returns:
            PeerComparison object
            
        Example:
            >>> peer_comp = analyzer.infer_from_peer_comparison(graph, "user_123")
            >>> print(f"Peer median: ${peer_comp.peer_median_income:.2f}")
            >>> print(f"Your percentile: {peer_comp.user_income_percentile:.0%}")
        """
        if user_id not in graph.nodes():
            logger.warning(f"User {user_id} not found in graph")
            return PeerComparison(
                user_id=user_id,
                peer_group_size=0,
                peer_median_income=0.0,
                peer_p25_income=0.0,
                peer_p75_income=0.0,
                user_income_percentile=0.0,
                similarity_score=0.0,
                confidence=0.0
            )
        
        # Find similar peers using network structure
        similar_peers = self._find_similar_peers(graph, user_id, k=k_neighbors)
        
        if len(similar_peers) == 0:
            logger.warning(f"No similar peers found for {user_id}")
            return PeerComparison(
                user_id=user_id,
                peer_group_size=0,
                peer_median_income=0.0,
                peer_p25_income=0.0,
                peer_p75_income=0.0,
                user_income_percentile=0.0,
                similarity_score=0.0,
                confidence=0.0
            )
        
        # Extract peer incomes from node attributes
        peer_incomes = []
        for peer_id, similarity in similar_peers:
            if peer_id in graph.nodes():
                income = graph.nodes[peer_id].get('estimated_income', 0)
                if income > 0:
                    peer_incomes.append(income)
        
        if len(peer_incomes) < 3:
            logger.warning(f"Insufficient peer income data for {user_id}")
            avg_similarity = np.mean([s for _, s in similar_peers])
            return PeerComparison(
                user_id=user_id,
                peer_group_size=len(similar_peers),
                peer_median_income=0.0,
                peer_p25_income=0.0,
                peer_p75_income=0.0,
                user_income_percentile=0.0,
                similarity_score=float(avg_similarity),
                confidence=0.0
            )
        
        # Calculate peer statistics
        peer_median = float(np.median(peer_incomes))
        peer_p25 = float(np.percentile(peer_incomes, 25))
        peer_p75 = float(np.percentile(peer_incomes, 75))
        
        # Calculate user's percentile if income provided
        if user_income:
            percentile = stats.percentileofscore(peer_incomes, user_income) / 100.0
        else:
            percentile = 0.5  # Assume median if unknown
        
        # Calculate average similarity
        avg_similarity = np.mean([s for _, s in similar_peers])
        
        # Confidence based on peer group size and similarity
        confidence = min(
            (len(peer_incomes) / k_neighbors) * 0.5 +
            avg_similarity * 0.5,
            1.0
        )
        
        return PeerComparison(
            user_id=user_id,
            peer_group_size=len(similar_peers),
            peer_median_income=peer_median,
            peer_p25_income=peer_p25,
            peer_p75_income=peer_p75,
            user_income_percentile=float(percentile),
            similarity_score=float(avg_similarity),
            confidence=float(confidence)
        )
    
    def validate_income_with_network(
        self,
        estimated_income: float,
        graph: nx.DiGraph,
        user_id: str,
        deposits_df: Optional[pd.DataFrame] = None
    ) -> NetworkValidation:
        """
        Validate income estimate using network patterns.
        
        Args:
            estimated_income: Estimated income to validate
            graph: Payment network graph
            user_id: User ID
            deposits_df: Optional deposit data
            
        Returns:
            NetworkValidation object
            
        Example:
            >>> validation = analyzer.validate_income_with_network(
            ...     50000, graph, "user_123"
            ... )
            >>> if validation.is_consistent:
            ...     print("Income estimate validated by network")
        """
        flags = []
        
        if user_id not in graph.nodes():
            logger.warning(f"User {user_id} not in graph")
            return NetworkValidation(
                is_consistent=False,
                consistency_score=0.0,
                employer_match=False,
                peer_deviation=0.0,
                anomaly_score=1.0,
                validation_confidence=0.0,
                flags=['user_not_in_network']
            )
        
        # 1. Check employer match
        employer_match = False
        employer_income = None
        
        # Find if user receives payments from employer
        predecessors = list(graph.predecessors(user_id))
        for pred in predecessors:
            if graph.nodes[pred].get('is_employer', False):
                employer_match = True
                # Get expected income from employer
                edge_data = graph[pred][user_id]
                transactions = edge_data.get('transactions', [])
                if transactions:
                    amounts = [t['amount'] for t in transactions]
                    employer_income = np.median(amounts)
                break
        
        # 2. Compare with peer group
        peer_comp = self.infer_from_peer_comparison(
            graph, user_id, user_income=estimated_income
        )
        
        peer_deviation = 0.0
        if peer_comp.peer_median_income > 0:
            peer_std = (
                peer_comp.peer_p75_income - peer_comp.peer_p25_income
            ) / 1.35  # Approximate std from IQR
            
            if peer_std > 0:
                peer_deviation = (
                    estimated_income - peer_comp.peer_median_income
                ) / peer_std
            
            # Flag if too far from peers
            if abs(peer_deviation) > self.anomaly_threshold:
                flags.append('peer_deviation_high')
        
        # 3. Check employer vs peer consistency
        if employer_match and employer_income:
            employer_deviation = abs(estimated_income - employer_income) / employer_income
            if employer_deviation > 0.3:  # 30% difference
                flags.append('employer_income_mismatch')
        
        # 4. Network anomaly detection
        anomaly_score = self._detect_network_anomaly(graph, user_id, estimated_income)
        if anomaly_score > 0.7:
            flags.append('network_anomaly_detected')
        
        # 5. Calculate consistency score
        consistency_factors = []
        
        # Employer consistency
        if employer_match and employer_income:
            employer_consistency = 1.0 - min(
                abs(estimated_income - employer_income) / employer_income,
                1.0
            )
            consistency_factors.append(employer_consistency)
        
        # Peer consistency
        if peer_comp.confidence > 0:
            peer_consistency = 1.0 - min(abs(peer_deviation) / 3.0, 1.0)
            consistency_factors.append(peer_consistency)
        
        # Anomaly consistency
        consistency_factors.append(1.0 - anomaly_score)
        
        consistency_score = float(np.mean(consistency_factors)) if consistency_factors else 0.0
        
        # Overall validation
        is_consistent = (
            consistency_score >= 0.6 and
            abs(peer_deviation) <= self.anomaly_threshold and
            anomaly_score <= 0.7
        )
        
        # Validation confidence
        validation_confidence = min(
            peer_comp.confidence * 0.6 +
            (1.0 if employer_match else 0.5) * 0.4,
            1.0
        )
        
        return NetworkValidation(
            is_consistent=is_consistent,
            consistency_score=consistency_score,
            employer_match=employer_match,
            peer_deviation=float(peer_deviation),
            anomaly_score=float(anomaly_score),
            validation_confidence=float(validation_confidence),
            flags=flags
        )
    
    def infer_industry_sector(
        self,
        graph: nx.DiGraph,
        user_id: str,
        employer_nodes: Optional[List[EmployerNode]] = None
    ) -> Dict[str, float]:
        """
        Infer industry/sector from payment patterns.
        
        Args:
            graph: Payment network graph
            user_id: User ID
            employer_nodes: Optional list of employer nodes
            
        Returns:
            Dictionary of industry probabilities
            
        Example:
            >>> industries = analyzer.infer_industry_sector(graph, "user_123")
            >>> print(f"Most likely: {max(industries, key=industries.get)}")
        """
        if user_id not in graph.nodes():
            return {}
        
        # Industry inference based on:
        # 1. Employer patterns
        # 2. Peer industry distribution
        # 3. Spending patterns (if available)
        
        industry_scores = defaultdict(float)
        
        # Check employer industry
        predecessors = list(graph.predecessors(user_id))
        for pred in predecessors:
            industry = graph.nodes[pred].get('industry')
            if industry:
                industry_scores[industry] += 0.5
        
        # Check peer industries
        similar_peers = self._find_similar_peers(graph, user_id, k=10)
        for peer_id, similarity in similar_peers:
            industry = graph.nodes[peer_id].get('industry')
            if industry:
                industry_scores[industry] += similarity * 0.3
        
        # Normalize to probabilities
        total = sum(industry_scores.values())
        if total > 0:
            industry_probs = {
                k: v / total for k, v in industry_scores.items()
            }
        else:
            industry_probs = {}
        
        return industry_probs
    
    def calculate_network_stability(
        self,
        graph: nx.DiGraph,
        user_id: str,
        lookback_months: int = 6
    ) -> Dict[str, float]:
        """
        Calculate income stability indicators from network.
        
        Args:
            graph: Payment network graph
            user_id: User ID
            lookback_months: Months to analyze
            
        Returns:
            Dictionary of stability metrics
            
        Example:
            >>> stability = analyzer.calculate_network_stability(graph, "user_123")
            >>> print(f"Employer stability: {stability['employer_stability']:.2f}")
        """
        if user_id not in graph.nodes():
            return {
                'employer_stability': 0.0,
                'peer_stability': 0.0,
                'network_stability': 0.0
            }
        
        # 1. Employer stability (consistency of employer payments)
        employer_stability = 0.0
        predecessors = list(graph.predecessors(user_id))
        
        for pred in predecessors:
            if graph.nodes[pred].get('is_employer', False):
                edge_data = graph[pred][user_id]
                transactions = edge_data.get('transactions', [])
                
                if len(transactions) >= 3:
                    amounts = [t['amount'] for t in transactions]
                    cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 1.0
                    employer_stability = max(employer_stability, 1.0 - min(cv, 1.0))
        
        # 2. Peer stability (how stable is peer group)
        similar_peers = self._find_similar_peers(graph, user_id, k=10)
        peer_stabilities = []
        
        for peer_id, _ in similar_peers:
            peer_stability = graph.nodes[peer_id].get('income_stability', 0)
            if peer_stability > 0:
                peer_stabilities.append(peer_stability)
        
        peer_stability = float(np.mean(peer_stabilities)) if peer_stabilities else 0.5
        
        # 3. Network stability (overall network characteristics)
        # Stable networks have consistent degree and edge weights
        degree = graph.degree(user_id)
        avg_degree = np.mean([d for _, d in graph.degree()])
        
        network_stability = 1.0 - min(
            abs(degree - avg_degree) / max(avg_degree, 1),
            1.0
        )
        
        return {
            'employer_stability': float(employer_stability),
            'peer_stability': float(peer_stability),
            'network_stability': float(network_stability),
            'overall_stability': float(
                0.5 * employer_stability +
                0.3 * peer_stability +
                0.2 * network_stability
            )
        }
    
    # ==================== Private Helper Methods ====================
    
    def _compute_node_attributes(self, graph: nx.DiGraph) -> None:
        """Compute and store node attributes."""
        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        
        # PageRank
        try:
            pagerank = nx.pagerank(graph)
        except:
            pagerank = {node: 0.0 for node in graph.nodes()}
        
        # Set attributes
        for node in graph.nodes():
            graph.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
            graph.nodes[node]['pagerank'] = pagerank.get(node, 0)
            
            # Classify as employer if has many outgoing edges
            out_degree = graph.out_degree(node)
            graph.nodes[node]['is_employer'] = out_degree >= self.min_employees
    
    def _find_similar_peers(
        self,
        graph: nx.DiGraph,
        user_id: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find k most similar peers using network structure."""
        if user_id not in graph.nodes():
            return []
        
        # Get user's neighbors
        user_neighbors = set(graph.predecessors(user_id)) | set(graph.successors(user_id))
        
        # Calculate Jaccard similarity with other nodes
        similarities = []
        
        for node in graph.nodes():
            if node == user_id:
                continue
            
            # Get node's neighbors
            node_neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
            
            # Calculate Jaccard similarity
            if len(user_neighbors | node_neighbors) > 0:
                similarity = len(user_neighbors & node_neighbors) / len(user_neighbors | node_neighbors)
                
                if similarity >= self.similarity_threshold:
                    similarities.append((node, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def _detect_network_anomaly(
        self,
        graph: nx.DiGraph,
        user_id: str,
        estimated_income: float
    ) -> float:
        """Detect if user's income is anomalous in network context."""
        if user_id not in graph.nodes():
            return 1.0
        
        # Get local neighborhood incomes
        neighbors = list(graph.predecessors(user_id)) | list(graph.successors(user_id))
        
        neighbor_incomes = []
        for neighbor in neighbors:
            income = graph.nodes[neighbor].get('estimated_income', 0)
            if income > 0:
                neighbor_incomes.append(income)
        
        if len(neighbor_incomes) < 3:
            return 0.5  # Insufficient data
        
        # Calculate z-score
        mean_income = np.mean(neighbor_incomes)
        std_income = np.std(neighbor_incomes)
        
        if std_income > 0:
            z_score = abs(estimated_income - mean_income) / std_income
            # Normalize to 0-1
            anomaly_score = min(z_score / self.anomaly_threshold, 1.0)
        else:
            anomaly_score = 0.0
        
        return float(anomaly_score)
