"""
Graph Payment Network for Income Estimation
============================================
Detects employer networks, peer income comparison, and income validation
from transaction graphs.

networkx is required: pip install networkx
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class EmployerNode:
    employer_id: str
    name: str
    employee_count: int
    avg_salary: float
    median_salary: float
    salary_std: float
    payment_frequency: str
    confidence: float = 0.0


@dataclass
class PeerComparison:
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
    is_consistent: bool
    consistency_score: float
    employer_match: bool
    peer_deviation: float
    anomaly_score: float
    validation_confidence: float
    flags: List[str]


class PaymentNetworkAnalyzer:
    """
    Analyzes payment graphs for income intelligence.

    Core capabilities:
    - detect_employer_networks: nodes with many outgoing periodic payments
    - infer_from_peer_comparison: Jaccard-similarity peer group income
    - validate_income_with_network: cross-check estimate vs employers/peers
    - calculate_network_stability: employer payment consistency + peer stability
    """

    def __init__(
        self,
        min_employees: int = 3,
        similarity_threshold: float = 0.6,
        anomaly_threshold: float = 3.0,
        entity_col: str = "account_id",
        timestamp_col: str = "business_date",
        amount_col: str = "amount",
        source_col: str = "source",
        target_col: str = "target",
    ):
        self.min_employees         = min_employees
        self.similarity_threshold  = similarity_threshold
        self.anomaly_threshold     = anomaly_threshold
        self.entity_col            = entity_col
        self.timestamp_col         = timestamp_col
        self.amount_col            = amount_col
        self.source_col            = source_col
        self.target_col            = target_col

    def build_payment_network(self, transactions: pd.DataFrame):
        """Build directed payment network (requires networkx)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required: pip install networkx")

        df = transactions.copy()
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], errors="coerce")

        G = nx.DiGraph()

        for _, row in df.iterrows():
            src = row.get(self.source_col) or row.get(self.entity_col)
            tgt = row.get(self.target_col) or row.get(self.entity_col)
            amt = row[self.amount_col]
            ts  = row[self.timestamp_col]

            if pd.isna(src) or pd.isna(tgt):
                continue

            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += abs(float(amt))
                G[src][tgt]["count"]  += 1
                G[src][tgt]["transactions"].append({"amount": float(amt), "timestamp": ts})
            else:
                G.add_edge(src, tgt, weight=abs(float(amt)), count=1,
                           transactions=[{"amount": float(amt), "timestamp": ts}])

        # Mark employer nodes
        for node in G.nodes():
            G.nodes[node]["is_employer"] = G.out_degree(node) >= self.min_employees

        logger.info("Payment network: %d nodes, %d edges",
                    G.number_of_nodes(), G.number_of_edges())
        return G

    def detect_employer_networks(self, graph) -> List[EmployerNode]:
        """Detect employer nodes based on many regular outgoing payments."""
        employers = []

        for node in graph.nodes():
            employees = list(graph.successors(node))
            if len(employees) < self.min_employees:
                continue

            salaries, intervals = [], []

            for emp in employees:
                txns = graph[node][emp].get("transactions", [])
                if len(txns) < 2:
                    continue
                amounts    = [t["amount"] for t in txns]
                salaries.append(np.mean(amounts))
                timestamps = sorted(t["timestamp"] for t in txns)
                intervals += [
                    (timestamps[i+1] - timestamps[i]).days
                    for i in range(len(timestamps) - 1)
                ]

            if len(salaries) < self.min_employees:
                continue

            med_int   = np.median(intervals) if intervals else None
            frequency = (
                "monthly"   if med_int and 27 <= med_int <= 33 else
                "bi-weekly" if med_int and 13 <= med_int <= 17 else
                "weekly"    if med_int and 6  <= med_int <= 8  else "irregular"
            )
            confidence = min(
                len(employees) / (self.min_employees * 3) * 0.5
                + (1.0 if frequency != "irregular" else 0.3) * 0.5,
                1.0,
            )

            employers.append(EmployerNode(
                employer_id=str(node), name=str(node),
                employee_count=len(employees),
                avg_salary=float(np.mean(salaries)),
                median_salary=float(np.median(salaries)),
                salary_std=float(np.std(salaries)),
                payment_frequency=frequency,
                confidence=confidence,
            ))

        employers.sort(key=lambda x: x.employee_count, reverse=True)
        logger.info("Detected %d employer networks", len(employers))
        return employers

    def infer_from_peer_comparison(
        self, graph, user_id: str, user_income: Optional[float] = None, k: int = 10
    ) -> PeerComparison:
        """Find similar users in payment network, use their incomes as reference."""
        if user_id not in graph.nodes():
            return PeerComparison(user_id=user_id, peer_group_size=0,
                                   peer_median_income=0., peer_p25_income=0.,
                                   peer_p75_income=0., user_income_percentile=0.,
                                   similarity_score=0., confidence=0.)

        peers = self._find_similar_peers(graph, user_id, k=k)
        peer_incomes = [
            graph.nodes[pid].get("estimated_income", 0)
            for pid, _ in peers
            if graph.nodes[pid].get("estimated_income", 0) > 0
        ]

        if len(peer_incomes) < 3:
            return PeerComparison(user_id=user_id, peer_group_size=len(peers),
                                   peer_median_income=0., peer_p25_income=0.,
                                   peer_p75_income=0., user_income_percentile=0.,
                                   similarity_score=float(np.mean([s for _, s in peers])) if peers else 0.,
                                   confidence=0.)

        p_med  = float(np.median(peer_incomes))
        p_p25  = float(np.percentile(peer_incomes, 25))
        p_p75  = float(np.percentile(peer_incomes, 75))
        pctile = stats.percentileofscore(peer_incomes, user_income) / 100. if user_income else 0.5
        sim    = float(np.mean([s for _, s in peers]))
        conf   = min(len(peer_incomes) / k * 0.5 + sim * 0.5, 1.0)

        return PeerComparison(user_id=user_id, peer_group_size=len(peers),
                               peer_median_income=p_med, peer_p25_income=p_p25,
                               peer_p75_income=p_p75, user_income_percentile=float(pctile),
                               similarity_score=sim, confidence=float(conf))

    def validate_income_with_network(
        self, estimated_income: float, graph, user_id: str
    ) -> NetworkValidation:
        """Validate income estimate against employer and peer evidence."""
        flags: List[str] = []

        if user_id not in graph.nodes():
            return NetworkValidation(is_consistent=False, consistency_score=0.,
                                      employer_match=False, peer_deviation=0.,
                                      anomaly_score=1., validation_confidence=0.,
                                      flags=["user_not_in_network"])

        # Employer check
        employer_match, employer_income = False, None
        for pred in graph.predecessors(user_id):
            if graph.nodes[pred].get("is_employer", False):
                employer_match = True
                txns = graph[pred][user_id].get("transactions", [])
                if txns:
                    employer_income = float(np.median([t["amount"] for t in txns]))
                break

        if employer_match and employer_income:
            dev = abs(estimated_income - employer_income) / max(employer_income, 1)
            if dev > 0.3:
                flags.append("employer_income_mismatch")

        # Peer deviation
        peer = self.infer_from_peer_comparison(graph, user_id, user_income=estimated_income)
        peer_dev = 0.
        if peer.peer_median_income > 0:
            iqr = peer.peer_p75_income - peer.peer_p25_income
            std = iqr / 1.35 if iqr > 0 else 1.
            peer_dev = (estimated_income - peer.peer_median_income) / std
            if abs(peer_dev) > self.anomaly_threshold:
                flags.append("peer_deviation_high")

        # Anomaly score
        anomaly = self._detect_anomaly(graph, user_id, estimated_income)
        if anomaly > 0.7:
            flags.append("network_anomaly")

        # Consistency
        factors = [1. - anomaly]
        if employer_match and employer_income:
            factors.append(1. - min(abs(estimated_income - employer_income) / max(employer_income, 1), 1.))
        if peer.confidence > 0:
            factors.append(1. - min(abs(peer_dev) / 3., 1.))

        consistency = float(np.mean(factors))
        is_consistent = consistency >= 0.6 and abs(peer_dev) <= self.anomaly_threshold and anomaly <= 0.7

        val_conf = min(peer.confidence * 0.6 + (1. if employer_match else 0.5) * 0.4, 1.)

        return NetworkValidation(is_consistent=is_consistent, consistency_score=consistency,
                                  employer_match=employer_match, peer_deviation=float(peer_dev),
                                  anomaly_score=float(anomaly), validation_confidence=float(val_conf),
                                  flags=flags)

    def calculate_network_stability(self, graph, user_id: str) -> Dict[str, float]:
        """Coefficient-of-variation based employer payment stability."""
        if user_id not in graph.nodes():
            return {"employer_stability": 0., "peer_stability": 0.,
                    "network_stability": 0., "overall_stability": 0.}

        emp_stab = 0.
        for pred in graph.predecessors(user_id):
            if graph.nodes[pred].get("is_employer", False):
                txns = graph[pred][user_id].get("transactions", [])
                if len(txns) >= 3:
                    amts = [t["amount"] for t in txns]
                    cv   = np.std(amts) / max(np.mean(amts), 1e-9)
                    emp_stab = max(emp_stab, 1. - min(cv, 1.))

        peers = self._find_similar_peers(graph, user_id, k=10)
        peer_stabs = [
            graph.nodes[pid].get("income_stability", 0)
            for pid, _ in peers
            if graph.nodes[pid].get("income_stability", 0) > 0
        ]
        peer_stab = float(np.mean(peer_stabs)) if peer_stabs else 0.5

        avg_deg  = np.mean([d for _, d in graph.degree()]) if graph.number_of_nodes() else 1.
        net_stab = 1. - min(abs(graph.degree(user_id) - avg_deg) / max(avg_deg, 1), 1.)

        return {
            "employer_stability": float(emp_stab),
            "peer_stability":     float(peer_stab),
            "network_stability":  float(net_stab),
            "overall_stability":  float(0.5*emp_stab + 0.3*peer_stab + 0.2*net_stab),
        }

    # -- private ---------------------------------------------------------------

    def _find_similar_peers(self, graph, user_id: str, k: int = 10) -> List[Tuple[str, float]]:
        user_nbrs = set(graph.predecessors(user_id)) | set(graph.successors(user_id))
        sims = []
        for node in graph.nodes():
            if node == user_id:
                continue
            nbrs = set(graph.predecessors(node)) | set(graph.successors(node))
            union = len(user_nbrs | nbrs)
            if union > 0:
                sim = len(user_nbrs & nbrs) / union
                if sim >= self.similarity_threshold:
                    sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def _detect_anomaly(self, graph, user_id: str, income: float) -> float:
        nbrs = list(graph.predecessors(user_id)) + list(graph.successors(user_id))
        incomes = [graph.nodes[n].get("estimated_income", 0) for n in nbrs
                   if graph.nodes[n].get("estimated_income", 0) > 0]
        if len(incomes) < 3:
            return 0.5
        z = abs(income - np.mean(incomes)) / max(np.std(incomes), 1e-9)
        return float(min(z / self.anomaly_threshold, 1.))
