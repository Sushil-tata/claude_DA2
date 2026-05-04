"""
Customer Level Coordinator
===========================
Resolves channel conflicts when one customer has multiple products (CC + SPC).

Design answers (configurable via CoordinatorConfig):
  Q1 - Channel conflict:  mode="customer" → coordinator picks 1 best action per channel
  Q2 - Fatigue scope:     fatigue_scope="product" → each product tracks fatigue independently
  Q3 - Action selection:  action_selection="max_value" → highest net_expected_value wins

All three answers are YAML-configurable — change without touching code.

Usage:
    config = CoordinatorConfig()                  # defaults from your answers
    coord  = CustomerLevelCoordinator(config)

    # decisions_df: output of DecisionEngine.decide_batch() for each product
    final  = coord.coordinate(decisions_df)       # one row per customer per channel
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoordinatorConfig:
    """
    All coordination rules in one place.
    Load from YAML via CoordinatorConfig(**yaml_config["coordination"]).
    """

    # Q1: how to resolve channel conflicts across products
    mode: str = "customer"          # "product" | "customer"

    # Q2: fatigue tracked per product or shared across all customer products
    fatigue_scope: str = "product"  # "product" | "customer"

    # Q3: action selection when products conflict
    action_selection: str = "max_value"  # "max_value" | "consensus"

    # Future value extraction levers (flip without code changes)
    cross_product_agent_call: bool = True   # agent call covers CC+SPC together
    channel_dedup: bool = True              # never same channel twice same day

    # Channel rank order for tie-breaking (lower index = higher priority)
    channel_priority: List[str] = field(default_factory=lambda: [
        "AGENT_CALL",
        "VOICE_IVR",
        "LINE",
        "SMS",
        "EMAIL",
        "SETTLEMENT_ONE_TIME",
        "SETTLEMENT_PAYMENT_PLAN",
        "NO_ACTION",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATOR
# ─────────────────────────────────────────────────────────────────────────────

class CustomerLevelCoordinator:
    """
    Resolves multi-product NBA decisions at customer level.

    Input:
        decisions_df — concatenated output of DecisionEngine.decide_batch()
                       for all products. Required columns:
                       customer_id, account_id, product (CC|SPC),
                       recommended_action, net_expected_value,
                       pay_any_score, expected_amount, uplift_score,
                       blocked (bool), blocked_reasons

    Output:
        DataFrame with one final decision per customer per channel slot,
        tagged with source product and coordination metadata.
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()
        logger.info(
            "CustomerLevelCoordinator init: mode=%s, fatigue_scope=%s, action_selection=%s",
            self.config.mode, self.config.fatigue_scope, self.config.action_selection,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────────

    def coordinate(self, decisions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main entry point. Returns coordinated decisions.

        Args:
            decisions_df: All product-level NBA decisions for all customers.
                          Must include columns: customer_id, account_id, product,
                          recommended_action, net_expected_value, blocked.

        Returns:
            DataFrame: one final action per customer (or per customer per channel
                       if cross_product_agent_call=True and mode=customer).
        """
        self._validate_input(decisions_df)

        if self.config.mode == "product":
            # No coordination — pass through as-is
            logger.info("mode=product: returning decisions unchanged.")
            return decisions_df.copy()

        # mode = "customer"
        eligible = decisions_df[~decisions_df["blocked"]].copy()
        blocked  = decisions_df[decisions_df["blocked"]].copy()

        if eligible.empty:
            logger.warning("All decisions are blocked — nothing to coordinate.")
            return decisions_df.copy()

        coordinated = self._coordinate_customer(eligible)

        # Re-attach blocked rows for audit trail
        result = pd.concat([coordinated, blocked], ignore_index=True)
        result = result.sort_values(["customer_id", "product"]).reset_index(drop=True)

        logger.info(
            "Coordination complete: %d customers, %d final actions",
            result["customer_id"].nunique(),
            len(result[~result.get("blocked", pd.Series(False, index=result.index))]),
        )
        return result

    def summarise(self, coordinated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Human-readable summary: one row per customer showing final action per product.
        """
        cols = ["customer_id", "product", "account_id",
                "recommended_action", "net_expected_value",
                "coordination_note", "blocked"]
        available = [c for c in cols if c in coordinated_df.columns]
        return (
            coordinated_df[available]
            .sort_values(["customer_id", "product"])
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — COORDINATION LOGIC
    # ──────────────────────────────────────────────────────────────────────────

    def _coordinate_customer(self, eligible_df: pd.DataFrame) -> pd.DataFrame:
        """Apply Q1/Q2/Q3 logic per customer group."""
        results = []

        for customer_id, grp in eligible_df.groupby("customer_id"):
            products = grp["product"].unique() if "product" in grp.columns else ["unknown"]

            if len(products) == 1:
                # Single product — no conflict possible
                rows = grp.copy()
                rows["coordination_note"] = "single_product"
                results.append(rows)
                continue

            # Multi-product customer → apply coordination
            coordinated = self._resolve_conflict(grp, customer_id)
            results.append(coordinated)

        if not results:
            return eligible_df.copy()

        return pd.concat(results, ignore_index=True)

    def _resolve_conflict(self, grp: pd.DataFrame, customer_id: str) -> pd.DataFrame:
        """
        Resolve channel conflicts for a multi-product customer.

        Q3 (action_selection=max_value): highest net_expected_value wins per channel.
        Q1 (channel_dedup=True): same channel can only be assigned once per day.
        """
        grp = grp.copy()

        if self.config.action_selection == "max_value":
            return self._resolve_max_value(grp, customer_id)
        elif self.config.action_selection == "consensus":
            return self._resolve_consensus(grp, customer_id)
        else:
            logger.warning("Unknown action_selection=%s — using max_value.",
                           self.config.action_selection)
            return self._resolve_max_value(grp, customer_id)

    def _resolve_max_value(self, grp: pd.DataFrame, customer_id: str) -> pd.DataFrame:
        """
        Q3-A: highest net_expected_value action wins per channel slot.

        With channel_dedup=True (Q1-C):
          - Sort all eligible rows by net_expected_value descending
          - Assign actions greedily: first row claiming a channel wins
          - Losers get downgraded to NO_ACTION with coordination_note
        """
        grp = grp.sort_values("net_expected_value", ascending=False).copy()

        if not self.config.channel_dedup:
            grp["coordination_note"] = "max_value_no_dedup"
            return grp

        claimed_channels: Dict[str, str] = {}  # channel → winning product
        rows = []

        for _, row in grp.iterrows():
            action  = row["recommended_action"]
            product = row.get("product", "unknown")
            channel = self._action_to_channel(action)

            if action == "NO_ACTION":
                row = row.copy()
                row["coordination_note"] = "no_action"
                rows.append(row)
                continue

            if channel not in claimed_channels:
                # Channel free — this product wins
                claimed_channels[channel] = product
                row = row.copy()
                row["coordination_note"] = f"won_channel_{channel}"

                # Tag cross-product agent call
                if (action == "AGENT_CALL"
                        and self.config.cross_product_agent_call
                        and len(grp["product"].unique()) > 1):
                    other_products = [p for p in grp["product"].unique() if p != product]
                    row["coordination_note"] += f"_covers_{'_'.join(other_products)}"

                rows.append(row)
            else:
                # Channel already claimed — downgrade to NO_ACTION
                winner = claimed_channels[channel]
                row = row.copy()
                original_action = row["recommended_action"]
                row["recommended_action"] = "NO_ACTION"
                row["net_expected_value"]  = 0.0
                row["coordination_note"]   = (
                    f"channel_{channel}_taken_by_{winner}_"
                    f"was_{original_action}"
                )
                row["blocked"]             = True
                row["blocked_reasons"]     = row.get("blocked_reasons", "") + \
                                             f"; channel_conflict:{winner}"
                rows.append(row)

        result = pd.DataFrame(rows)

        logger.debug(
            "customer=%s resolved: %d products, channels_claimed=%s",
            customer_id, len(grp), list(claimed_channels.keys()),
        )
        return result

    def _resolve_consensus(self, grp: pd.DataFrame, customer_id: str) -> pd.DataFrame:
        """
        Consensus: only execute an action if ALL products recommend the same action type.
        Otherwise fall back to NO_ACTION.
        (Future use — user answered Q3-A so this is a configurable alternative)
        """
        actions = grp["recommended_action"].unique()

        if len(actions) == 1:
            grp = grp.copy()
            grp["coordination_note"] = "consensus_unanimous"
            return grp

        # Disagreement → all get NO_ACTION
        grp = grp.copy()
        grp["recommended_action"] = "NO_ACTION"
        grp["net_expected_value"]  = 0.0
        grp["coordination_note"]   = f"consensus_failed_actions_were_{'|'.join(actions)}"
        logger.debug("customer=%s consensus failed: %s", customer_id, actions)
        return grp

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE — HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _action_to_channel(self, action: str) -> str:
        """Map NBA action to channel bucket for dedup purposes."""
        channel_map = {
            "SMS":                      "sms",
            "EMAIL":                    "email",
            "LINE":                     "line",
            "VOICE_IVR":                "voice",
            "AGENT_CALL":               "voice",   # same channel as IVR — human call
            "SETTLEMENT_ONE_TIME":      "offer",
            "SETTLEMENT_PAYMENT_PLAN":  "offer",
            "NO_ACTION":                "none",
        }
        return channel_map.get(action, action.lower())

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {"customer_id", "account_id", "recommended_action",
                    "net_expected_value", "blocked"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"decisions_df missing required columns: {missing}. "
                f"Run DecisionEngine.decide_batch() first and ensure "
                f"customer_id is present (join from account master)."
            )


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY — load from YAML config
# ─────────────────────────────────────────────────────────────────────────────

def build_coordinator_from_config(config: dict) -> CustomerLevelCoordinator:
    """
    Build coordinator from YAML config dict.

    Example YAML:
        coordination:
          mode: customer
          fatigue_scope: product
          action_selection: max_value
          cross_product_agent_call: true
          channel_dedup: true
    """
    coord_cfg = config.get("coordination", {})
    return CustomerLevelCoordinator(CoordinatorConfig(**coord_cfg))
