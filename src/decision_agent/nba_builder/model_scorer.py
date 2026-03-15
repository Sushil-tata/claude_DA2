"""
Model Scorer
=============
Loads trained models (from disk or MLflow) and scores accounts.
Acts as the bridge between ModelTrainer and DecisionEngine.

Usage:
    scorer = ModelScorer()
    scorer.load_from_trainer(trainer)        # after training
    scores_df = scorer.score(accounts_df)    # get all action scores
"""

import os
import pickle
import logging
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelScorer:
    """
    Wraps trained models and provides scoring interface for DecisionEngine.
    """

    def __init__(self):
        self.pay_any_model  = None
        self.amount_model   = None
        self.action_encoder = None
        self.feature_cols: List[str] = []
        self.enabled_actions: List[str] = []

    # ──────────────────────────────────────────────────────────────────────────
    # LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def load_from_trainer(self, trainer_artifacts: Dict[str, Any]):
        """
        Load models directly from ModelTrainer.train() output.

        Args:
            trainer_artifacts: return value of ModelTrainer.train()
        """
        self.pay_any_model  = trainer_artifacts["pay_any_model"]
        self.amount_model   = trainer_artifacts["amount_model"]
        self.action_encoder = trainer_artifacts["action_encoder"]
        self.feature_cols   = trainer_artifacts["feature_cols"]
        self.enabled_actions = list(self.action_encoder.classes_)
        logger.info(f"Loaded models. Actions: {self.enabled_actions}")

    def load_from_disk(self, model_dir: str):
        """
        Load models previously saved to disk.
        """
        with open(os.path.join(model_dir, "pay_any_model.pkl"), "rb") as f:
            self.pay_any_model = pickle.load(f)
        with open(os.path.join(model_dir, "amount_model.pkl"), "rb") as f:
            self.amount_model = pickle.load(f)
        with open(os.path.join(model_dir, "action_encoder.pkl"), "rb") as f:
            self.action_encoder = pickle.load(f)

        meta_path = os.path.join(model_dir, "model_meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_cols    = meta.get("feature_cols", [])
            self.enabled_actions = meta.get("enabled_actions", list(self.action_encoder.classes_))

    def save_to_disk(self, model_dir: str):
        """Save models to disk for later loading."""
        import json
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "pay_any_model.pkl"), "wb") as f:
            pickle.dump(self.pay_any_model, f)
        with open(os.path.join(model_dir, "amount_model.pkl"), "wb") as f:
            pickle.dump(self.amount_model, f)
        with open(os.path.join(model_dir, "action_encoder.pkl"), "wb") as f:
            pickle.dump(self.action_encoder, f)
        with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
            json.dump({
                "feature_cols":    self.feature_cols,
                "enabled_actions": self.enabled_actions,
            }, f, indent=2)
        logger.info(f"Models saved to {model_dir}")

    # ──────────────────────────────────────────────────────────────────────────
    # SCORING
    # ──────────────────────────────────────────────────────────────────────────

    def score(
        self,
        accounts_df: pd.DataFrame,
        actions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Score all accounts under all actions.

        Returns:
            DataFrame: account_id, action, pay_any_score, expected_amount, uplift_score
        """
        if self.pay_any_model is None:
            raise RuntimeError("No models loaded. Call load_from_trainer() or load_from_disk() first.")

        target_actions = actions or self.enabled_actions
        rows = []

        # Baseline: NO_ACTION scores for uplift calculation
        baseline = self._score_action_batch(accounts_df, "NO_ACTION")
        base_pay = baseline["pay_any_score"].values

        for action in target_actions:
            if action not in self.action_encoder.classes_:
                logger.warning(f"Action {action} not in encoder — skipping")
                continue

            scored   = self._score_action_batch(accounts_df, action)
            uplift   = scored["pay_any_score"].values - base_pay

            rows.append(pd.DataFrame({
                "account_id":      accounts_df["account_id"].values,
                "action":          action,
                "pay_any_score":   np.round(scored["pay_any_score"].values, 4),
                "expected_amount": np.round(scored["expected_amount"].values, 2),
                "uplift_score":    np.round(uplift, 4),
            }))

        return pd.concat(rows, ignore_index=True)

    def score_single_account(
        self,
        account_id: str,
        account_features: Dict[str, Any],
        actions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Score a single account (real-time use case).
        """
        row_df = pd.DataFrame([account_features])
        row_df["account_id"] = account_id
        return self.score(row_df, actions)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _score_action_batch(self, df: pd.DataFrame, action: str) -> pd.DataFrame:
        """Score all accounts assuming they all receive `action`."""
        action_code = self.action_encoder.transform([action])[0]

        # Build feature matrix
        available_feats = [c for c in self.feature_cols if c in df.columns]
        missing_feats   = [c for c in self.feature_cols if c not in df.columns]

        X = df[available_feats].fillna(0).values.astype(float)

        # Fill missing feature columns with zeros
        if missing_feats:
            zeros = np.zeros((len(df), len(missing_feats)))
            X = np.column_stack([X, zeros])

        # Re-order columns to match training order
        X_full = np.column_stack([X, np.full(len(df), action_code)])

        pay_proba = self.pay_any_model.predict_proba(X_full)[:, 1]
        amt_log   = self.amount_model.predict(X_full)
        amounts   = np.expm1(amt_log).clip(0) * pay_proba

        return pd.DataFrame({
            "pay_any_score":   pay_proba,
            "expected_amount": amounts,
        })
