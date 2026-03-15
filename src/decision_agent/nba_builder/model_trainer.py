"""
NBA Model Trainer
==================
Trains three model types from labeled data:

  1. pay_any_model   — Binary classifier: P(payment | action, features)
                       Algorithm: GradientBoostingClassifier (S-Learner: action is a feature)

  2. amount_model    — Regression: E(payment_amount | paid=True, action, features)
                       Algorithm: GradientBoostingRegressor with Tweedie loss

  3. uplift_scores   — Computed post-training by scoring each account
                       under all actions and differencing from NO_ACTION baseline

MLflow integration:
  - Logs all metrics, params, feature importances
  - Registers models in MLflow Model Registry
  - Returns model artifacts for use in DecisionEngine

Usage:
    trainer = ModelTrainer(mlflow_tracking_uri="mlruns")
    models  = trainer.train(train_df, val_df, feature_cols, config)
    scores  = trainer.score_all_actions(accounts_df, feature_cols)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder

from .uplift_engine import TLearner, XLearner, UpliftEnsemble, qini_coefficient
from .leakage_detector import LeakageDetector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_auc(y_true, y_score) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


class ModelTrainer:
    """
    Trains pay_any and amount models in S-Learner format.
    Action is encoded as a categorical feature → single model scores all actions.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str = "mlruns",
        experiment_name: str = "nba_models",
    ):
        self.mlflow_uri       = mlflow_tracking_uri
        self.experiment_name  = experiment_name
        self.pay_any_model    = None
        self.amount_model     = None
        self.action_encoder   = LabelEncoder()
        self.feature_cols: List[str] = []
        self._mlflow_available = False
        self._try_setup_mlflow()

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        feature_cols: List[str],
        config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Train both models and return metrics + model artifacts.

        Args:
            train_df:     Training data (features + action + outcome_pay_any + outcome_pay_amount)
            val_df:       Validation data for evaluation
            feature_cols: Feature column names
            config:       Optional hyperparameter overrides

        Returns:
            {
                "pay_any_model": model,
                "amount_model":  model,
                "metrics":       {auc_roc, pr_auc, mae, rmse},
                "feature_importance": {feature: importance, ...}
            }
        """
        self.feature_cols = feature_cols
        cfg = config or {}
        model_type = cfg.get("model_type", "slearner")  # "slearner" | "xlearner" | "ensemble"

        # ── 0. Leakage check before training ─────────────────────────────
        leakage_cfg = cfg.get("leakage_check", {})
        if leakage_cfg.get("enabled", True):
            detector = LeakageDetector(
                correlation_threshold=leakage_cfg.get("correlation_threshold", 0.90),
                adversarial_auc_threshold=leakage_cfg.get("adversarial_auc_threshold", 0.55),
            )
            report = detector.check_all(
                train_df, val_df,
                target_col="outcome_pay_any",
                feature_cols=feature_cols,
            )
            if report["leakage_detected"]:
                msg = (f"Leakage detected! Flagged features: {report['flagged_features']}. "
                       f"Adversarial AUC: {report['checks'].get('adversarial', {}).get('adversarial_auc', 'n/a')}")
                if leakage_cfg.get("block_on_leakage", False):
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)
                    print(f"  ⚠️  {msg}")
            else:
                print("  ✓ Leakage check passed.")

        # ── 1. Encode action as numeric feature ───────────────────────────
        all_actions = pd.concat([train_df["action"], val_df["action"]]).unique()
        self.action_encoder.fit(all_actions)

        X_train = self._build_X(train_df, feature_cols)
        X_val   = self._build_X(val_df,   feature_cols)
        y_pay_train   = train_df["outcome_pay_any"].values
        y_pay_val     = val_df["outcome_pay_any"].values
        y_amt_train   = train_df["outcome_pay_amount"].values
        y_amt_val     = val_df["outcome_pay_amount"].values

        # Treatment vector for uplift models (NO_ACTION = control)
        treatment_train = (train_df["action"] != "NO_ACTION").astype(int).values
        treatment_val   = (val_df["action"]   != "NO_ACTION").astype(int).values

        print(f"  Training pay_any model [{model_type}] ({len(X_train):,} samples)...")

        # ── 2. Train pay_any model ────────────────────────────────────────
        pay_params = {
            "n_estimators":  cfg.get("n_estimators_pay", 150),
            "max_depth":     cfg.get("max_depth_pay",    4),
            "learning_rate": cfg.get("learning_rate",    0.05),
            "subsample":     0.8,
            "random_state":  42,
        }
        pos_weight     = (y_pay_train == 0).sum() / max((y_pay_train == 1).sum(), 1)
        sample_weights = np.where(y_pay_train == 1, pos_weight, 1.0)

        # S-Learner (default): action is encoded as a feature
        self.pay_any_model = GradientBoostingClassifier(**pay_params)
        self.pay_any_model.fit(X_train, y_pay_train, sample_weight=sample_weights)

        pay_val_proba = self.pay_any_model.predict_proba(X_val)[:, 1]
        pay_auc  = _safe_auc(y_pay_val, pay_val_proba)
        pay_pr   = average_precision_score(y_pay_val, pay_val_proba)
        print(f"  pay_any (S-Learner): AUC={pay_auc:.4f}, PR-AUC={pay_pr:.4f}")

        # ── 2b. Uplift model (XLearner or Ensemble) ───────────────────────
        self.uplift_model = None
        uplift_qini       = float("nan")

        if model_type in ("xlearner", "ensemble"):
            # Feature matrix without action encoding (uplift models handle treatment separately)
            X_feat_train = train_df[feature_cols].fillna(0).values.astype(float)
            X_feat_val   = val_df[feature_cols].fillna(0).values.astype(float)

            if model_type == "xlearner":
                self.uplift_model = XLearner(
                    n_estimators=cfg.get("n_estimators_pay", 100),
                    max_depth=cfg.get("max_depth_pay", 4),
                    learning_rate=cfg.get("learning_rate", 0.05),
                )
            else:
                self.uplift_model = UpliftEnsemble(
                    models=[TLearner(n_estimators=100), XLearner(n_estimators=100)]
                )

            try:
                self.uplift_model.fit(X_feat_train, treatment_train, y_pay_train)
                uplift_preds = self.uplift_model.predict_uplift(X_feat_val)
                uplift_qini  = qini_coefficient(uplift_preds, treatment_val, y_pay_val)
                print(f"  uplift ({model_type}): Qini={uplift_qini:.4f}")
            except Exception as e:
                logger.warning("Uplift model training failed: %s — using S-Learner only.", e)
                self.uplift_model = None

        # ── 3. Train amount model (Tweedie-style regression) ──────────────
        print(f"  Training amount model...")
        amt_params = {
            "n_estimators":  cfg.get("n_estimators_amt", 100),
            "max_depth":     cfg.get("max_depth_amt",    4),
            "learning_rate": cfg.get("learning_rate",    0.05),
            "loss":          "squared_error",
            "subsample":     0.8,
            "random_state":  42,
        }
        # Train on all rows but use log(1+y) to handle zero-inflation
        y_amt_log = np.log1p(y_amt_train)
        self.amount_model = GradientBoostingRegressor(**amt_params)
        self.amount_model.fit(X_train, y_amt_log)

        amt_val_log_pred = self.amount_model.predict(X_val)
        amt_val_pred     = np.expm1(amt_val_log_pred).clip(0)
        amt_mae  = mean_absolute_error(y_amt_val, amt_val_pred)
        amt_rmse = mean_squared_error(y_amt_val, amt_val_pred) ** 0.5

        print(f"  amount model: MAE=${amt_mae:.2f}, RMSE=${amt_rmse:.2f}")

        # ── 4. Feature importances ────────────────────────────────────────
        feat_names = feature_cols + ["action_encoded"]
        pay_imp = dict(zip(feat_names, self.pay_any_model.feature_importances_))
        amt_imp = dict(zip(feat_names, self.amount_model.feature_importances_))
        top_pay = sorted(pay_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  Top pay_any features: {[(f, round(v,3)) for f,v in top_pay[:5]]}")

        # ── 5. MLflow logging ─────────────────────────────────────────────
        run_id = self._log_to_mlflow(
            pay_params, amt_params,
            {"pay_auc": pay_auc, "pay_pr_auc": pay_pr, "amt_mae": amt_mae, "amt_rmse": amt_rmse},
            pay_imp, amt_imp,
        )

        return {
            "pay_any_model":      self.pay_any_model,
            "amount_model":       self.amount_model,
            "uplift_model":       self.uplift_model,   # None for slearner
            "model_type":         model_type,
            "action_encoder":     self.action_encoder,
            "feature_cols":       feature_cols,
            "metrics": {
                "pay_auc":      pay_auc,
                "pay_pr_auc":   pay_pr,
                "amt_mae":      amt_mae,
                "amt_rmse":     amt_rmse,
                "uplift_qini":  uplift_qini,
            },
            "feature_importance": {"pay_any": pay_imp, "amount": amt_imp},
            "mlflow_run_id":      run_id,
        }

    def score_all_actions(
        self,
        accounts_df: pd.DataFrame,
        feature_cols: List[str],
        actions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Score every account under EVERY action (S-Learner inference).

        Returns:
            DataFrame with columns:
            account_id, action, pay_any_score, expected_amount, uplift_score
        """
        if self.pay_any_model is None:
            raise RuntimeError("Models not trained yet. Call train() first.")

        if actions is None:
            actions = list(self.action_encoder.classes_)

        rows = []
        # Baseline: NO_ACTION pay probability for uplift computation
        baseline_scores = self._score_single_action(accounts_df, feature_cols, "NO_ACTION")

        for action in actions:
            if action not in self.action_encoder.classes_:
                continue
            action_scores = self._score_single_action(accounts_df, feature_cols, action)
            uplift        = action_scores["pay_any_score"] - baseline_scores["pay_any_score"]

            rows.append(pd.DataFrame({
                "account_id":      accounts_df["account_id"].values,
                "action":          action,
                "pay_any_score":   action_scores["pay_any_score"].round(4),
                "expected_amount": action_scores["expected_amount"].round(2),
                "uplift_score":    uplift.round(4),
            }))

        return pd.concat(rows, ignore_index=True)

    def evaluate_segments(
        self,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        segment_col: str = "bucket",
    ) -> pd.DataFrame:
        """
        Evaluate pay_any model performance broken down by segment.
        """
        if self.pay_any_model is None:
            raise RuntimeError("Models not trained yet.")

        X_test    = self._build_X(test_df, feature_cols)
        y_true    = test_df["outcome_pay_any"].values
        y_scores  = self.pay_any_model.predict_proba(X_test)[:, 1]

        results = []
        for seg_val in sorted(test_df[segment_col].unique()):
            mask = (test_df[segment_col] == seg_val).values
            if mask.sum() < 20:
                continue
            seg_auc = _safe_auc(y_true[mask], y_scores[mask])
            results.append({
                "segment":  seg_val,
                "n":        int(mask.sum()),
                "pay_rate": float(y_true[mask].mean()),
                "auc":      round(seg_auc, 4),
            })
        return pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _build_X(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Build feature matrix with action encoded as numeric column."""
        # Encode action column
        action_enc = self.action_encoder.transform(df["action"].values)

        feature_matrix = df[feature_cols].fillna(0).values.astype(float)
        return np.column_stack([feature_matrix, action_enc])

    def _score_single_action(
        self, df: pd.DataFrame, feature_cols: List[str], action: str
    ) -> pd.DataFrame:
        """Score all accounts assuming they receive a specific action."""
        df_action = df.copy()
        df_action["action"] = action

        try:
            df_action["action"] = self.action_encoder.transform(df_action["action"])
        except ValueError:
            # Action not seen during training — use mean encoding
            df_action["action"] = 0

        X = df[feature_cols].fillna(0).values.astype(float)
        action_enc = df_action["action"].values.reshape(-1, 1)
        X_full = np.column_stack([X, action_enc])

        pay_proba = self.pay_any_model.predict_proba(X_full)[:, 1]
        amt_log   = self.amount_model.predict(X_full)
        amounts   = np.expm1(amt_log).clip(0) * pay_proba  # E[amount] = P(pay) * E[amount|pay]

        return pd.DataFrame({
            "pay_any_score":   pay_proba,
            "expected_amount": amounts,
        })

    def _try_setup_mlflow(self):
        """Try to import and configure MLflow."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow = mlflow
            self._mlflow_available = True
        except ImportError:
            logger.info("MLflow not installed — skipping experiment tracking.")

    def _log_to_mlflow(
        self,
        pay_params: Dict,
        amt_params: Dict,
        metrics:    Dict,
        pay_imp:    Dict,
        amt_imp:    Dict,
    ) -> Optional[str]:
        """Log training run to MLflow."""
        if not self._mlflow_available:
            return None
        try:
            mlflow = self._mlflow
            with mlflow.start_run(run_name="nba_model_training") as run:
                # Params
                mlflow.log_params({f"pay_{k}": v for k, v in pay_params.items()})
                mlflow.log_params({f"amt_{k}": v for k, v in amt_params.items()})

                # Metrics
                mlflow.log_metrics(metrics)

                # Feature importances as JSON artifact
                imp_path = "/tmp/feature_importance.json"
                with open(imp_path, "w") as f:
                    json.dump({"pay_any": pay_imp, "amount": amt_imp}, f, indent=2)
                mlflow.log_artifact(imp_path)

                # Log models
                import mlflow.sklearn
                mlflow.sklearn.log_model(self.pay_any_model, "pay_any_model")
                mlflow.sklearn.log_model(self.amount_model,  "amount_model")

                print(f"  MLflow run logged: {run.info.run_id}")
                return run.info.run_id
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
            return None
