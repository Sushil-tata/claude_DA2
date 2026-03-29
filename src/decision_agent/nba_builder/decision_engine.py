"""
Decision Engine - Scores and ranks actions for each account.

For each account:
  1. Constraint Engine filters to feasible actions
  2. Scores each feasible action (pay_any_score * amount * uplift)
  3. Applies capacity allocation (limited actions like AGENT_CALL)
  4. Returns ranked action with explanation (Why Card)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


@dataclass
class ActionScore:
    action: str
    pay_any_score: float        # P(payment | action)
    expected_amount: float      # E(amount | payment, action)
    uplift_score: float         # P(pay|action) - P(pay|no_action)
    net_expected_value: float   # uplift * expected_amount - action_cost
    rank: int                   = 0
    explanation: str            = ""


@dataclass
class NBADecision:
    account_id: str
    recommended_action: str
    action_scores: List[ActionScore]
    feasible_actions: List[str]
    blocked_actions: Dict[str, List[str]]
    persona: str
    why_card: Dict[str, Any]


class DecisionEngine:
    """
    Scores and ranks NBA decisions combining model scores + business constraints.
    """

    def __init__(self, config: Dict, constraint_engine=None):
        self.config = config
        self.constraint_engine = constraint_engine
        self.objective = config.get("business_objective", "maximize_net_recovery")
        self.actions = config.get("actions", {})

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def decide(
        self,
        account_context,             # AccountContext from constraint_engine
        model_scores: Dict[str, float],
        persona: str = "unknown",
        current_datetime=None,
    ) -> NBADecision:
        """
        Make NBA decision for a single account.

        Args:
            account_context: AccountContext dataclass
            model_scores: {action: {"pay_any": float, "amount": float, "uplift": float}}
            persona: Assigned persona name
            current_datetime: For quiet hours check

        Returns:
            NBADecision with recommended action and full explanation
        """
        # 1. Get feasible actions
        evals = self.constraint_engine.evaluate_all_actions(account_context, current_datetime)
        feasible = [a for a, e in evals.items() if e.eligible]
        blocked  = {a: e.blocked_reasons for a, e in evals.items() if not e.eligible and a != "NO_ACTION"}

        # Always include NO_ACTION as fallback
        if "NO_ACTION" not in feasible:
            feasible.append("NO_ACTION")

        # 2. Score each feasible action
        scored = []
        for action in feasible:
            score = self._score_action(action, account_context, model_scores, evals.get(action))
            scored.append(score)

        # 3. Rank by objective
        scored = self._rank_actions(scored)

        # 4. Best action
        best = scored[0]

        # 5. Build Why Card
        why_card = self._build_why_card(
            account_context, best, scored, blocked, persona
        )

        return NBADecision(
            account_id=account_context.account_id,
            recommended_action=best.action,
            action_scores=scored,
            feasible_actions=feasible,
            blocked_actions=blocked,
            persona=persona,
            why_card=why_card,
        )

    def decide_batch(
        self,
        accounts_df: pd.DataFrame,
        model_scores_df: pd.DataFrame,
        personas_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Make NBA decisions for a batch of accounts.

        Args:
            accounts_df: Account features (one row per account)
            model_scores_df: Model scores per account per action
            personas_df: Persona assignments

        Returns:
            DataFrame with columns: account_id, recommended_action, score, why_card, ...
        """
        from .constraint_engine import AccountContext

        results = []
        for _, row in accounts_df.iterrows():
            acct_id = row["account_id"]

            # Build AccountContext
            ctx = self._row_to_context(row)

            # Get model scores for this account
            scores = self._extract_scores(acct_id, model_scores_df)

            # Get persona
            persona_rows = personas_df[personas_df["account_id"] == acct_id]
            persona = persona_rows["persona"].iloc[0] if len(persona_rows) > 0 else "unknown"

            decision = self.decide(ctx, scores, persona)
            results.append(self._decision_to_row(decision))

        return pd.DataFrame(results)

    def apply_capacity_constraints(
        self,
        decisions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply capacity limits: demote accounts beyond daily limits.
        Prioritizes by net_expected_value.

        Args:
            decisions_df: Output from decide_batch()

        Returns:
            DataFrame with capacity_status column added
        """
        result = decisions_df.copy()
        result["capacity_status"] = "OK"

        for action, cap_cfg in self.config.get("capacity", {}).items():
            daily_limit = cap_cfg.get("daily_limit", 999999)
            method      = cap_cfg.get("method", "top_expected_value")

            mask = result["recommended_action"] == action
            candidates = result[mask].copy()

            if len(candidates) <= daily_limit:
                continue  # Within capacity

            # Sort by priority
            if method == "top_expected_value":
                candidates = candidates.sort_values("net_expected_value", ascending=False)
            elif method == "worst_performers_high_balance":
                # Highest balance + worst payment history
                candidates = candidates.sort_values(
                    ["balance", "days_past_due"], ascending=[False, False]
                )
            else:
                candidates = candidates.sample(frac=1, random_state=42)

            # Top N get the action, rest get demoted to next best
            within_cap = candidates.iloc[:daily_limit].index
            over_cap   = candidates.iloc[daily_limit:].index

            result.loc[over_cap, "recommended_action"] = result.loc[over_cap, "fallback_action"]
            result.loc[over_cap, "capacity_status"] = f"DEMOTED_{action}_CAPACITY"

        return result

    def explain(self, account_id: str, decisions_df: pd.DataFrame) -> str:
        """Return human-readable explanation for a specific account."""
        rows = decisions_df[decisions_df["account_id"] == account_id]
        if rows.empty:
            return f"Account {account_id} not found in decisions."

        row = rows.iloc[0]
        why = row.get("why_card", {})
        if isinstance(why, str):
            import ast
            try:
                why = ast.literal_eval(why)
            except Exception:
                return why

        lines = [
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Account: {account_id}",
            f"Recommended Action: {row['recommended_action']}",
            f"Persona: {row.get('persona', 'N/A')}",
            f"",
            f"WHY THIS ACTION?",
            f"  Uplift score:        {why.get('uplift_score', 'N/A')}",
            f"  Pay probability:     {why.get('pay_any_score', 'N/A')}",
            f"  Expected recovery:   ${why.get('expected_amount', 0):,.2f}",
            f"  Action cost:         ${why.get('action_cost', 0):,.2f}",
            f"  Net expected value:  ${why.get('net_expected_value', 0):,.2f}",
            f"",
            f"FEASIBLE ACTIONS: {', '.join(why.get('feasible_actions', []))}",
            f"",
            f"BLOCKED ACTIONS:",
        ]

        for action, reasons in why.get("blocked_actions", {}).items():
            lines.append(f"  {action}: {'; '.join(reasons)}")

        lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _score_action(self, action, account, model_scores, eval_result) -> ActionScore:
        """Compute expected value for a single action."""
        action_cfg = self.actions.get(action, {})
        cost = action_cfg.get("cost", 0.0)
        multiplier = eval_result.score_multiplier if eval_result else 1.0

        # Get model predictions for this action
        action_models = model_scores.get(action, model_scores.get("default", {}))
        pay_any    = float(action_models.get("pay_any",   0.5))
        amount     = float(action_models.get("amount",    account.balance * 0.1))
        uplift     = float(action_models.get("uplift",    0.0))
        baseline   = float(model_scores.get("NO_ACTION", {}).get("pay_any", 0.0))

        # Expected recovery = P(pay|action) * E(amount|pay, action)
        expected_recovery = pay_any * amount

        # Net expected value = (uplift * amount) - cost
        net_ev = max(0.0, (uplift * amount) - cost) * multiplier

        explanation = (
            f"P(pay|{action})={pay_any:.2f}, "
            f"uplift={uplift:+.2f} vs baseline={baseline:.2f}, "
            f"E(amount)=${amount:,.0f}, "
            f"net_ev=${net_ev:,.0f}"
        )

        return ActionScore(
            action=action,
            pay_any_score=pay_any,
            expected_amount=amount,
            uplift_score=uplift,
            net_expected_value=net_ev,
            explanation=explanation,
        )

    def _rank_actions(self, scored: List[ActionScore]) -> List[ActionScore]:
        """Rank actions by business objective."""
        if self.objective == "maximize_net_recovery":
            scored.sort(key=lambda x: x.net_expected_value, reverse=True)
        elif self.objective == "maximize_recovery_rate":
            scored.sort(key=lambda x: x.pay_any_score, reverse=True)
        elif self.objective == "minimize_cost_per_collected":
            scored.sort(key=lambda x: (
                self.actions.get(x.action, {}).get("cost", 0) / max(x.expected_amount, 0.01)
            ))
        else:
            scored.sort(key=lambda x: x.net_expected_value, reverse=True)

        for i, s in enumerate(scored):
            s.rank = i + 1
        return scored

    def _build_why_card(self, account, best, scored, blocked, persona) -> Dict:
        top3 = scored[:3]
        return {
            "account_id":          account.account_id,
            "persona":             persona,
            "recommended_action":  best.action,
            "uplift_score":        round(best.uplift_score, 4),
            "pay_any_score":       round(best.pay_any_score, 4),
            "expected_amount":     round(best.expected_amount, 2),
            "action_cost":         self.actions.get(best.action, {}).get("cost", 0.0),
            "net_expected_value":  round(best.net_expected_value, 2),
            "top_3_actions": [
                {
                    "rank":           s.rank,
                    "action":         s.action,
                    "net_ev":         round(s.net_expected_value, 2),
                    "pay_any_score":  round(s.pay_any_score, 4),
                }
                for s in top3
            ],
            "feasible_actions":    [s.action for s in scored],
            "blocked_actions":     blocked,
            "account_state": {
                "balance":       account.balance,
                "days_past_due": account.days_past_due,
                "bucket":        account.bucket,
                "fatigue_score": account.fatigue_score,
            },
        }

    def _row_to_context(self, row: pd.Series):
        """Convert DataFrame row to AccountContext."""
        from .constraint_engine import AccountContext
        return AccountContext(
            account_id=str(row.get("account_id", "")),
            days_past_due=int(row.get("days_past_due", 0)),
            bucket=int(row.get("bucket", 0)),
            balance=float(row.get("balance", 0.0)),
            mobile_present=bool(row.get("mobile_present", False)),
            email_present=bool(row.get("email_present", False)),
            line_optin=bool(row.get("line_optin", False)),
            sms_optin=bool(row.get("sms_optin", False)),
            dnc=bool(row.get("dnc", False)),
            cease_and_desist=bool(row.get("cease_and_desist", False)),
            active_dispute=bool(row.get("active_dispute", False)),
            bankruptcy_flag=bool(row.get("bankruptcy_flag", False)),
            deceased_flag=bool(row.get("deceased_flag", False)),
            fraud_flag=bool(row.get("fraud_flag", False)),
            total_contacts_7d=int(row.get("total_contacts_7d", 0)),
            days_since_last_sms=int(row.get("days_since_last_sms", 99)),
            days_since_last_call=int(row.get("days_since_last_call", 99)),
            days_since_last_email=int(row.get("days_since_last_email", 99)),
            fatigue_score=float(row.get("fatigue_score", 0.0)),
            contact_cost_this_month=float(row.get("contact_cost_this_month", 0.0)),
            active_settlement=bool(row.get("active_settlement", False)),
            active_payment_plan=bool(row.get("active_payment_plan", False)),
            active_ptp=bool(row.get("active_ptp", False)),
            settlement_offered_last_30d=bool(row.get("settlement_offered_last_30d", False)),
        )

    def _extract_scores(self, account_id: str, scores_df: pd.DataFrame) -> Dict:
        """Extract model scores for one account from scores DataFrame."""
        if scores_df is None or scores_df.empty:
            return {}
        rows = scores_df[scores_df["account_id"] == account_id]
        if rows.empty:
            return {}
        result = {}
        for _, r in rows.iterrows():
            action = r.get("action", "default")
            result[action] = {
                "pay_any": float(r.get("pay_any_score", 0.5)),
                "amount":  float(r.get("expected_amount", 0.0)),
                "uplift":  float(r.get("uplift_score", 0.0)),
            }
        return result

    def _decision_to_row(self, d: NBADecision) -> Dict:
        scores = {s.action: s for s in d.action_scores}
        best   = scores.get(d.recommended_action, ActionScore(d.recommended_action, 0, 0, 0, 0))
        fallback = d.action_scores[1].action if len(d.action_scores) > 1 else "NO_ACTION"
        return {
            "account_id":           d.account_id,
            "recommended_action":   d.recommended_action,
            "fallback_action":      fallback,
            "persona":              d.persona,
            "net_expected_value":   best.net_expected_value,
            "pay_any_score":        best.pay_any_score,
            "expected_amount":      best.expected_amount,
            "uplift_score":         best.uplift_score,
            "balance":              d.why_card["account_state"]["balance"],
            "days_past_due":        d.why_card["account_state"]["days_past_due"],
            "feasible_actions":     ",".join(d.feasible_actions),
            "why_card":             str(d.why_card),
            "capacity_status":      "OK",
        }
