"""
ClaudeReasoner
==============
Thin wrapper around the Anthropic Claude API for agent judgment calls.

STRICT PII POLICY — enforced in code, not just documentation:
  - Only columns in CLAUDE_ALLOWED_COLUMNS (from contracts/) may be sent
  - Any attempt to pass a PII column raises PIIViolationError BEFORE the API call
  - API key is read from Databricks Secrets — never hardcoded

When is claude_reasoner called?
  Decision Agent calls it ONLY for:
    1. High-value accounts:    erv_at_d_optimal > THB 30,000
    2. D-quadrant + high bal:  low_confidence_flag=True AND balance > THB 20,000
    3. Unexpected low recovery: propensity_180d < 0.15 AND segment_label is NOT null

  All other accounts → rule-based decision (no Claude API call, zero latency cost)
"""

import logging
from typing import Optional

from contracts.model_output_contract import (
    CLAUDE_ALLOWED_COLUMNS,
    PIIViolationError,
    validate_pii,
)

logger = logging.getLogger(__name__)

# ── Thresholds that trigger a Claude call ─────────────────────────────────────
HIGH_VALUE_ERV_THB        = 30_000.0
D_QUADRANT_BALANCE_THB    = 20_000.0
LOW_RECOVERY_THRESHOLD    = 0.15


def _should_call_claude(account: dict) -> tuple[bool, str]:
    """
    Returns (should_call: bool, reason: str).
    Centralised so the threshold logic is testable independently.
    """
    erv     = account.get("erv_at_d_optimal", 0.0) or 0.0
    low_conf = account.get("low_confidence_flag", False)
    p180    = account.get("propensity_180d", 1.0) or 1.0
    seg     = account.get("segment_label")

    if erv > HIGH_VALUE_ERV_THB:
        return True, f"high_value_account (ERV={erv:.0f} THB > {HIGH_VALUE_ERV_THB:.0f})"

    if low_conf and erv > D_QUADRANT_BALANCE_THB:
        return True, f"d_quadrant_high_balance (low_confidence=True, ERV={erv:.0f} THB)"

    if p180 < LOW_RECOVERY_THRESHOLD and seg is not None:
        return True, f"unexpected_low_recovery (p180d={p180:.3f} < {LOW_RECOVERY_THRESHOLD}, segment={seg})"

    return False, "rule_based"


class ClaudeReasoner:
    """
    Calls Claude API for judgment on individual accounts.
    Instantiated once per Decision Agent run.

    In dry_run mode: returns a canned response without making API calls.
    Useful for integration tests and local development.
    """

    SYSTEM_PROMPT = """You are a Principal Data Scientist at a Thai bank (SCB/CardX).
You advise on debt collection decisions for specific accounts.
You receive ONLY anonymised model metadata — no customer names, phone numbers, or PII.
Your role: given model signals, recommend whether to proceed with the model's suggested
action, override it, or refer to a human reviewer.
Be concise. Return JSON only."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        max_tokens: int = 512,
        dry_run: bool = False,
    ):
        self.model      = model
        self.max_tokens = max_tokens
        self.dry_run    = dry_run
        self._client    = None   # lazy-loaded

    def _get_client(self):
        if self._client is None:
            import anthropic
            try:
                # Databricks Secrets (production path)
                from pyspark.dbutils import DBUtils  # noqa
                dbutils = DBUtils(None)
                api_key = dbutils.secrets.get(scope="cardx-secrets", key="anthropic-api-key")
            except Exception:
                # Fallback: environment variable (local dev / CI)
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "ANTHROPIC_API_KEY not set. "
                        "In Databricks, store it in Secrets scope 'cardx-secrets'."
                    )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────────

    def advise(self, account: dict) -> dict:
        """
        Main entry point. Decides whether to call Claude and returns advice.

        Args:
            account: dict containing ONLY allowed columns (enforced here)

        Returns:
            {
              "called_claude": bool,
              "reason_for_call": str,
              "recommendation": str,      # "PROCEED" | "OVERRIDE" | "REFER_HUMAN"
              "confidence": str,          # "HIGH" | "MEDIUM" | "LOW"
              "rationale": str,
              "suggested_d_optimal": float | None,
            }
        """
        # 1. Strip any disallowed columns before doing anything
        safe_account = self._enforce_pii_policy(account)

        # 2. Check if Claude is needed
        should_call, reason = _should_call_claude(safe_account)

        if not should_call:
            return {
                "called_claude":        False,
                "reason_for_call":      reason,
                "recommendation":       "PROCEED",
                "confidence":           "HIGH",
                "rationale":            "Rule-based decision — no Claude call needed.",
                "suggested_d_optimal":  None,
            }

        logger.info(
            f"[ClaudeReasoner] Calling Claude for account={safe_account.get('account_id')} "
            f"| reason={reason}"
        )

        if self.dry_run:
            return self._dry_run_response(safe_account, reason)

        return self._call_claude(safe_account, reason)

    def advise_batch(self, accounts: list[dict]) -> list[dict]:
        """Process a list of accounts. Only calls Claude where threshold is met."""
        return [self.advise(a) for a in accounts]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _enforce_pii_policy(self, account: dict) -> dict:
        """Remove any columns not in CLAUDE_ALLOWED_COLUMNS. Raise if PII found."""
        import pandas as pd
        # Check for PII column names
        validate_pii(pd.DataFrame([account]), context="claude_reasoner.advise")

        # Keep only explicitly allowed columns
        safe = {k: v for k, v in account.items() if k in CLAUDE_ALLOWED_COLUMNS}
        dropped = set(account.keys()) - set(safe.keys())
        if dropped:
            logger.debug(f"[ClaudeReasoner] Dropped non-allowed columns: {dropped}")
        return safe

    def _call_claude(self, account: dict, reason: str) -> dict:
        import json
        client = self._get_client()

        prompt = f"""Account metadata (anonymised):
{json.dumps(account, indent=2, default=str)}

Reason this account was escalated: {reason}

Task: Review the model's recommended action.
Return ONLY valid JSON with these fields:
{{
  "recommendation": "PROCEED" | "OVERRIDE" | "REFER_HUMAN",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "rationale": "one sentence",
  "suggested_d_optimal": <float or null>
}}"""

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            import re
            # Extract JSON even if Claude adds explanation text
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(json_match.group()) if json_match else {}

            return {
                "called_claude":       True,
                "reason_for_call":     reason,
                "recommendation":      parsed.get("recommendation", "PROCEED"),
                "confidence":          parsed.get("confidence", "MEDIUM"),
                "rationale":           parsed.get("rationale", raw[:200]),
                "suggested_d_optimal": parsed.get("suggested_d_optimal"),
            }

        except Exception as e:
            logger.error(f"[ClaudeReasoner] API call failed: {e}. Defaulting to PROCEED.")
            return {
                "called_claude":       True,
                "reason_for_call":     reason,
                "recommendation":      "PROCEED",
                "confidence":          "LOW",
                "rationale":           f"Claude API error — defaulting to model recommendation. Error: {e}",
                "suggested_d_optimal": None,
            }

    def _dry_run_response(self, account: dict, reason: str) -> dict:
        """Deterministic response for testing — no API call."""
        erv = account.get("erv_at_d_optimal", 0.0) or 0.0
        return {
            "called_claude":       True,
            "reason_for_call":     reason,
            "recommendation":      "PROCEED",
            "confidence":          "HIGH",
            "rationale":           (
                f"[DRY RUN] Account {account.get('account_id')} | "
                f"quadrant={account.get('signal_quadrant')} | "
                f"ERV={erv:.0f} THB — model recommendation accepted."
            ),
            "suggested_d_optimal": account.get("d_optimal"),
        }
