"""
Collection Action Aggregator
==============================
Aggregates row-level contact events from T4/T5 into account-level features.

Source tables (post schema_mapper rename):
  T4: efs_cax_extract_actions     — one row per contact attempt
  T5: efs_cax_extract_settlements — one row per PTP record

Output: one row per account with aggregated features ready for:
  - NBA model (willingness, response rate, PTP behaviour)
  - Suppression overlay (voice_calls_today, voice_calls_week)
  - Recovery scorecard (contact intensity, promise quality)
  - Data contract (replaces assumed fields with actual aggregates)

Design:
  - Works on pandas for local/unit tests
  - Column names and result code mappings in config — update when
    actual ACTION_RESULT values are confirmed from EFS system
  - All windows computed relative to a reference date (snapshot_date)
    so output is point-in-time safe

Features produced from T4 (contact actions):
  outbound_calls_made        : total voice call attempts (all time in window)
  calls_connected            : calls where result = connected
  calls_no_answer            : calls where result = no answer
  calls_refused              : calls where result = refused/declined
  sms_sent                   : SMS attempts
  sms_responded              : SMS with positive response
  email_sent                 : email attempts
  line_sent                  : LINE message attempts
  total_contacts_90d         : all channel contacts in last 90 days
  total_contacts_30d         : all channel contacts in last 30 days
  voice_calls_today          : calls on snapshot_date (for BOT suppression)
  voice_calls_week           : calls in last 7 days (for BOT suppression)
  last_contact_date_ordinal  : ordinal of last contact date (model-ready)
  days_since_last_contact    : snapshot_date - last_contact_date
  last_contact_outcome_code  : encoded last result (0=no_answer,1=connected,2=promise,3=refused)
  call_response_rate         : calls_connected / outbound_calls_made
  sms_response_rate          : sms_responded / sms_sent
  contact_intensity_30d      : total_contacts_30d / 30 (contacts per day)
  unique_channels_used       : number of distinct channels tried
  last_agent_id_hash         : hash of last ACTOR (continuity signal — same agent)
  right_party_contact_flag   : 1 if any call was connected in last 30d
  escalation_flag            : 1 if any FIELD_VISIT action in window

Features produced from T5 (PTP / settlements):
  ptp_made                   : total PTPs created
  ptp_kept                   : PTPs with status=KEPT
  ptp_broken                 : PTPs with status=BROKEN
  ptp_partial                : PTPs with status=PARTIAL
  ptp_open                   : PTPs still open (not yet evaluated)
  ptp_kept_rate              : ptp_kept / ptp_made
  ptp_broken_rate            : ptp_broken / ptp_made
  last_ptp_date_ordinal      : ordinal of most recent PTP date
  days_since_last_ptp        : snapshot_date - last_ptp_date
  last_ptp_amount            : most recent PTP amount
  avg_ptp_amount             : mean PTP amount across all PTPs
  ptp_amount_kept_ratio      : sum of kept PTP amounts / sum of all PTP amounts
  ptp_recency_score          : 1 if PTP made in last 30d, 0.5 if 31-90d, 0 otherwise
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionAggregatorConfig:
    """
    Result code mappings and window sizes.
    Update ACTION_* sets when actual EFS result codes are confirmed.
    These are educated guesses based on standard collections systems —
    verify against real data before production use.
    """

    # ── ACTION_CATEGORY values → channel groups ───────────────────────────────
    voice_categories:  Set[str] = field(default_factory=lambda: {
        "CALL", "OUTBOUND_CALL", "VOICE", "PHONE", "AGENT_CALL", "IVR"
    })
    sms_categories:    Set[str] = field(default_factory=lambda: {
        "SMS", "TEXT", "TEXT_MESSAGE"
    })
    email_categories:  Set[str] = field(default_factory=lambda: {
        "EMAIL", "E-MAIL"
    })
    line_categories:   Set[str] = field(default_factory=lambda: {
        "LINE", "LINE_MESSAGE"
    })
    field_categories:  Set[str] = field(default_factory=lambda: {
        "FIELD_VISIT", "FIELD", "IN_PERSON"
    })

    # ── RESULTS values → outcome groups ──────────────────────────────────────
    # TODO: Replace with actual EFS RESULTS code values once confirmed
    connected_results: Set[str] = field(default_factory=lambda: {
        "CONNECTED", "ANSWERED", "RIGHT_PARTY", "RPC", "CONTACT", "SPOKE_TO_CUSTOMER"
    })
    no_answer_results: Set[str] = field(default_factory=lambda: {
        "NO_ANSWER", "NA", "NOT_AVAILABLE", "BUSY", "VOICEMAIL", "VM",
        "LEFT_MESSAGE", "RING_NO_ANSWER", "RNA"
    })
    refused_results:   Set[str] = field(default_factory=lambda: {
        "REFUSED", "DECLINED", "HUNG_UP", "REJECTED", "WRONG_PARTY", "WPC"
    })
    promise_results:   Set[str] = field(default_factory=lambda: {
        "PROMISE", "PTP", "PROMISE_TO_PAY", "COMMITTED"
    })
    sms_responded_results: Set[str] = field(default_factory=lambda: {
        "RESPONDED", "REPLIED", "RESPONSE_RECEIVED"
    })

    # ── PROMISE_STATUS values ─────────────────────────────────────────────────
    kept_statuses:    Set[str] = field(default_factory=lambda: {"KEPT", "PAID", "HONOURED", "FULFILLED"})
    broken_statuses:  Set[str] = field(default_factory=lambda: {"BROKEN", "MISSED", "DEFAULT", "FAILED", "DISHONOURED"})
    partial_statuses: Set[str] = field(default_factory=lambda: {"PARTIAL", "PART_PAID", "PARTIALLY_KEPT"})
    open_statuses:    Set[str] = field(default_factory=lambda: {"OPEN", "PENDING", "ACTIVE"})

    # ── Aggregation windows ───────────────────────────────────────────────────
    window_days_short: int  = 30
    window_days_medium: int = 90
    payment_due_soon_days: int = 7

    # ── Last-outcome encoding (for model) ────────────────────────────────────
    outcome_encoding: Dict[str, int] = field(default_factory=lambda: {
        "no_answer": 0,
        "connected": 1,
        "promise":   2,
        "refused":   3,
    })


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATOR
# ─────────────────────────────────────────────────────────────────────────────

class CollectionActionAggregator:
    """
    Aggregates T4 (actions) and T5 (settlements) to account-level features.

    Usage:
        agg = CollectionActionAggregator()

        # Full batch (recommended for pipeline)
        features_df = agg.aggregate(
            actions_df=t4_df,
            settlements_df=t5_df,
            snapshot_date=date(2026, 2, 9),
        )

        # Single account (useful for on-demand scoring)
        feats = agg.aggregate_one(
            account_id="ACC001",
            actions_df=t4_df[t4_df["action_account_id"] == "ACC001"],
            settlements_df=t5_df[t5_df["ptp_account_id"] == "ACC001"],
            snapshot_date=date(2026, 2, 9),
        )
    """

    def __init__(self, config: Optional[ActionAggregatorConfig] = None):
        self.cfg = config or ActionAggregatorConfig()

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def aggregate(
        self,
        actions_df: pd.DataFrame,
        settlements_df: pd.DataFrame,
        snapshot_date: Optional[date] = None,
        account_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate T4 + T5 to one row per account.

        Args:
            actions_df     : T4 with internal column names (post schema_mapper)
            settlements_df : T5 with internal column names
            snapshot_date  : reference date for windowed features (default=today)
            account_ids    : optional list to restrict output (default=all)

        Returns:
            DataFrame with one row per account_id, all aggregated features.
        """
        snap = snapshot_date or date.today()

        # Ensure date columns are parsed
        actions_df     = self._parse_dates(actions_df,     "action_date")
        settlements_df = self._parse_dates(settlements_df, "ptp_date")

        # Get universe of accounts
        all_ids: Set[str] = set()
        if "action_account_id" in actions_df.columns:
            all_ids |= set(actions_df["action_account_id"].dropna().astype(str))
        if "ptp_account_id" in settlements_df.columns:
            all_ids |= set(settlements_df["ptp_account_id"].dropna().astype(str))

        if account_ids:
            all_ids &= set(str(a) for a in account_ids)

        if not all_ids:
            logger.warning("No accounts found in actions/settlements data")
            return pd.DataFrame()

        records = []
        for aid in sorted(all_ids):
            act_rows  = actions_df[
                actions_df["action_account_id"].astype(str) == aid
            ] if "action_account_id" in actions_df.columns else pd.DataFrame()

            ptp_rows  = settlements_df[
                settlements_df["ptp_account_id"].astype(str) == aid
            ] if "ptp_account_id" in settlements_df.columns else pd.DataFrame()

            feats = self.aggregate_one(aid, act_rows, ptp_rows, snap)
            records.append(feats)

        return pd.DataFrame(records)

    def aggregate_one(
        self,
        account_id: str,
        actions_df: pd.DataFrame,
        settlements_df: pd.DataFrame,
        snapshot_date: date,
    ) -> Dict:
        """Aggregate actions + settlements for a single account."""
        feats: Dict = {"account_id": account_id}

        feats.update(self._aggregate_actions(actions_df, snapshot_date))
        feats.update(self._aggregate_settlements(settlements_df, snapshot_date))

        return feats

    # ── PRIVATE — ACTIONS (T4) ────────────────────────────────────────────────

    def _aggregate_actions(
        self, df: pd.DataFrame, snap: date
    ) -> Dict:
        cfg = self.cfg
        f: Dict = {}

        if df.empty:
            return self._zero_action_features()

        # Normalise category and result to uppercase strings
        categories = df["action_category"].fillna("").str.upper()
        results    = df["action_result"].fillna("").str.upper()
        dates      = pd.to_datetime(df["action_date"], errors="coerce").dt.date

        snap_ts    = pd.Timestamp(snap)

        # ── Channel counts (all time in DataFrame window) ─────────────────────
        is_voice   = categories.isin({c.upper() for c in cfg.voice_categories})
        is_sms     = categories.isin({c.upper() for c in cfg.sms_categories})
        is_email   = categories.isin({c.upper() for c in cfg.email_categories})
        is_line    = categories.isin({c.upper() for c in cfg.line_categories})
        is_field   = categories.isin({c.upper() for c in cfg.field_categories})

        f["outbound_calls_made"] = int(is_voice.sum())
        f["sms_sent"]            = int(is_sms.sum())
        f["email_sent"]          = int(is_email.sum())
        f["line_sent"]           = int(is_line.sum())

        # ── Result counts ─────────────────────────────────────────────────────
        is_connected = results.isin({r.upper() for r in cfg.connected_results})
        is_no_answer = results.isin({r.upper() for r in cfg.no_answer_results})
        is_refused   = results.isin({r.upper() for r in cfg.refused_results})
        is_promise   = results.isin({r.upper() for r in cfg.promise_results})
        is_sms_resp  = results.isin({r.upper() for r in cfg.sms_responded_results})

        f["calls_connected"]  = int((is_voice & is_connected).sum())
        f["calls_no_answer"]  = int((is_voice & is_no_answer).sum())
        f["calls_refused"]    = int((is_voice & is_refused).sum())
        f["sms_responded"]    = int((is_sms  & is_sms_resp).sum())

        # ── Rates ─────────────────────────────────────────────────────────────
        f["call_response_rate"] = round(
            f["calls_connected"] / f["outbound_calls_made"], 4
        ) if f["outbound_calls_made"] > 0 else 0.0

        f["sms_response_rate"] = round(
            f["sms_responded"] / f["sms_sent"], 4
        ) if f["sms_sent"] > 0 else 0.0

        # ── Windowed contact counts ───────────────────────────────────────────
        date_ser = pd.Series(dates.values)

        cutoff_30d  = snap - timedelta(days=cfg.window_days_short)
        cutoff_90d  = snap - timedelta(days=cfg.window_days_medium)
        cutoff_7d   = snap - timedelta(days=7)
        cutoff_today = snap

        in_30d  = date_ser.apply(lambda d: d is not None and d >= cutoff_30d  and d <= snap)
        in_90d  = date_ser.apply(lambda d: d is not None and d >= cutoff_90d  and d <= snap)
        in_7d   = date_ser.apply(lambda d: d is not None and d >= cutoff_7d   and d <= snap)
        on_today = date_ser.apply(lambda d: d is not None and d == cutoff_today)

        f["total_contacts_30d"] = int(in_30d.sum())
        f["total_contacts_90d"] = int(in_90d.sum())
        f["voice_calls_today"]  = int((is_voice & on_today).sum())
        f["voice_calls_week"]   = int((is_voice & in_7d).sum())

        # ── Contact intensity ─────────────────────────────────────────────────
        f["contact_intensity_30d"] = round(f["total_contacts_30d"] / 30, 4)

        # ── Unique channels ───────────────────────────────────────────────────
        channel_flags = {
            "voice": is_voice.any(),
            "sms":   is_sms.any(),
            "email": is_email.any(),
            "line":  is_line.any(),
            "field": is_field.any(),
        }
        f["unique_channels_used"] = sum(channel_flags.values())

        # ── Escalation flag ───────────────────────────────────────────────────
        f["escalation_flag"] = int(is_field.any())

        # ── Right-party contact in last 30d ───────────────────────────────────
        rpc_30d = (is_voice & is_connected & in_30d)
        f["right_party_contact_flag"] = int(rpc_30d.any())

        # ── Last contact ──────────────────────────────────────────────────────
        valid_dates = date_ser[date_ser.notna()]
        if not valid_dates.empty:
            last_date = valid_dates.max()
            f["days_since_last_contact"]   = (snap - last_date).days
            f["last_contact_date_ordinal"] = last_date.toordinal()

            # Last contact outcome encoding
            last_idx    = date_ser[date_ser == last_date].index[-1]
            last_result = results.iloc[last_idx]
            f["last_contact_outcome_code"] = self._encode_outcome(last_result)

            # Last agent (hashed for anonymity)
            if "action_actor" in df.columns:
                last_actor = str(df["action_actor"].iloc[last_idx] or "")
                f["last_agent_id_hash"] = int(
                    hashlib.md5(last_actor.encode()).hexdigest()[:8], 16
                ) % 10000
            else:
                f["last_agent_id_hash"] = 0
        else:
            f["days_since_last_contact"]   = 999
            f["last_contact_date_ordinal"] = 0
            f["last_contact_outcome_code"] = 0
            f["last_agent_id_hash"]        = 0

        return f

    # ── PRIVATE — SETTLEMENTS (T5) ────────────────────────────────────────────

    def _aggregate_settlements(
        self, df: pd.DataFrame, snap: date
    ) -> Dict:
        cfg = self.cfg
        f: Dict = {}

        if df.empty:
            return self._zero_ptp_features()

        statuses   = df["promise_status"].fillna("").str.upper()
        ptp_dates  = pd.to_datetime(df["ptp_date"], errors="coerce").dt.date
        ptp_amts   = pd.to_numeric(df["ptp_amount"], errors="coerce").fillna(0.0)

        # ── Status counts ─────────────────────────────────────────────────────
        is_kept    = statuses.isin({s.upper() for s in cfg.kept_statuses})
        is_broken  = statuses.isin({s.upper() for s in cfg.broken_statuses})
        is_partial = statuses.isin({s.upper() for s in cfg.partial_statuses})
        is_open    = statuses.isin({s.upper() for s in cfg.open_statuses})

        f["ptp_made"]    = len(df)
        f["ptp_kept"]    = int(is_kept.sum())
        f["ptp_broken"]  = int(is_broken.sum())
        f["ptp_partial"] = int(is_partial.sum())
        f["ptp_open"]    = int(is_open.sum())

        # ── Rates ─────────────────────────────────────────────────────────────
        f["ptp_kept_rate"]   = round(f["ptp_kept"]   / f["ptp_made"], 4) if f["ptp_made"] > 0 else 0.0
        f["ptp_broken_rate"] = round(f["ptp_broken"] / f["ptp_made"], 4) if f["ptp_made"] > 0 else 0.0

        # ── Amount stats ──────────────────────────────────────────────────────
        f["avg_ptp_amount"]  = round(float(ptp_amts.mean()), 2) if not ptp_amts.empty else 0.0

        kept_amt  = float(ptp_amts[is_kept].sum())
        total_amt = float(ptp_amts.sum())
        f["ptp_amount_kept_ratio"] = round(
            kept_amt / total_amt, 4
        ) if total_amt > 0 else 0.0

        # ── Last PTP ──────────────────────────────────────────────────────────
        valid_ptp_dates = ptp_dates.dropna()
        if not valid_ptp_dates.empty:
            last_ptp = valid_ptp_dates.max()
            f["days_since_last_ptp"]   = (snap - last_ptp).days
            f["last_ptp_date_ordinal"] = last_ptp.toordinal()

            # Use positional index on reset df to avoid slice index issues
            df_reset = df.reset_index(drop=True)
            ptp_dates_reset = pd.to_datetime(df_reset["ptp_date"], errors="coerce").dt.date
            last_pos = ptp_dates_reset[ptp_dates_reset == last_ptp].index[-1]
            ptp_amts_reset = pd.to_numeric(df_reset["ptp_amount"], errors="coerce").fillna(0.0)
            f["last_ptp_amount"] = round(float(ptp_amts_reset.iloc[last_pos]), 2)
        else:
            f["days_since_last_ptp"]   = 999
            f["last_ptp_date_ordinal"] = 0
            f["last_ptp_amount"]       = 0.0

        # ── PTP recency score ─────────────────────────────────────────────────
        if f["days_since_last_ptp"] <= 30:
            f["ptp_recency_score"] = 1.0
        elif f["days_since_last_ptp"] <= 90:
            f["ptp_recency_score"] = 0.5
        else:
            f["ptp_recency_score"] = 0.0

        return f

    # ── PRIVATE — HELPERS ─────────────────────────────────────────────────────

    def _encode_outcome(self, result_str: str) -> int:
        """Map raw result string to ordinal outcome code."""
        r = result_str.upper()
        cfg = self.cfg
        if r in {x.upper() for x in cfg.promise_results}:
            return cfg.outcome_encoding["promise"]
        if r in {x.upper() for x in cfg.connected_results}:
            return cfg.outcome_encoding["connected"]
        if r in {x.upper() for x in cfg.refused_results}:
            return cfg.outcome_encoding["refused"]
        return cfg.outcome_encoding["no_answer"]

    @staticmethod
    def _parse_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        if date_col in df.columns:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df

    @staticmethod
    def _zero_action_features() -> Dict:
        return {
            "outbound_calls_made":        0,
            "calls_connected":            0,
            "calls_no_answer":            0,
            "calls_refused":              0,
            "sms_sent":                   0,
            "sms_responded":              0,
            "email_sent":                 0,
            "line_sent":                  0,
            "total_contacts_30d":         0,
            "total_contacts_90d":         0,
            "voice_calls_today":          0,
            "voice_calls_week":           0,
            "contact_intensity_30d":      0.0,
            "unique_channels_used":       0,
            "escalation_flag":            0,
            "right_party_contact_flag":   0,
            "call_response_rate":         0.0,
            "sms_response_rate":          0.0,
            "days_since_last_contact":    999,
            "last_contact_date_ordinal":  0,
            "last_contact_outcome_code":  0,
            "last_agent_id_hash":         0,
        }

    @staticmethod
    def _zero_ptp_features() -> Dict:
        return {
            "ptp_made":               0,
            "ptp_kept":               0,
            "ptp_broken":             0,
            "ptp_partial":            0,
            "ptp_open":               0,
            "ptp_kept_rate":          0.0,
            "ptp_broken_rate":        0.0,
            "avg_ptp_amount":         0.0,
            "ptp_amount_kept_ratio":  0.0,
            "days_since_last_ptp":    999,
            "last_ptp_date_ordinal":  0,
            "last_ptp_amount":        0.0,
            "ptp_recency_score":      0.0,
        }

    @staticmethod
    def feature_names() -> List[str]:
        return [
            # Action features
            "outbound_calls_made", "calls_connected", "calls_no_answer",
            "calls_refused", "sms_sent", "sms_responded", "email_sent", "line_sent",
            "total_contacts_30d", "total_contacts_90d",
            "voice_calls_today", "voice_calls_week",
            "contact_intensity_30d", "unique_channels_used",
            "escalation_flag", "right_party_contact_flag",
            "call_response_rate", "sms_response_rate",
            "days_since_last_contact", "last_contact_date_ordinal",
            "last_contact_outcome_code", "last_agent_id_hash",
            # PTP features
            "ptp_made", "ptp_kept", "ptp_broken", "ptp_partial", "ptp_open",
            "ptp_kept_rate", "ptp_broken_rate",
            "avg_ptp_amount", "ptp_amount_kept_ratio",
            "days_since_last_ptp", "last_ptp_date_ordinal",
            "last_ptp_amount", "ptp_recency_score",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    SNAP = date(2026, 2, 9)

    # T4 — collection actions (using internal names post schema_mapper)
    actions = pd.DataFrame([
        # ACC001: frequent contact, one connection, one PTP on call
        {"action_account_id": "ACC001", "action_category": "CALL",  "action_result": "NO_ANSWER",  "action_date": "2026-02-09", "action_actor": "AGT01"},
        {"action_account_id": "ACC001", "action_category": "CALL",  "action_result": "CONNECTED",  "action_date": "2026-02-07", "action_actor": "AGT01"},
        {"action_account_id": "ACC001", "action_category": "CALL",  "action_result": "PROMISE",    "action_date": "2026-02-05", "action_actor": "AGT02"},
        {"action_account_id": "ACC001", "action_category": "SMS",   "action_result": "RESPONDED",  "action_date": "2026-02-03", "action_actor": "SYSTEM"},
        {"action_account_id": "ACC001", "action_category": "SMS",   "action_result": "NO_ANSWER",  "action_date": "2026-01-28", "action_actor": "SYSTEM"},
        {"action_account_id": "ACC001", "action_category": "CALL",  "action_result": "REFUSED",    "action_date": "2026-01-10", "action_actor": "AGT03"},
        # ACC002: multi-channel, field visit, no PTP
        {"action_account_id": "ACC002", "action_category": "CALL",        "action_result": "NO_ANSWER", "action_date": "2026-02-08", "action_actor": "AGT01"},
        {"action_account_id": "ACC002", "action_category": "CALL",        "action_result": "NO_ANSWER", "action_date": "2026-02-06", "action_actor": "AGT01"},
        {"action_account_id": "ACC002", "action_category": "SMS",         "action_result": "NO_ANSWER", "action_date": "2026-02-04", "action_actor": "SYSTEM"},
        {"action_account_id": "ACC002", "action_category": "EMAIL",       "action_result": "NO_ANSWER", "action_date": "2026-01-20", "action_actor": "SYSTEM"},
        {"action_account_id": "ACC002", "action_category": "FIELD_VISIT", "action_result": "CONNECTED", "action_date": "2026-01-15", "action_actor": "FIELD01"},
        # ACC003: no recent contact — stale
        {"action_account_id": "ACC003", "action_category": "CALL", "action_result": "NO_ANSWER", "action_date": "2025-10-01", "action_actor": "AGT01"},
        {"action_account_id": "ACC003", "action_category": "CALL", "action_result": "NO_ANSWER", "action_date": "2025-09-15", "action_actor": "AGT02"},
    ])

    # T5 — PTP / settlements
    settlements = pd.DataFrame([
        # ACC001: 3 PTPs — 2 kept, 1 broken
        {"ptp_account_id": "ACC001", "promise_status": "KEPT",   "ptp_date": "2026-02-05", "ptp_amount": 5000},
        {"ptp_account_id": "ACC001", "promise_status": "KEPT",   "ptp_date": "2026-01-10", "ptp_amount": 5000},
        {"ptp_account_id": "ACC001", "promise_status": "BROKEN", "ptp_date": "2025-12-15", "ptp_amount": 8000},
        # ACC002: 1 open PTP
        {"ptp_account_id": "ACC002", "promise_status": "OPEN",   "ptp_date": "2026-02-04", "ptp_amount": 3000},
        # ACC003: no PTPs
    ])

    agg = CollectionActionAggregator()
    feats_df = agg.aggregate(actions, settlements, snapshot_date=SNAP)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"Snapshot: {SNAP}  |  Total features: {len(agg.feature_names())}")
    print("\n── CONTACT FEATURES ──")
    contact_cols = [
        "account_id",
        "outbound_calls_made", "calls_connected", "calls_refused",
        "sms_sent", "sms_responded", "email_sent",
        "voice_calls_today", "voice_calls_week",
        "total_contacts_30d", "total_contacts_90d",
        "call_response_rate", "sms_response_rate",
        "days_since_last_contact", "last_contact_outcome_code",
        "right_party_contact_flag", "escalation_flag",
        "unique_channels_used", "contact_intensity_30d",
    ]
    print(feats_df[contact_cols].to_string(index=False))

    print("\n── PTP FEATURES ──")
    ptp_cols = [
        "account_id",
        "ptp_made", "ptp_kept", "ptp_broken", "ptp_open",
        "ptp_kept_rate", "ptp_broken_rate",
        "last_ptp_amount", "avg_ptp_amount", "ptp_amount_kept_ratio",
        "days_since_last_ptp", "ptp_recency_score",
    ]
    print(feats_df[ptp_cols].to_string(index=False))

    print("\n── BOT SUPPRESSION INPUTS ──")
    bot_cols = ["account_id", "voice_calls_today", "voice_calls_week"]
    print(feats_df[bot_cols].to_string(index=False))
    print("(Feed voice_calls_today + voice_calls_week into SuppressionOverlay)")
