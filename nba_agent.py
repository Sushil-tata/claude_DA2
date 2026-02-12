#!/usr/bin/env python3
"""
Interactive NBA Builder Agent
==============================
Run this script to start the interactive NBA system builder.

Usage:
    python nba_agent.py                          # Full interactive setup
    python nba_agent.py --schema my_table.sql    # Start with schema file
    python nba_agent.py --demo                   # Run with synthetic demo data

Once running:
    Type your schema, answer questions, and get a live NBA system.
    Then query it interactively: 'decide account 12345'
    Or explain decisions: 'why account 12345'
    Or update constraints: 'change max calls to 7000'
"""

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# ── Internal imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from decision_agent.nba_builder.schema_parser  import SchemaParser
from decision_agent.nba_builder.questionnaire  import Questionnaire
from decision_agent.nba_builder.constraint_engine import ConstraintEngine, AccountContext
from decision_agent.nba_builder.decision_engine   import DecisionEngine


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def bold(s):   return f"{BOLD}{s}{RESET}"
def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (used for demo / when no real data)
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_accounts(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic debt collection accounts for demo."""
    rng = np.random.default_rng(seed)

    account_ids  = [f"ACC{str(i).zfill(6)}" for i in range(1, n + 1)]
    dpd          = rng.integers(0, 360, size=n)
    balances     = rng.uniform(500, 50000, size=n).round(2)
    mobile       = rng.choice([True, False], size=n, p=[0.85, 0.15])
    email        = rng.choice([True, False], size=n, p=[0.70, 0.30])
    sms_optin    = mobile & rng.choice([True, False], size=n, p=[0.80, 0.20])
    line_optin   = rng.choice([True, False], size=n, p=[0.40, 0.60])
    dnc          = rng.choice([True, False], size=n, p=[0.05, 0.95])
    c_and_d      = rng.choice([True, False], size=n, p=[0.02, 0.98])
    bankruptcy   = rng.choice([True, False], size=n, p=[0.01, 0.99])
    contacts_7d  = rng.integers(0, 5, size=n)
    fatigue      = np.clip(contacts_7d / 5.0 + rng.uniform(0, 0.3, size=n), 0, 1).round(4)
    bucket       = np.clip(dpd // 30, 0, 4).astype(int)
    active_stl   = rng.choice([True, False], size=n, p=[0.05, 0.95])
    ptp          = rng.choice([True, False], size=n, p=[0.08, 0.92])
    cost_month   = rng.uniform(0, 12, size=n).round(2)

    return pd.DataFrame({
        "account_id":              account_ids,
        "days_past_due":           dpd,
        "balance":                 balances,
        "bucket":                  bucket,
        "mobile_present":          mobile,
        "email_present":           email,
        "sms_optin":               sms_optin,
        "line_optin":              line_optin,
        "dnc":                     dnc,
        "cease_and_desist":        c_and_d,
        "bankruptcy_flag":         bankruptcy,
        "deceased_flag":           [False] * n,
        "fraud_flag":              rng.choice([True, False], size=n, p=[0.01, 0.99]).tolist(),
        "total_contacts_7d":       contacts_7d,
        "days_since_last_sms":     rng.integers(0, 30, size=n),
        "days_since_last_call":    rng.integers(0, 30, size=n),
        "days_since_last_email":   rng.integers(0, 30, size=n),
        "fatigue_score":           fatigue,
        "contact_cost_this_month": cost_month,
        "active_settlement":       active_stl,
        "active_payment_plan":     rng.choice([True, False], size=n, p=[0.10, 0.90]).tolist(),
        "active_ptp":              ptp,
        "settlement_offered_last_30d": rng.choice([True, False], size=n, p=[0.15, 0.85]).tolist(),
    })


def generate_mock_model_scores(accounts_df: pd.DataFrame, actions: List[str]) -> pd.DataFrame:
    """Generate synthetic model scores (simulates trained models)."""
    rng = np.random.default_rng(99)
    rows = []
    for _, acct in accounts_df.iterrows():
        acct_id = acct["account_id"]
        dpd     = acct["days_past_due"]
        balance = acct["balance"]
        base_pay = max(0.05, 0.6 - dpd / 500)  # lower pay prob for higher DPD

        for action in actions:
            if action == "NO_ACTION":
                pay_any  = base_pay
                uplift   = 0.0
            elif action in ("SMS", "EMAIL", "LINE"):
                pay_any  = min(0.95, base_pay + rng.uniform(0.05, 0.20))
                uplift   = pay_any - base_pay
            elif action == "VOICE_IVR":
                pay_any  = min(0.95, base_pay + rng.uniform(0.10, 0.25))
                uplift   = pay_any - base_pay
            elif action == "AGENT_CALL":
                pay_any  = min(0.95, base_pay + rng.uniform(0.15, 0.35))
                uplift   = pay_any - base_pay
            elif "SETTLEMENT" in action:
                pay_any  = min(0.95, base_pay + rng.uniform(0.20, 0.40))
                uplift   = pay_any - base_pay
            else:
                pay_any  = base_pay + rng.uniform(0.0, 0.10)
                uplift   = pay_any - base_pay

            expected_amount = balance * pay_any * rng.uniform(0.3, 0.9)
            rows.append({
                "account_id":      acct_id,
                "action":          action,
                "pay_any_score":   round(float(pay_any), 4),
                "expected_amount": round(float(expected_amount), 2),
                "uplift_score":    round(float(uplift), 4),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# NBA AGENT
# ─────────────────────────────────────────────────────────────────────────────
class NBAAgent:
    """
    Interactive NBA Builder Agent.
    Guides setup, then answers queries about decisions.
    """

    def __init__(self):
        self.schema_parser    : Optional[SchemaParser]     = None
        self.questionnaire    : Optional[Questionnaire]    = None
        self.constraint_engine: Optional[ConstraintEngine] = None
        self.decision_engine  : Optional[DecisionEngine]   = None
        self.config           : Dict                       = {}
        self.accounts_df      : Optional[pd.DataFrame]     = None
        self.decisions_df     : Optional[pd.DataFrame]     = None
        self.is_live          : bool                       = False

    # ── SETUP FLOW ────────────────────────────────────────────────────────────

    def run(self, schema_path: Optional[str] = None, demo_mode: bool = False):
        """Main interactive loop."""
        self._print_banner()

        # Phase 1: Schema intake
        schema_input = self._phase_schema(schema_path, demo_mode)

        # Phase 2: Business constraints
        self._phase_constraints(demo_mode)

        # Phase 3: Data & model setup
        self._phase_data(demo_mode)

        # Phase 4: Run decisions
        self._phase_decisions()

        # Phase 5: Interactive query loop
        self._phase_interactive()

    # ── PHASE 1: SCHEMA ───────────────────────────────────────────────────────

    def _phase_schema(self, schema_path: Optional[str], demo_mode: bool) -> str:
        print(f"\n{bold('━━━ PHASE 1: SCHEMA INTAKE ━━━')}")

        if demo_mode:
            print(green("✓ Demo mode: using synthetic debt collection schema"))
            demo_schema = (
                "CREATE TABLE accounts ("
                "account_id STRING, days_past_due INT, balance DECIMAL(10,2), "
                "bucket INT, mobile_present BOOLEAN, email_present BOOLEAN, "
                "sms_optin BOOLEAN, line_optin BOOLEAN, dnc BOOLEAN, "
                "cease_and_desist BOOLEAN, bankruptcy_flag BOOLEAN, "
                "total_contacts_7d INT, fatigue_score DECIMAL(5,4), "
                "active_settlement BOOLEAN, active_payment_plan BOOLEAN"
                ")"
            )
            return self._process_schema(demo_schema, skip_clarification=True)

        if schema_path:
            schema_text = Path(schema_path).read_text()
            print(f"  Loaded schema from: {schema_path}")
            return self._process_schema(schema_text, skip_clarification=False)

        print(f"\n{cyan('Paste your schema below.')} Accepted formats:")
        print(f"  • SQL DDL (CREATE TABLE ...)")
        print(f"  • Column names (comma or newline separated)")
        print(f"  • JSON list: [{{'name':'col','dtype':'type'}}, ...]")
        print(f"\n  Type your schema, then press {bold('Enter twice')} when done.\n")

        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        return self._process_schema("\n".join(lines))

    def _process_schema(self, schema_text: str, skip_clarification: bool = False) -> str:
        self.schema_parser = SchemaParser()
        report = self.schema_parser.parse(schema_text)

        print(f"\n  {green('✓')} Parsed {report['total_columns']} columns")
        print(f"  {green('✓')} Mapped {len(report['mapped'])} / {23} NBA concepts "
              f"({report['coverage_pct']}% coverage)")

        if report["mapped"]:
            print(f"\n  {bold('Mapped columns:')}")
            for concept, col in sorted(report["mapped"].items()):
                print(f"    {green('✓')} {concept:30s} → {col}")

        if report["missing_required"]:
            print(f"\n  {yellow('Missing required concepts:')}")
            for concept in report["missing_required"]:
                print(f"    {yellow('⚠')} {concept}")

        # Ask clarification questions
        questions = self.schema_parser.get_clarification_questions()
        if questions and not skip_clarification:
            print(f"\n{bold('━━━ CLARIFICATION NEEDED ━━━')}")
            for q in questions:
                print(f"\n  {yellow('?')} {q['message']}")
                if q["options"]:
                    print(f"  Options: {', '.join(str(o) for o in q['options'])}")
                answer = input(f"  {bold('Your answer:')} ").strip()
                if answer and answer.lower() not in ("skip", "none", ""):
                    if answer.isdigit() and q.get("options"):
                        idx = int(answer) - 1
                        if 0 <= idx < len(q["options"]):
                            answer = q["options"][idx]
                    if answer.lower() == "derive":
                        self.schema_parser.add_manual_mapping(q["concept"], f"_derived_{q['concept']}")
                    else:
                        self.schema_parser.add_manual_mapping(q["concept"], answer)

        print(f"\n  {green('✓')} Schema intake complete.")
        return schema_text

    # ── PHASE 2: CONSTRAINTS ──────────────────────────────────────────────────

    def _phase_constraints(self, demo_mode: bool):
        print(f"\n{bold('━━━ PHASE 2: BUSINESS CONSTRAINTS ━━━')}")

        self.questionnaire = Questionnaire()

        if demo_mode:
            print(green("✓ Demo mode: using default constraints for debt collection"))
            self.questionnaire.answer("market", "Thailand")
            self.questionnaire.answer("channels_available", ["SMS", "LINE", "VOICE_IVR", "AGENT_CALL"])
            self.questionnaire.answer("offers_available",   ["SETTLEMENT_ONE_TIME", "SETTLEMENT_PAYMENT_PLAN"])
            self.questionnaire.answer("cost_agent_call",    5.00)
            self.questionnaire.answer("cost_sms",           0.25)
            self.questionnaire.answer("agent_call_capacity",5000)
            self.questionnaire.answer("max_contacts_per_week", 3)
            self.questionnaire.answer("quiet_hours_start",  "08:00")
            self.questionnaire.answer("quiet_hours_end",    "21:00")
            self.questionnaire.answer("timezone",           "Asia/Bangkok")
            self.questionnaire.answer("business_objective", "maximize_net_recovery")
            self.questionnaire.answer("settlement_enabled", True)
            self.questionnaire.answer("min_dpd_for_settlement", 60)
            self.questionnaire.accept_defaults()
            self.config = self.questionnaire.build_config()
            print(self.questionnaire.summary())
            return

        print(f"\n  Answer each question. Press {bold('Enter')} to accept the default shown in [brackets].\n")
        print(f"  Type {bold('defaults')} at any point to accept all remaining defaults.\n")

        sections_done = set()
        for q in self.questionnaire.get_pending_questions():
            section = q["section"]
            if section not in sections_done:
                sections_done.add(section)
                print(f"\n  {cyan('── ' + section.upper() + ' ──')}")

            default_display = f" [{q['default']}]" if q.get("default") is not None else ""
            print(f"\n  {yellow('?')} {q['question']}{default_display}")

            if q.get("hint"):
                print(f"    {dim(q['hint'])}")
            if q.get("options") and q["type"] in ("single_select", "multi_select"):
                for i, opt in enumerate(q["options"], 1):
                    print(f"    {i}. {opt}")

            raw = input(f"  {bold('Answer:')} ").strip()

            # 'defaults' shortcut
            if raw.lower() == "defaults":
                self.questionnaire.accept_defaults()
                print(green("  ✓ Using defaults for all remaining questions."))
                break

            if raw == "":
                value = q.get("default")
            elif q["type"] == "multi_select":
                if raw.replace(",", "").replace(" ", "").isdigit():
                    indices = [int(x.strip()) - 1 for x in raw.split(",")]
                    value = [q["options"][i] for i in indices if 0 <= i < len(q["options"])]
                else:
                    value = [v.strip().upper() for v in raw.split(",")]
            elif q["type"] == "single_select":
                if raw.isdigit():
                    idx = int(raw) - 1
                    value = q["options"][idx] if 0 <= idx < len(q["options"]) else raw
                else:
                    value = raw
            elif q["type"] == "yes_no":
                value = raw.lower() in ("y", "yes", "true", "1")
            elif q["type"] == "integer":
                try:
                    value = int(raw)
                except ValueError:
                    value = q.get("default", 0)
            elif q["type"] == "number":
                try:
                    value = float(raw)
                except ValueError:
                    value = q.get("default", 0.0)
            else:
                value = raw if raw else q.get("default")

            followup = self.questionnaire.answer(q["key"], value)
            if value is not None:
                print(f"  {green('✓')} {q['key']} = {value}")
            if followup:
                print(f"  {cyan('ℹ')} {followup}")

        self.config = self.questionnaire.build_config()
        print(f"\n{self.questionnaire.summary()}")
        print(f"\n  {green('✓')} Business constraints configured.")

    # ── PHASE 3: DATA ─────────────────────────────────────────────────────────

    def _phase_data(self, demo_mode: bool):
        print(f"\n{bold('━━━ PHASE 3: DATA LOADING ━━━')}")

        if demo_mode:
            print("  Generating 200 synthetic accounts...")
            self.accounts_df = generate_demo_accounts(200)
            print(f"  {green('✓')} Generated {len(self.accounts_df)} accounts")
            return

        print(f"\n  Options:")
        print(f"  1. Load CSV file (enter path)")
        print(f"  2. Use synthetic demo data")
        print(f"  3. Connect to database (provide connection string)")

        choice = input(f"\n  {bold('Choice [1-3]:')} ").strip()

        if choice == "1":
            path = input("  CSV path: ").strip()
            try:
                self.accounts_df = pd.read_csv(path)
                print(f"  {green('✓')} Loaded {len(self.accounts_df)} accounts from {path}")
            except Exception as e:
                print(f"  {red('✗')} Error loading CSV: {e}")
                print(f"  Falling back to synthetic data...")
                self.accounts_df = generate_demo_accounts(200)
        elif choice == "3":
            print(f"  {yellow('⚠')} Database connection requires additional setup.")
            print(f"  Using synthetic data for now...")
            self.accounts_df = generate_demo_accounts(200)
        else:
            n = input("  How many synthetic accounts to generate? [200]: ").strip()
            n = int(n) if n.isdigit() else 200
            self.accounts_df = generate_demo_accounts(n)
            print(f"  {green('✓')} Generated {n} synthetic accounts")

        # Data quality checks
        self._check_data_quality()

    def _check_data_quality(self):
        """Warn about data quality issues and ask how to handle them."""
        df = self.accounts_df
        issues = []

        missing_mobile_pct = (1 - df.get("mobile_present", pd.Series([True]*len(df))).mean()) * 100
        if missing_mobile_pct > 20:
            issues.append({
                "issue": f"{missing_mobile_pct:.0f}% of accounts missing mobile number",
                "impact": "These accounts cannot receive SMS/Call actions",
                "question": "Should I default mobile_present=True if phone column exists? (y/n)",
                "key": "fix_mobile",
            })

        if "balance" in df.columns:
            zero_balance_pct = (df["balance"] == 0).mean() * 100
            if zero_balance_pct > 5:
                issues.append({
                    "issue": f"{zero_balance_pct:.0f}% of accounts have zero balance",
                    "impact": "Zero-balance accounts should typically not be contacted",
                    "question": "Should I exclude zero-balance accounts from NBA? (y/n)",
                    "key": "exclude_zero_balance",
                })

        if issues:
            print(f"\n  {yellow('━━━ DATA QUALITY ISSUES DETECTED ━━━')}")
            for issue in issues:
                print(f"\n  {yellow('⚠')} {issue['issue']}")
                print(f"    Impact: {issue['impact']}")
                ans = input(f"    {bold(issue['question'])} ").strip().lower()
                if ans in ("y", "yes"):
                    if issue["key"] == "fix_mobile" and "mobile_present" not in df.columns:
                        self.accounts_df["mobile_present"] = True
                        print(f"    {green('✓')} Set mobile_present=True for all accounts")
                    elif issue["key"] == "exclude_zero_balance":
                        before = len(self.accounts_df)
                        self.accounts_df = self.accounts_df[self.accounts_df["balance"] > 0]
                        print(f"    {green('✓')} Removed {before - len(self.accounts_df)} zero-balance accounts")

    # ── PHASE 4: DECISIONS ────────────────────────────────────────────────────

    def _phase_decisions(self):
        print(f"\n{bold('━━━ PHASE 4: RUNNING NBA DECISIONS ━━━')}")

        # Build engines
        self.constraint_engine = ConstraintEngine(self.config)
        self.decision_engine   = DecisionEngine(self.config, self.constraint_engine)

        # Generate model scores (mock or real)
        print("  Generating action scores for all accounts...")
        enabled_actions = [a for a, v in self.config["actions"].items() if v.get("enabled")]
        scores_df = generate_mock_model_scores(self.accounts_df, enabled_actions)

        # Run decisions
        print("  Running constraint checks and scoring...")
        personas_df = pd.DataFrame({
            "account_id": self.accounts_df["account_id"],
            "persona": self._assign_personas(self.accounts_df),
        })

        self.decisions_df = self.decision_engine.decide_batch(
            self.accounts_df, scores_df, personas_df
        )

        # Apply capacity constraints
        print("  Applying capacity constraints...")
        self.decisions_df = self.decision_engine.apply_capacity_constraints(self.decisions_df)

        self.is_live = True

        # Print summary
        self._print_decision_summary()

    def _assign_personas(self, df: pd.DataFrame) -> pd.Series:
        """Simple rule-based persona assignment."""
        def _assign(row):
            dpd = row.get("days_past_due", 0)
            bal = row.get("balance", 0)
            contacts = row.get("total_contacts_7d", 0)
            if bal >= 5000 and dpd < 60 and contacts < 2:
                return "high_value_cooperative"
            elif bal >= 5000 and contacts >= 2:
                return "high_value_unresponsive"
            elif 1000 <= bal < 5000 and dpd < 90:
                return "medium_value_willing"
            elif 1000 <= bal < 5000 and dpd >= 90:
                return "medium_value_struggling"
            elif dpd >= 90:
                return "low_value_chronic"
            else:
                return "standard"
        return df.apply(_assign, axis=1)

    def _print_decision_summary(self):
        df = self.decisions_df
        total = len(df)
        action_counts = df["recommended_action"].value_counts()
        demoted = (df["capacity_status"] != "OK").sum()

        print(f"\n  {green('✓ NBA System is LIVE!')}")
        header = "━━━ TODAY'S DECISIONS ━━━"
        print(f"\n  {bold(header)}")
        print(f"  Total accounts processed: {total:,}")
        print(f"\n  {'Action':<30} {'Count':>8} {'Pct':>8}")
        print(f"  {'-'*46}")
        for action, cnt in action_counts.items():
            pct = cnt / total * 100
            bar = "█" * int(pct / 3)
            print(f"  {action:<30} {cnt:>8,} {pct:>7.1f}%  {bar}")
        if demoted > 0:
            print(f"\n  {yellow(f'⚠ {demoted} accounts demoted due to capacity limits')}")

        top5 = df.nlargest(5, "net_expected_value")[
            ["account_id", "recommended_action", "persona", "net_expected_value", "days_past_due", "balance"]
        ]
        print(f"\n  {bold('TOP 5 HIGHEST VALUE OPPORTUNITIES:')}")
        print(f"  {'Account':<12} {'Action':<22} {'Persona':<28} {'Net EV':>10} {'DPD':>6} {'Balance':>12}")
        print(f"  {'-'*92}")
        for _, row in top5.iterrows():
            print(f"  {row['account_id']:<12} {row['recommended_action']:<22} "
                  f"{row['persona']:<28} ${row['net_expected_value']:>9,.2f} "
                  f"{int(row['days_past_due']):>6} ${row['balance']:>11,.2f}")

        persona_summary = df.groupby("persona")["recommended_action"].value_counts().unstack(fill_value=0)
        print(f"\n  {bold('PERSONA BREAKDOWN:')}")
        print(f"  {persona_summary.to_string()}")

    # ── PHASE 5: INTERACTIVE QUERY LOOP ───────────────────────────────────────

    def _phase_interactive(self):
        print(f"\n{bold('━━━ PHASE 5: INTERACTIVE NBA QUERY ━━━')}")
        print(f"\n  {cyan('Available commands:')}")
        print(f"  {bold('decide <account_id>')}          - Get NBA decision for an account")
        print(f"  {bold('why <account_id>')}             - Explain why an action was recommended")
        print(f"  {bold('change <setting> to <value>')}  - Update a constraint (e.g. change max calls to 7000)")
        print(f"  {bold('search <filter>')}              - Search decisions (e.g. search action=AGENT_CALL)")
        print(f"  {bold('stats')}                        - Show decision statistics")
        print(f"  {bold('export <filename.csv>')}        - Export decisions to CSV")
        print(f"  {bold('quit')}                         - Exit\n")

        while True:
            try:
                cmd = input(f"\n{bold(cyan('NBA>'))} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting NBA Agent. Goodbye!")
                break

            if not cmd:
                continue
            if cmd.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            self._handle_command(cmd)

    def _handle_command(self, cmd: str):
        parts = cmd.strip().split()
        verb  = parts[0].lower() if parts else ""

        # ── decide <account_id> ───────────────────────────────────────────
        if verb == "decide" and len(parts) >= 2:
            acct_id = parts[1].upper()
            rows = self.decisions_df[self.decisions_df["account_id"] == acct_id]
            if rows.empty:
                print(f"  {red('✗')} Account {acct_id} not found.")
                return
            row = rows.iloc[0]
            print(f"\n  {bold('Decision for')} {acct_id}:")
            print(f"  Recommended:   {green(row['recommended_action'])}")
            print(f"  Persona:       {row['persona']}")
            print(f"  Net EV:        ${row['net_expected_value']:,.2f}")
            print(f"  Pay prob:      {row['pay_any_score']:.2%}")
            print(f"  Exp. recovery: ${row['expected_amount']:,.2f}")
            print(f"  Uplift score:  {row['uplift_score']:+.4f}")
            print(f"  Balance:       ${row['balance']:,.2f}")
            print(f"  Days past due: {int(row['days_past_due'])}")
            print(f"  Capacity status: {row['capacity_status']}")

        # ── why <account_id> ─────────────────────────────────────────────
        elif verb == "why" and len(parts) >= 2:
            acct_id = parts[1].upper()
            explanation = self.decision_engine.explain(acct_id, self.decisions_df)
            print(explanation)

        # ── change <setting> to <value> ──────────────────────────────────
        elif verb == "change" and "to" in parts:
            to_idx   = parts.index("to")
            setting  = " ".join(parts[1:to_idx]).lower()
            value_str = " ".join(parts[to_idx + 1:])
            self._apply_setting_change(setting, value_str)

        # ── search <filter> ──────────────────────────────────────────────
        elif verb == "search" and len(parts) >= 2:
            filter_str = " ".join(parts[1:])
            self._search_decisions(filter_str)

        # ── stats ─────────────────────────────────────────────────────────
        elif verb == "stats":
            self._print_decision_summary()

        # ── export <filename> ────────────────────────────────────────────
        elif verb == "export" and len(parts) >= 2:
            filename = parts[1]
            self.decisions_df.to_csv(filename, index=False)
            print(f"  {green('✓')} Decisions exported to {filename}")

        # ── help ─────────────────────────────────────────────────────────
        elif verb in ("help", "?"):
            self._phase_interactive.__doc__

        else:
            print(f"  {yellow('?')} Unknown command: '{cmd}'")
            print(f"  Type {bold('help')} for available commands.")

    def _apply_setting_change(self, setting: str, value_str: str):
        """Update a constraint setting and re-run decisions."""
        changed = False

        if "max calls" in setting or "call capacity" in setting or "agent call" in setting:
            try:
                new_limit = int(value_str.replace(",", ""))
                self.config["capacity"]["AGENT_CALL"]["daily_limit"] = new_limit
                print(f"  {green('✓')} Updated AGENT_CALL daily capacity to {new_limit:,}")
                changed = True
            except ValueError:
                print(f"  {red('✗')} Invalid number: {value_str}")

        elif "max sms" in setting or "sms capacity" in setting:
            try:
                new_limit = int(value_str.replace(",", ""))
                if "SMS" not in self.config["capacity"]:
                    self.config["capacity"]["SMS"] = {}
                self.config["capacity"]["SMS"]["daily_limit"] = new_limit
                print(f"  {green('✓')} Updated SMS daily capacity to {new_limit:,}")
                changed = True
            except ValueError:
                print(f"  {red('✗')} Invalid number: {value_str}")

        elif "max contacts" in setting or "contacts per week" in setting:
            try:
                new_limit = int(value_str)
                self.config["compliance"]["max_contacts_per_week"] = new_limit
                self.config["fatigue"]["max_contacts_per_week"]    = new_limit
                print(f"  {green('✓')} Updated max contacts/week to {new_limit}")
                changed = True
            except ValueError:
                print(f"  {red('✗')} Invalid number: {value_str}")

        elif "objective" in setting:
            valid = ["maximize_net_recovery", "maximize_recovery_rate",
                     "minimize_cost_per_collected", "balance_recovery_and_experience"]
            if value_str.lower().replace(" ", "_") in valid:
                self.config["business_objective"] = value_str.lower().replace(" ", "_")
                print(f"  {green('✓')} Updated business objective to {value_str}")
                changed = True
            else:
                print(f"  {yellow('?')} Valid objectives: {', '.join(valid)}")

        else:
            print(f"  {yellow('?')} Unknown setting: '{setting}'")
            print(f"  Supported: 'max calls', 'max sms', 'max contacts per week', 'objective'")

        if changed:
            print(f"  Re-running decisions with updated constraints...")
            self.constraint_engine = ConstraintEngine(self.config)
            self.decision_engine   = DecisionEngine(self.config, self.constraint_engine)
            self._phase_decisions()

    def _search_decisions(self, filter_str: str):
        """Search decisions with simple filter syntax."""
        df = self.decisions_df.copy()
        try:
            if "=" in filter_str:
                key, val = filter_str.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key in df.columns:
                    try:
                        val = float(val) if "." in val else int(val)
                    except ValueError:
                        pass
                    mask = df[key].astype(str).str.upper() == str(val).upper()
                    results = df[mask]
                else:
                    print(f"  {yellow('?')} Column '{key}' not found.")
                    return
            elif filter_str.startswith("persona"):
                persona_val = filter_str.split()[-1]
                results = df[df["persona"].str.contains(persona_val, case=False)]
            else:
                results = df

            if results.empty:
                print(f"  No accounts match filter: {filter_str}")
                return

            cols = ["account_id", "recommended_action", "persona",
                    "net_expected_value", "days_past_due", "balance"]
            print(f"\n  Found {len(results)} accounts:")
            print(results[cols].head(20).to_string(index=False))
            if len(results) > 20:
                print(f"  ... and {len(results)-20} more")

        except Exception as e:
            print(f"  {red('✗')} Search error: {e}")

    # ── BANNER ────────────────────────────────────────────────────────────────

    def _print_banner(self):
        print(f"""
{bold(green('╔══════════════════════════════════════════════════════════════╗'))}
{bold(green('║       INTERACTIVE NBA BUILDER AGENT                          ║'))}
{bold(green('║       Next Best Action for Debt Collection                   ║'))}
{bold(green('╚══════════════════════════════════════════════════════════════╝'))}
{dim('  Provide your schema → answer a few questions → NBA goes LIVE')}
""")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive NBA Builder Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--schema", type=str, help="Path to schema file (SQL DDL, CSV headers, etc.)")
    parser.add_argument("--demo",   action="store_true", help="Run with synthetic demo data")
    args = parser.parse_args()

    agent = NBAAgent()
    agent.run(schema_path=args.schema, demo_mode=args.demo)


if __name__ == "__main__":
    main()
