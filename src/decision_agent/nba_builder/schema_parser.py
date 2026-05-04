"""
Schema Parser - Reads user-provided schema and maps to NBA concepts.

Supports: SQL DDL, CSV headers, JSON, Python dict, Parquet schema.
Asks clarifying questions when column purpose is ambiguous.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# NBA CONCEPT DEFINITIONS
# Each concept has: description, aliases (for auto-detection), required flag
# ─────────────────────────────────────────────────────────────────────────────
NBA_CONCEPTS = {
    # Core identifiers
    "account_id":        {"desc": "Unique account identifier",                 "aliases": ["account_id","acct_id","account_number","acct_no","loan_id","contract_id"], "required": True},
    "customer_id":       {"desc": "Customer identifier (may have multiple accounts)", "aliases": ["customer_id","cust_id","client_id","member_id"],          "required": False},
    "business_date":     {"desc": "Date of the decision snapshot",             "aliases": ["business_date","run_date","as_of_date","snapshot_date","date"],  "required": True},

    # Delinquency state
    "days_past_due":     {"desc": "Days past due (system definition)",         "aliases": ["days_past_due","dpd","days_overdue","overdue_days"],             "required": True},
    "bucket":            {"desc": "Delinquency bucket (0=current, 1=1-30 DPD etc.)", "aliases": ["bucket","bucket_id","delinquency_bucket","dpd_bucket","stage"], "required": False},
    "balance":           {"desc": "Outstanding debt balance",                  "aliases": ["balance","outstanding_balance","current_balance","total_balance","amount_due","principal"], "required": True},
    "due_date":          {"desc": "Payment due date",                          "aliases": ["due_date","payment_due_date","billing_date","bill_date"],         "required": False},

    # Payment history
    "last_payment_date": {"desc": "Date of most recent payment",              "aliases": ["last_payment_date","last_pay_date","last_paid_date"],            "required": False},
    "last_payment_amount":{"desc":"Amount of most recent payment",            "aliases": ["last_payment_amount","last_paid_amount","last_pay_amount"],      "required": False},
    "payment_count_30d": {"desc": "Number of payments in last 30 days",       "aliases": ["payment_count_30d","payments_30d","num_payments_30d"],           "required": False},

    # Contact/channel info
    "mobile_present":    {"desc": "Flag: valid mobile number exists",          "aliases": ["mobile_present","has_mobile","mobile_flag","phone_present","has_phone"], "required": False},
    "email_present":     {"desc": "Flag: valid email exists",                  "aliases": ["email_present","has_email","email_flag","email_valid"],          "required": False},
    "line_optin":        {"desc": "Flag: customer opted in to LINE messages",  "aliases": ["line_optin","line_opt_in","line_consent","line_flag"],           "required": False},
    "sms_optin":         {"desc": "Flag: customer opted in to SMS",            "aliases": ["sms_optin","sms_opt_in","sms_consent","sms_flag"],               "required": False},

    # Compliance
    "dnc":               {"desc": "Do Not Call flag",                          "aliases": ["dnc","do_not_call","dnd","do_not_disturb","no_call"],            "required": False},
    "cease_and_desist":  {"desc": "Cease and desist legal flag",               "aliases": ["cease_and_desist","c_and_d","cnd","legal_hold"],                 "required": False},
    "active_dispute":    {"desc": "Account under active dispute",              "aliases": ["active_dispute","dispute_flag","in_dispute","dispute_status"],   "required": False},
    "bankruptcy_flag":   {"desc": "Customer filed for bankruptcy",             "aliases": ["bankruptcy_flag","bankruptcy","bankrupt","bk_flag"],             "required": False},

    # Contact history
    "total_contacts_7d": {"desc": "Total contact attempts in last 7 days",    "aliases": ["total_contacts_7d","contacts_7d","contact_count_7d","attempts_7d"], "required": False},
    "days_since_last_contact": {"desc": "Days since any contact attempt",     "aliases": ["days_since_last_contact","days_since_contact","last_contact_days"], "required": False},

    # Offer history
    "active_settlement": {"desc": "Flag: active settlement agreement exists",  "aliases": ["active_settlement","settlement_flag","has_settlement","in_settlement"], "required": False},
    "active_payment_plan":{"desc":"Flag: active payment plan exists",          "aliases": ["active_payment_plan","payment_plan_flag","has_plan"],            "required": False},
}

# Concepts that can be DERIVED if not present (shown during clarification)
DERIVABLE_CONCEPTS = {
    "bucket":        "Can be derived from days_past_due (0-30=B1, 31-60=B2, etc.)",
    "fatigue_score": "Can be computed from contact history columns",
    "business_date": "Can default to today's date",
}


class SchemaParser:
    """
    Parses user-provided schema and maps columns to NBA concepts.
    Tracks which concepts are mapped, ambiguous, or missing.
    """

    def __init__(self):
        self.raw_columns: List[Dict] = []          # {name, dtype, sample_values}
        self.mapped: Dict[str, str] = {}            # concept -> column_name
        self.ambiguous: List[Dict] = []             # [{concept, candidates}]
        self.missing_required: List[str] = []       # required concepts not found
        self.missing_optional: List[str] = []       # optional concepts not found
        self.unmapped_columns: List[str] = []       # columns with no concept match

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def parse(self, schema_input: Any, input_format: str = "auto") -> Dict:
        """
        Parse schema from various formats.

        Args:
            schema_input: SQL DDL string, list of column names, dict, or JSON string
            input_format: "sql", "csv", "json", "dict", or "auto"

        Returns:
            Parsed schema dict with mapped/ambiguous/missing info
        """
        fmt = input_format if input_format != "auto" else self._detect_format(schema_input)

        if fmt == "sql":
            self.raw_columns = self._parse_sql_ddl(schema_input)
        elif fmt == "csv":
            self.raw_columns = self._parse_csv_headers(schema_input)
        elif fmt in ("json", "dict"):
            self.raw_columns = self._parse_dict(schema_input)
        else:
            # Fallback: treat as comma/newline separated column names
            self.raw_columns = self._parse_plain_list(schema_input)

        self._auto_map()
        return self.get_mapping_report()

    def get_mapping_report(self) -> Dict:
        """Return full mapping status."""
        return {
            "mapped":            self.mapped,
            "ambiguous":         self.ambiguous,
            "missing_required":  self.missing_required,
            "missing_optional":  self.missing_optional,
            "unmapped_columns":  self.unmapped_columns,
            "total_columns":     len(self.raw_columns),
            "coverage_pct":      round(len(self.mapped) / max(len(NBA_CONCEPTS), 1) * 100, 1),
        }

    def resolve_ambiguity(self, concept: str, chosen_column: str):
        """User resolves an ambiguous mapping."""
        self.mapped[concept] = chosen_column
        self.ambiguous = [a for a in self.ambiguous if a["concept"] != concept]

    def add_manual_mapping(self, concept: str, column: str):
        """User manually maps a concept to a column."""
        if concept in NBA_CONCEPTS:
            self.mapped[concept] = column
            if concept in self.missing_required:
                self.missing_required.remove(concept)
            if concept in self.missing_optional:
                self.missing_optional.remove(concept)

    def get_clarification_questions(self) -> List[Dict]:
        """
        Return list of questions needing user input.
        Each question has: type, concept, message, options (if applicable).
        """
        questions = []

        # 1. Ambiguous mappings
        for amb in self.ambiguous:
            questions.append({
                "type":    "ambiguous_mapping",
                "concept": amb["concept"],
                "message": (
                    f"Multiple columns could be '{amb['concept']}' "
                    f"({NBA_CONCEPTS[amb['concept']]['desc']}):\n"
                    + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(amb["candidates"]))
                    + "\nWhich column maps to this concept? (enter number or column name, or 'none')"
                ),
                "options": amb["candidates"] + ["none"],
            })

        # 2. Missing required concepts
        for concept in self.missing_required:
            derivable_msg = f" ({DERIVABLE_CONCEPTS[concept]})" if concept in DERIVABLE_CONCEPTS else ""
            questions.append({
                "type":    "missing_required",
                "concept": concept,
                "message": (
                    f"Required concept '{concept}' ({NBA_CONCEPTS[concept]['desc']}) "
                    f"not found in schema{derivable_msg}.\n"
                    "Options:\n"
                    "  1. Enter column name that contains this data\n"
                    "  2. Type 'derive' to auto-compute it\n"
                    "  3. Type 'skip' if not applicable"
                ),
                "options": ["<column_name>", "derive", "skip"],
            })

        # 3. Key optional concepts that improve NBA quality
        key_optional = ["mobile_present", "sms_optin", "line_optin", "dnc",
                        "active_settlement", "total_contacts_7d"]
        for concept in key_optional:
            if concept in self.missing_optional:
                questions.append({
                    "type":    "missing_optional",
                    "concept": concept,
                    "message": (
                        f"Optional concept '{concept}' ({NBA_CONCEPTS[concept]['desc']}) "
                        f"not found.\n"
                        "Options:\n"
                        "  1. Enter column name\n"
                        "  2. Type 'derive' to compute it\n"
                        "  3. Type 'skip' to exclude (NBA will work without it)"
                    ),
                    "options": ["<column_name>", "derive", "skip"],
                })

        return questions

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: PARSERS
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_format(self, schema_input: Any) -> str:
        if isinstance(schema_input, dict):
            return "dict"
        if isinstance(schema_input, list):
            return "csv"
        if isinstance(schema_input, str):
            if re.search(r"\bCREATE\s+TABLE\b", schema_input, re.IGNORECASE):
                return "sql"
            try:
                json.loads(schema_input)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass
            return "csv"
        return "csv"

    def _parse_sql_ddl(self, ddl: str) -> List[Dict]:
        """Parse CREATE TABLE SQL DDL into column list."""
        columns = []
        # Extract content between ( and )
        match = re.search(r"CREATE\s+TABLE[^(]+\((.+)\)", ddl, re.IGNORECASE | re.DOTALL)
        if not match:
            return self._parse_plain_list(ddl)

        body = match.group(1)
        # Handle both newline-separated and inline comma-separated column defs
        # Split on comma but be careful of DECIMAL(10,2) — split on comma NOT inside parens
        entries = re.split(r",\s*(?![^()]*\))", body)
        for entry in entries:
            entry = entry.strip()
            if not entry or entry.upper().startswith(("PRIMARY", "FOREIGN", "INDEX",
                                                       "UNIQUE", "CONSTRAINT", "--")):
                continue
            parts = entry.split()
            if len(parts) >= 2:
                col_name = parts[0].strip("`\"[]")
                col_type = parts[1].upper()
                comment = ""
                comment_match = re.search(r"COMMENT\s+'([^']+)'", entry, re.IGNORECASE)
                if comment_match:
                    comment = comment_match.group(1)
                columns.append({"name": col_name, "dtype": col_type, "comment": comment})
        return columns

    def _parse_csv_headers(self, headers: Any) -> List[Dict]:
        """Parse CSV header string or list."""
        if isinstance(headers, list):
            return [{"name": h.strip(), "dtype": "UNKNOWN", "comment": ""} for h in headers]
        # String: could be comma or newline separated
        sep = "," if "," in headers else "\n"
        return [{"name": h.strip(), "dtype": "UNKNOWN", "comment": ""}
                for h in headers.split(sep) if h.strip()]

    def _parse_dict(self, schema_dict: Any) -> List[Dict]:
        """Parse dict or JSON string: {column_name: dtype} or [{name, dtype}]."""
        if isinstance(schema_dict, str):
            schema_dict = json.loads(schema_dict)
        if isinstance(schema_dict, list):
            return [{"name": c.get("name", c.get("column", "")),
                     "dtype": c.get("dtype", c.get("type", "UNKNOWN")),
                     "comment": c.get("comment", c.get("description", ""))}
                    for c in schema_dict]
        return [{"name": k, "dtype": str(v), "comment": ""} for k, v in schema_dict.items()]

    def _parse_plain_list(self, text: str) -> List[Dict]:
        """Parse plain text list of column names."""
        sep = "," if "," in text else "\n"
        return [{"name": h.strip(), "dtype": "UNKNOWN", "comment": ""}
                for h in text.split(sep) if h.strip()]

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE: AUTO-MAPPER
    # ──────────────────────────────────────────────────────────────────────────

    def _auto_map(self):
        """Map raw columns to NBA concepts using alias matching."""
        self.mapped.clear()
        self.ambiguous.clear()
        column_names = [c["name"].lower() for c in self.raw_columns]
        original_names = {c["name"].lower(): c["name"] for c in self.raw_columns}
        used_columns = set()

        for concept, meta in NBA_CONCEPTS.items():
            matches = []
            for alias in meta["aliases"]:
                if alias.lower() in column_names and alias.lower() not in used_columns:
                    matches.append(original_names[alias.lower()])

            if len(matches) == 1:
                self.mapped[concept] = matches[0]
                used_columns.add(matches[0].lower())
            elif len(matches) > 1:
                self.ambiguous.append({"concept": concept, "candidates": matches})
            else:
                if meta["required"]:
                    self.missing_required.append(concept)
                else:
                    self.missing_optional.append(concept)

        # Find completely unmapped columns
        mapped_cols = set(v.lower() for v in self.mapped.values())
        self.unmapped_columns = [c["name"] for c in self.raw_columns
                                  if c["name"].lower() not in mapped_cols]
