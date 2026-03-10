# ============================================================
# NCB Behavioral Feature Factory (PySpark) - PRODUCTION
# THREE-BUCKET + 7 PHYSICS FAMILIES ARCHITECTURE (377 features total)
#
# Runs one DL_DATA_DT at a time (oldest -> newest), writes to OUTPUT_TABLE, then moves to next month.
# Dedup key: (cust_id, as_of_month). Keep the latest load (by dl_data_dt, tie-breaker created_ts).
#
# FEATURE ARCHITECTURE:
#   BUCKET A (Original Engines): 128 features
#     - StateBuilder: 7 features (transition metadata)
#     - TrajectoryEngine: 60 features (velocity, acceleration, entropy)
#     - LenderEcology: 15 features (HHI, Gini, diversity)
#     - EnquiriesEngine: 2 features (diversity, secured share)
#     - CardXBureauInteractions: 15 features (Lead-Lag, Divergence, Spread)
#     - LegalActions: 18 features
#     - TDRRestructuring: 22 features
#
#   BUCKET B (Template): 121 features (+1 from dpd_accel_1m)
#     - Base state features (12)
#     - Bureau aggregations (14)
#     - Lender type counts (8 with Thai classification)
#     - Trajectory features (21: dpd_ols_slope, dpd_accel_1m, shock, deteriorate, improve, stress_frac)
#     - Enquiry features (25)
#     - Repayment dynamics (22)
#     - CardX triggers (10)
#     - Exposure dynamics (15)
#
#   BUCKET C (Advanced Physics): 87 features
#     - Momentum & Inertia (15)
#     - Energy Dynamics (14)
#     - Thermodynamics (7)
#     - Wave Mechanics (4)
#     - Stress Tensor (14)
#     - Chaos & Attractors (2)
#     - Network Topology (15)
#     - Field Theory (7)
#     - Phase Transitions (4)
#     - Relativity (0 - rejected)
#
#   7 PHYSICS FAMILIES (User-Specified): 41 features
#     - Family 1: Inertia & Momentum (7)
#     - Family 2: Critical Slowing (6)
#     - Family 3: Phase Boundary (7)
#     - Family 4: Hysteresis (6)
#     - Family 5: Lender Ecology Topology (5)
#     - Family 6: Enquiry Physics (5)
#     - Family 7: Utilization Physics (5)
#
# CHANGES FROM ORIGINAL TEMPLATE:
#   - CHANGE 1: CHECK_D → RECEIVE_DT in latest_ref_per_cif()
#   - CHANGE 2: Thai lender patterns added to classification
#   - CHANGE 3: Integrated all 3 buckets with deduplication
#
# Optimizations included:
#   - Column pruning on reads (reduce IO)
#   - Broadcast anchor (latest_ref_per_cif) into large tables (reduce shuffle)
#   - Repartition by cust_id before heavy windows (better locality)
#   - Persist panel_base once before long window chains (avoid recompute)
#   - Process DL_DATA_DT sequentially (memory safe; avoids "crash")
# ============================================================

from typing import Dict, List, Tuple, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, StringType
from pyspark.storagelevel import StorageLevel
from delta.tables import DeltaTable

# Import advanced behavioral physics features (Bucket C - 87 features)
from advanced_behavioral_physics import add_advanced_behavioral_physics_features

# Import 7 physics families (41 features)
from physics_families import add_all_physics_families

# Import original behavioral physics engines (Bucket A - 128 features)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from modules.state_builder import StateBuilder
from modules.trajectory_engine import TrajectoryEngine
from modules.lender_ecology import LenderEcologyEngine
from modules.enquiries_engine import EnquiriesEngine
from modules.cardx_bureau_interactions import CardXBureauInteractions
from modules.legal_actions import LegalActionsEngine
from modules.tdr_restructuring import TDRRestructuringEngine

# -------------------------
# OUTPUT / INPUT TABLES
# -------------------------
OUTPUT_TABLE = "cdx_mdz_ana_prd.cdx_ana_riskanalytics_db.recovery_bureau_features_monthly_v2_070326"

ID_DUMMY_TBL  = "cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_id_dummy"
S_ACCOUNT_TBL = "cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account"
S_ENQUIRY_TBL = "cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_enquiry"
S_HISTORY_TBL = "cdx_mdz_prd.cdx_persist_mnf_res_db.mnf_cra_rvw_s_history"

# -------------------------
# CONFIG
# -------------------------
CFG: Dict = {
    "windows_months": [1, 3, 6, 12],
    "dpd_state_bins": [(0, 0, "S0"), (1, 30, "S1"), (31, 90, "S2"), (91, 180, "S3"), (181, 99999, "S4")],
    "stress_states": ["S2", "S3", "S4"],
    "normal_states": ["S0", "S1"],
    "shock_dpd_jump": 60,
    "burst_enquiry_threshold_1m": 3,
    "odm_to_dpd_multiplier": 30,
}

# -------------------------
# THAI LENDER CLASSIFICATION (CHANGE 2)
# -------------------------
def classify_thai_lender(member_id: str) -> str:
    """
    Classify Thai lender by MEMBERSHORTNAME patterns.

    Categories:
    - COMMERCIAL_BANK: ธนาคาร, BANK, BBL, KBANK, SCB, KTB, BAY, TTB, TISCO, CIMB, UOB, LH
    - PERSONAL_LOAN: สินเชื่อ
    - SFI: ออมสิน, GSB, BAAC, GHB, SME
    - FINTECH: TIDLOR, NGERN, EASY, RABBIT
    - CARDX: exact "CARDX"
    - NANO: นาโน, NANO
    - OTHER: unclassified
    """
    if not member_id:
        return "OTHER"

    m = str(member_id).upper().strip()

    # CARDX (exact match)
    if m == "CARDX":
        return "CARDX"

    # COMMERCIAL_BANK
    if "ธนาคาร" in m or "BANK" in m:
        return "COMMERCIAL_BANK"
    if any(code in m for code in ["BBL", "KBANK", "SCB", "KTB", "BAY", "TTB", "TISCO", "CIMB", "UOB", "LH"]):
        return "COMMERCIAL_BANK"

    # PERSONAL_LOAN
    if "สินเชื่อ" in m:
        return "PERSONAL_LOAN"

    # SFI
    if "ออมสิน" in m:
        return "SFI"
    if any(code in m for code in ["GSB", "BAAC", "GHB", "SME"]):
        return "SFI"

    # FINTECH
    if any(code in m for code in ["TIDLOR", "NGERN", "EASY", "RABBIT"]):
        return "FINTECH"

    # NANO
    if "นาโน" in m or "NANO" in m:
        return "NANO"

    return "OTHER"

classify_lender_udf = F.udf(classify_thai_lender, StringType())

# -------------------------
# UTILS
# -------------------------
def _to_month(col):
    return F.to_date(F.date_trunc("month", col))

def _month_end(month_col):
    return F.last_day(month_col)

def _safe_div(num, den, default=F.lit(None)):
    return F.when(den.isNull() | (den == 0), default).otherwise(num / den)

def _as_int_safe_col(col_expr: F.Column) -> F.Column:
    return F.regexp_extract(col_expr.cast("string"), r"(-?\d+)", 1).cast("int")

def _state_from_dpd(dpd_col):
    expr = None
    for lo, hi, lab in CFG["dpd_state_bins"]:
        cond = (dpd_col >= F.lit(lo)) & (dpd_col <= F.lit(hi))
        expr = F.when(cond, F.lit(lab)) if expr is None else expr.when(cond, F.lit(lab))
    return expr.otherwise(F.lit(None))

def _state_num(state_col):
    return (
        F.when(state_col == "S0", 0)
         .when(state_col == "S1", 1)
         .when(state_col == "S2", 2)
         .when(state_col == "S3", 3)
         .when(state_col == "S4", 4)
         .otherwise(F.lit(None))
    )

def del_class(state_col):
    return (
        F.when(state_col == "S0", "Curr/X")
         .when(state_col == "S1", "Curr/X")
         .when(state_col == "S2", "SM")
         .when(state_col == "S3", "NPL")
         .when(state_col == "S4", "CO")
         .otherwise(F.lit(None))
    )

def _entropy_from_state_counts(*counts):
    total = None
    for c in counts:
        total = c if total is None else (total + c)
    terms = []
    for c in counts:
        p = _safe_div(c.cast("double"), total.cast("double"), default=F.lit(0.0))
        terms.append(F.when(p <= 0, F.lit(0.0)).otherwise(-p * F.log(p)))
    ent = None
    for t in terms:
        ent = t if ent is None else (ent + t)
    return ent

# -------------------------
# Latest REF per CIF (CHANGE 1: CHECK_D → RECEIVE_DT)
# -------------------------
def latest_ref_per_cif(id_dummy_df: DataFrame) -> DataFrame:
    """Get latest bureau report per customer using RECEIVE_DT (not CHECK_D)"""
    d = (id_dummy_df
         .withColumn("cust_id", F.col("ID_NO"))
         .withColumn("receive_dt", F.to_date("RECEIVE_DT"))  # CHANGE 1: CHECK_D → RECEIVE_DT
         .withColumn("seq_id", F.col("SEQ_ID").cast("decimal(34,0)"))
         .select("cust_id", "REF_NO", "receive_dt", "seq_id"))
    w = Window.partitionBy("cust_id").orderBy(F.col("receive_dt").desc(), F.col("seq_id").desc_nulls_last())  # CHANGE 1
    return d.withColumn("rn", F.row_number().over(w)).where("rn=1").drop("rn")

# -------------------------
# 3-char split UDF (SAFE)
# -------------------------
def split_string_into_3_chars(input_string):
    if input_string is None:
        return []
    s = str(input_string).replace("\u200b", "").replace(" ", "")
    if len(s) == 0:
        return []
    return [s[i:i+3] for i in range(0, len(s), 3)]

split_string_udf = F.udf(split_string_into_3_chars, ArrayType(StringType()))

# -------------------------
# PAYMENTHISTORY code -> DPD proxy
# -------------------------
def ph_code_to_dpd(col_code: F.Column) -> F.Column:
    c = F.upper(F.trim(col_code))
    return (
        F.when((c.isNull()) | (c == ""), F.lit(None).cast("int"))
         .when(c.isin("Y", "N"), F.lit(None).cast("int"))
         .when(c.endswith("F"), F.lit(300))
         .when(c == "000", F.lit(0))
         .when(c == "001", F.lit(30))
         .when(c == "002", F.lit(60))
         .when(c == "003", F.lit(90))
         .when(c == "004", F.lit(120))
         .when(c == "005", F.lit(150))
         .when(c == "006", F.lit(180))
         .when(c == "007", F.lit(210))
         .when(c == "008", F.lit(240))
         .when(c == "009", F.lit(270))
         .otherwise(F.lit(None).cast("int"))
    )

# -------------------------
# Explode PAYMENTHISTORY into monthly dpd
# -------------------------
def explode_payment_history_monthly(acct_df: DataFrame) -> DataFrame:
    base = (acct_df
        .withColumn(
            "end_month",
            _to_month(F.coalesce(F.to_date("PAYMENTHISTORYENDDATE"), F.to_date("ASOFDATE")))
        )
        .withColumn("ph1_arr", split_string_udf(F.col("PAYMENTHISTORY1")))
        .withColumn("ph2_arr", split_string_udf(F.col("PAYMENTHISTORY2")))
        .withColumn("ph_all", F.concat(F.col("ph1_arr"), F.col("ph2_arr")))
        .withColumn("ph_len", F.size("ph_all"))
        .where(F.col("end_month").isNotNull())
        .where(F.col("ph_len") > 0)
    )

    return (base
        .withColumn("idx", F.explode(F.sequence(F.lit(0), F.col("ph_len") - 1)))
        .withColumn("ph_code", F.element_at(F.col("ph_all"), F.col("idx") + 1))
        .withColumn("as_of_month", F.add_months(F.col("end_month"), -F.col("idx")))
        .withColumn("dpd_from_ph", ph_code_to_dpd(F.col("ph_code")))
        .select("cust_id", "REF_NO", "seq_tl", "as_of_month", "dpd_from_ph")
    )

# -------------------------
# Panel build (optimized)
# -------------------------
def build_monthly_panel_from_ncb(
    id_dummy_df: DataFrame,
    s_account_df: DataFrame,
    s_history_df: DataFrame,
    s_enquiry_df: DataFrame,
    cardx_membercodes: Optional[List[str]] = None,
    use_latest_report_only: bool = True,
) -> Tuple[DataFrame, DataFrame, DataFrame]:

    cardx_codes = [str(x) for x in (cardx_membercodes or [])]

    anchor = latest_ref_per_cif(id_dummy_df) if use_latest_report_only else (
        id_dummy_df.withColumn("cust_id", F.col("ID_NO"))
                   .withColumn("receive_dt", F.to_date("RECEIVE_DT"))  # CHANGE 1: CHECK_D → RECEIVE_DT
                   .select("cust_id", "REF_NO", "receive_dt")
    )

    # broadcast anchor (big win)
    anchor_small = F.broadcast(anchor.select("cust_id", "REF_NO"))

    # ---- Account table
    acc = (s_account_df
        .join(anchor_small, "REF_NO", "inner")
        .withColumn("seq_tl", F.col("SEQ_TL").cast("decimal(24,0)"))
        .withColumn("member_id", F.coalesce(F.col("MEMBERCODE").cast("string"), F.col("MEMBERSHORTNAME").cast("string")))
        .withColumn("is_cardx", F.col("member_id").isin(cardx_codes).cast("int"))
        .withColumn("lender_type", classify_lender_udf(F.col("member_id")))  # CHANGE 2: Thai classification
        .withColumn("defaultdate", F.to_date("DEFAULTDATE"))
        .select(
            "cust_id","REF_NO","seq_tl",
            "member_id","is_cardx","lender_type","defaultdate",
            F.col("ACCOUNTTYPE").alias("acct_type"),
            F.col("ACCOUNTSTATUS").alias("account_status"),
            "PAYMENTHISTORY1","PAYMENTHISTORY2","PAYMENTHISTORYENDDATE","ASOFDATE"
        )
    )

    # ---- History monthly
    hs = (s_history_df
        .join(anchor_small, "REF_NO", "inner")
        .withColumn("seq_tl", F.col("SEQ_TL").cast("decimal(24,0)"))
        .withColumn("as_of_month", _to_month(F.to_date("ASOFDATE")))
        .withColumn("overdue_months_i", _as_int_safe_col(F.col("OVERDUEMONTHS")))
        .withColumn(
            "dpd_from_hs",
            F.when(F.col("overdue_months_i").isNull(), F.lit(None).cast("int"))
             .otherwise(F.greatest(F.lit(0), F.col("overdue_months_i") * F.lit(CFG["odm_to_dpd_multiplier"])))
        )
        .select(
            "cust_id","REF_NO","seq_tl","as_of_month",
            F.coalesce(F.col("CREDITLIMIT").cast("double"), F.lit(0.0)).alias("creditlimit_hs"),
            F.coalesce(F.col("AMOUNTOWED").cast("double"), F.lit(0.0)).alias("amountowed_hs"),
            "dpd_from_hs"
        )
    )

    # ---- Payment history monthly dpd
    ph_monthly = explode_payment_history_monthly(acc)

    # ---- Join per tradeline-month
    hist_joined = (hs
        .join(acc.select("cust_id","REF_NO","seq_tl","member_id","is_cardx","lender_type","acct_type","account_status","defaultdate"),
              ["cust_id","REF_NO","seq_tl"], "left")
        .join(ph_monthly, ["cust_id","REF_NO","seq_tl","as_of_month"], "left")
        .withColumn(
            "default_active_m",
            F.when(
                F.col("defaultdate").isNotNull() & (F.col("defaultdate") <= _month_end(F.col("as_of_month"))),
                F.lit(1)
            ).otherwise(F.lit(0))
        )
        .withColumn("dpd_proxy_final", F.coalesce(F.col("dpd_from_ph"), F.col("dpd_from_hs")))
        .repartition("cust_id")  # reduce shuffle later
    )

    # ---- Monthly customer aggregation
    m = (hist_joined.groupBy("cust_id","as_of_month")
        .agg(
            F.max("dpd_proxy_final").alias("bureau_max_dpd"),
            F.max("default_active_m").alias("bureau_default_active_m"),

            F.sum("amountowed_hs").alias("bureau_owed_sum"),
            F.sum("creditlimit_hs").alias("bureau_limit_sum"),
            F.countDistinct("member_id").alias("bureau_member_cnt"),
            F.count("*").alias("bureau_trade_month_rows"),

            F.sum(F.when(F.col("is_cardx")==1, F.col("amountowed_hs")).otherwise(F.lit(0.0))).alias("bureau_cardx_owed"),
            F.sum(F.when(F.col("is_cardx")==1, F.col("creditlimit_hs")).otherwise(F.lit(0.0))).alias("bureau_cardx_limit"),
            F.max(F.when(F.col("is_cardx")==1, F.col("dpd_proxy_final")).otherwise(F.lit(None))).alias("bureau_cardx_max_dpd"),

            F.sum(F.when(F.col("is_cardx")==0, F.col("amountowed_hs")).otherwise(F.lit(0.0))).alias("bureau_others_owed"),
            F.sum(F.when(F.col("is_cardx")==0, F.col("creditlimit_hs")).otherwise(F.lit(0.0))).alias("bureau_others_limit"),
            F.max(F.when(F.col("is_cardx")==0, F.col("dpd_proxy_final")).otherwise(F.lit(None))).alias("bureau_others_max_dpd"),

            # CHANGE 2: Lender type aggregations
            F.countDistinct(F.when(F.col("lender_type")=="COMMERCIAL_BANK", F.col("member_id"))).alias("cnt_commercial_bank"),
            F.countDistinct(F.when(F.col("lender_type")=="SFI", F.col("member_id"))).alias("cnt_sfi"),
            F.countDistinct(F.when(F.col("lender_type")=="PERSONAL_LOAN", F.col("member_id"))).alias("cnt_personal_loan"),
            F.countDistinct(F.when(F.col("lender_type")=="FINTECH", F.col("member_id"))).alias("cnt_fintech"),
            F.countDistinct(F.when(F.col("lender_type")=="NANO", F.col("member_id"))).alias("cnt_nano"),
        )
    )

    m = (m
        .withColumn("bureau_util", _safe_div(F.col("bureau_owed_sum"), F.col("bureau_limit_sum"), default=F.lit(None)))
        .withColumn("share_owed_cardx", _safe_div(F.col("bureau_cardx_owed"), F.col("bureau_owed_sum"), default=F.lit(0.0)))
        .withColumn("share_limit_cardx", _safe_div(F.col("bureau_cardx_limit"), F.col("bureau_limit_sum"), default=F.lit(0.0)))
        .withColumn("bureau_cardx_max_dpd", F.coalesce(F.col("bureau_cardx_max_dpd"), F.lit(0)))
        .withColumn("bureau_others_max_dpd", F.coalesce(F.col("bureau_others_max_dpd"), F.lit(0)))
        .withColumn("state", _state_from_dpd(F.col("bureau_max_dpd")))
        .withColumn("state_num", _state_num(F.col("state")))
        .withColumn("del_class", del_class(F.col("state")))
        .withColumn("cardx_stressed_m", (F.col("bureau_cardx_max_dpd") >= 31).cast("int"))
        .withColumn("others_stressed_m", (F.col("bureau_others_max_dpd") >= 31).cast("int"))
    )

    # ---- Enquiry monthly
    enq = (s_enquiry_df
        .join(anchor_small, "REF_NO", "inner")
        .withColumn("as_of_month", _to_month(F.to_date("DATEOFENQUIRY")))
        .withColumn("member_id", F.coalesce(F.col("MEMBERCODE").cast("string"), F.col("MEMBERSHORTNAME").cast("string")))
        .withColumn("is_cardx_enq", F.col("member_id").isin(cardx_codes).cast("int"))
        .withColumn("enq_amt", F.coalesce(F.col("ENQUIRYAMOUNT").cast("double"), F.lit(0.0)))
    )

    enq_m = (enq.groupBy("cust_id","as_of_month")
        .agg(
            F.count("*").alias("enq_cnt_m"),
            F.sum("enq_amt").alias("enq_amt_sum_m"),
            F.sum(F.when(F.col("is_cardx_enq")==1, 1).otherwise(0)).alias("enq_cardx_cnt_m"),
            F.sum(F.when(F.col("is_cardx_enq")==0, 1).otherwise(0)).alias("enq_others_cnt_m"),
            F.countDistinct("member_id").alias("enq_member_cnt_m"),
        )
    )

    panel_base = (m.join(enq_m, ["cust_id","as_of_month"], "left")
        .withColumn("enq_cnt_m", F.coalesce(F.col("enq_cnt_m"), F.lit(0)))
        .withColumn("enq_amt_sum_m", F.coalesce(F.col("enq_amt_sum_m"), F.lit(0.0)))
        .withColumn("enq_cardx_cnt_m", F.coalesce(F.col("enq_cardx_cnt_m"), F.lit(0)))
        .withColumn("enq_others_cnt_m", F.coalesce(F.col("enq_others_cnt_m"), F.lit(0)))
        .withColumn("enq_burst_1m_flag", (F.col("enq_cnt_m") >= F.lit(CFG["burst_enquiry_threshold_1m"])).cast("int"))
        .repartition("cust_id")
    )

    return panel_base, hist_joined, enq_m

# -------------------------
# Trajectory features
# -------------------------
def add_trajectory_features(panel: DataFrame) -> DataFrame:
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    panel = (panel
        .withColumn("lag_state", F.lag("state").over(w))
        .withColumn("lag_state_num", F.lag("state_num").over(w))
        .withColumn("lag_dpd", F.lag("bureau_max_dpd").over(w))
        .withColumn("dpd_jump", (F.col("bureau_max_dpd") - F.col("lag_dpd")).cast("int"))
        .withColumn("shock_flag_m", (F.abs(F.col("dpd_jump")) >= F.lit(CFG["shock_dpd_jump"])).cast("int"))
        .withColumn("deteriorate_flag_m", (F.col("state_num") > F.col("lag_state_num")).cast("int"))
        .withColumn("improve_flag_m", (F.col("state_num") < F.col("lag_state_num")).cast("int"))
        .withColumn("state_changed_m", (F.col("state") != F.col("lag_state")).cast("int"))
    )

    panel = panel.withColumn("streak_id", F.sum(F.when(F.col("state_changed_m") == 1, 1).otherwise(0)).over(w))
    w_streak = Window.partitionBy("cust_id", "streak_id")
    panel = panel.withColumn("streak_len", F.count("*").over(w_streak))

    panel = panel.withColumn("is_stressed_m", F.col("state").isin(CFG["stress_states"]).cast("int"))
    panel = panel.withColumn("is_normal_m", F.col("state").isin(CFG["normal_states"]).cast("int"))

    for m in CFG["windows_months"]:
        ww = w.rowsBetween(-m + 1, 0)

        panel = (panel
            .withColumn(f"flip_cnt_{m}m", F.sum("state_changed_m").over(ww))
            .withColumn(f"shock_cnt_{m}m", F.sum("shock_flag_m").over(ww))
            .withColumn(f"deteriorate_cnt_{m}m", F.sum("deteriorate_flag_m").over(ww))
            .withColumn(f"improve_cnt_{m}m", F.sum("improve_flag_m").over(ww))
            .withColumn(f"dpd_std_{m}m", F.stddev_pop("bureau_max_dpd").over(ww))
            .withColumn(f"dpd_max_jump_{m}m", F.max(F.abs(F.col("dpd_jump"))).over(ww))
        )

        for s in ["S0","S1","S2","S3","S4"]:
            panel = panel.withColumn(f"cnt_{s.lower()}_{m}m", F.sum(F.when(F.col("state")==s, 1).otherwise(0)).over(ww))

        panel = panel.withColumn(
            f"state_entropy_{m}m",
            _entropy_from_state_counts(
                F.col(f"cnt_s0_{m}m"), F.col(f"cnt_s1_{m}m"), F.col(f"cnt_s2_{m}m"), F.col(f"cnt_s3_{m}m"), F.col(f"cnt_s4_{m}m")
            )
        )

        # slope approximation
        t_sum = (m - 1) * m / 2.0
        t2_sum = (m - 1) * m * (2*m - 1) / 6.0
        var_t = t2_sum - (t_sum * t_sum) / m

        panel = panel.withColumn(f"y_sum_{m}m", F.sum(F.col("bureau_max_dpd").cast("double")).over(ww))

        ty = None
        for k in range(m):
            term = F.lit(float(k)) * F.lag(F.col("bureau_max_dpd").cast("double"), k).over(w)
            ty = term if ty is None else (ty + term)
        panel = panel.withColumn(f"ty_sum_{m}m", ty)

        panel = panel.withColumn(
            f"dpd_ols_slope_{m}m",
            F.when(F.lit(m) >= 2,
                   (F.col(f"ty_sum_{m}m") - (F.lit(t_sum) * F.col(f"y_sum_{m}m") / F.lit(float(m)))) / F.lit(float(var_t))
            ).otherwise(F.lit(None))
        )

        secdiff = (F.col("bureau_max_dpd").cast("double")
                   - 2*F.lag(F.col("bureau_max_dpd").cast("double"), 1).over(w)
                   + F.lag(F.col("bureau_max_dpd").cast("double"), 2).over(w))
        panel = panel.withColumn(f"dpd_accel_mean_{m}m", F.avg(secdiff).over(ww))
        panel = panel.withColumn(f"stress_frac_{m}m", F.avg("is_stressed_m").over(ww))

    # Single-month acceleration (not averaged) for jerk computation
    secdiff_1m = (F.col("bureau_max_dpd").cast("double")
                  - 2*F.lag(F.col("bureau_max_dpd").cast("double"), 1).over(w)
                  + F.lag(F.col("bureau_max_dpd").cast("double"), 2).over(w))
    panel = panel.withColumn("dpd_accel_1m", secdiff_1m)

    w12 = Window.partitionBy("cust_id").orderBy("as_of_month").rowsBetween(-11, 0)
    panel = (panel
        .withColumn("streak_len_curr", F.col("streak_len"))
        .withColumn("max_streak_12m", F.max("streak_len").over(w12))
        .withColumn("stress_streak_len", F.when(F.col("is_stressed_m")==1, F.col("streak_len")).otherwise(F.lit(0)))
        .withColumn("max_stress_streak_12m", F.max("stress_streak_len").over(w12))
        .withColumn("stress_frac_12m", F.avg("is_stressed_m").over(w12))
        .withColumn("phase_clean_to_stress_m",
                    ((F.col("lag_state").isin(CFG["normal_states"])) & (F.col("state").isin(CFG["stress_states"]))).cast("int"))
        .withColumn("phase_stress_to_default_m",
                    ((F.col("lag_state").isin(["S2","S3"])) & (F.col("state")=="S4")).cast("int"))
        .withColumn("phase_default_to_cure_m",
                    ((F.col("lag_state")=="S4") & (F.col("state").isin(["S2","S1","S0"]))).cast("int"))
    )

    return panel

# -------------------------
# Enquiry rolling
# -------------------------
def add_enquiry_roll_features(panel: DataFrame) -> DataFrame:
    w = Window.partitionBy("cust_id").orderBy("as_of_month")
    for m in CFG["windows_months"]:
        ww = w.rowsBetween(-m + 1, 0)
        panel = (panel
            .withColumn(f"enq_cnt_{m}m", F.sum("enq_cnt_m").over(ww))
            .withColumn(f"enq_amt_sum_{m}m", F.sum("enq_amt_sum_m").over(ww))
            .withColumn(f"enq_cardx_cnt_{m}m", F.sum("enq_cardx_cnt_m").over(ww))
            .withColumn(f"enq_others_cnt_{m}m", F.sum("enq_others_cnt_m").over(ww))
            .withColumn(f"enq_cardx_share_{m}m",
                        _safe_div(F.col(f"enq_cardx_cnt_{m}m"), F.col(f"enq_cnt_{m}m"), default=F.lit(0.0)))
            .withColumn(f"enq_cnt_{m}m_lag{m}m", F.lag(F.col(f"enq_cnt_{m}m"), m).over(w))
            .withColumn(f"enq_accel_{m}m", F.col(f"enq_cnt_{m}m") - F.col(f"enq_cnt_{m}m_lag{m}m"))
        )
    return panel

# -------------------------
# Repayment dynamics proxy
# -------------------------
def add_repayment_dynamics_from_ncb(panel: DataFrame) -> DataFrame:
    w = Window.partitionBy("cust_id").orderBy("as_of_month")
    panel = (panel
        .withColumn("owed_lag1", F.lag("bureau_owed_sum", 1).over(w))
        .withColumn("proxy_delever_amt_m", F.greatest(F.lit(0.0), F.col("owed_lag1") - F.col("bureau_owed_sum")))
        .withColumn("proxy_delever_rate_m", _safe_div(F.col("proxy_delever_amt_m"), F.col("owed_lag1"), default=F.lit(None)))
        .withColumn("proxy_delever_flag_m", (F.col("proxy_delever_amt_m") > 0).cast("int"))
        .withColumn("is_stress_m", F.col("state").isin(CFG["stress_states"]).cast("int"))
    )
    panel = panel.withColumn("proxy_delever_rate_lag1", F.lag("proxy_delever_rate_m", 1).over(w))
    panel = panel.withColumn("proxy_delever_rate_drop_m", (F.col("proxy_delever_rate_lag1") - F.col("proxy_delever_rate_m")))

    for m in CFG["windows_months"]:
        ww = w.rowsBetween(-m + 1, 0)

        panel = (panel
            .withColumn(
                f"norm_delever_hit_{m}m",
                _safe_div(
                    F.sum(F.when(F.col("is_stress_m")==0, F.col("proxy_delever_flag_m")).otherwise(0)).over(ww),
                    F.sum(F.when(F.col("is_stress_m")==0, 1).otherwise(0)).over(ww),
                    default=F.lit(None)
                )
            )
            .withColumn(
                f"norm_delever_rate_avg_{m}m",
                F.avg(F.when(F.col("is_stress_m")==0, F.col("proxy_delever_rate_m")).otherwise(F.lit(None))).over(ww)
            )
            .withColumn(
                f"norm_delever_vol_{m}m",
                F.stddev_pop(F.when(F.col("is_stress_m")==0, F.col("proxy_delever_rate_m")).otherwise(F.lit(None))).over(ww)
            )
            .withColumn(
                f"stress_delever_hit_{m}m",
                _safe_div(
                    F.sum(F.when(F.col("is_stress_m")==1, F.col("proxy_delever_flag_m")).otherwise(0)).over(ww),
                    F.sum(F.when(F.col("is_stress_m")==1, 1).otherwise(0)).over(ww),
                    default=F.lit(None)
                )
            )
            .withColumn(
                f"stress_delever_rate_avg_{m}m",
                F.avg(F.when(F.col("is_stress_m")==1, F.col("proxy_delever_rate_m")).otherwise(F.lit(None))).over(ww)
            )
            .withColumn(
                f"stress_fatigue_drop_avg_{m}m",
                F.avg(F.when(F.col("is_stress_m")==1, F.col("proxy_delever_rate_drop_m")).otherwise(F.lit(None))).over(ww)
            )
            .withColumn(f"delta_delever_hit_{m}m", F.col(f"norm_delever_hit_{m}m") - F.col(f"stress_delever_hit_{m}m"))
            .withColumn(f"delta_delever_rate_{m}m", F.col(f"norm_delever_rate_avg_{m}m") - F.col(f"stress_delever_rate_avg_{m}m"))
        )

    return panel

# -------------------------
# CardX vs Others triggers
# -------------------------
def add_cardx_vs_others_triggers(panel: DataFrame) -> DataFrame:
    w = Window.partitionBy("cust_id").orderBy("as_of_month")
    panel = (panel
        .withColumn("cardx_stressed_lag1", F.lag("cardx_stressed_m", 1).over(w))
        .withColumn("others_stressed_lag1", F.lag("others_stressed_m", 1).over(w))
        .withColumn("others_then_cardx_trigger_m",
                    ((F.col("others_stressed_lag1")==1) & (F.col("cardx_stressed_m")==1) & (F.col("cardx_stressed_lag1")==0)).cast("int"))
        .withColumn("cardx_then_others_trigger_m",
                    ((F.col("cardx_stressed_lag1")==1) & (F.col("others_stressed_m")==1) & (F.col("others_stressed_lag1")==0)).cast("int"))
    )

    for m in CFG["windows_months"]:
        ww = w.rowsBetween(-m + 1, 0)
        panel = (panel
            .withColumn(f"others_then_cardx_trigger_cnt_{m}m", F.sum("others_then_cardx_trigger_m").over(ww))
            .withColumn(f"cardx_then_others_trigger_cnt_{m}m", F.sum("cardx_then_others_trigger_m").over(ww))
        )
    return panel

# -------------------------
# Audit (optional; keep light)
# -------------------------
def build_audit_log(panel: DataFrame) -> DataFrame:
    dup = panel.groupBy("cust_id", "as_of_month").count().where(F.col("count") > 1)
    dup_cnt = dup.count()

    cols_check = ["bureau_max_dpd","state","dpd_ols_slope_6m","state_entropy_12m","stress_frac_12m"]
    miss_exprs = [(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)) / F.count(F.lit(1))).alias(f"nullrate__{c}") for c in cols_check]
    miss_df = panel.agg(*miss_exprs)

    audit = (panel.agg(
            F.count("*").alias("rows"),
            F.countDistinct("cust_id").alias("customers"),
            F.countDistinct("as_of_month").alias("months"),
        )
        .withColumn("dup_cust_month_cnt", F.lit(int(dup_cnt)))
    )
    return audit.crossJoin(miss_df)

# -------------------------
# Main builder
# -------------------------
def build_ncb_behavioral_feature_store(
    id_dummy_df: DataFrame,
    s_account_df: DataFrame,
    s_history_df: DataFrame,
    s_enquiry_df: DataFrame,
    cardx_membercodes: Optional[List[str]] = None,
    use_latest_report_only: bool = True,
    run_audit: bool = False,
) -> Tuple[DataFrame, DataFrame, Optional[DataFrame]]:

    # Get SparkSession for initializing engines
    spark = id_dummy_df.sql_ctx.sparkSession

    panel_base, hist_joined_df, enq_m = build_monthly_panel_from_ncb(
        id_dummy_df=id_dummy_df,
        s_account_df=s_account_df,
        s_history_df=s_history_df,
        s_enquiry_df=s_enquiry_df,
        cardx_membercodes=cardx_membercodes,
        use_latest_report_only=use_latest_report_only,
    )

    # persist before many window chains (big win)
    panel_base = panel_base.persist(StorageLevel.MEMORY_AND_DISK)
    panel_base.count()

    # ========================================================================
    # BUCKET B: TEMPLATE FEATURES (120 features after dedup)
    # ========================================================================
    panel = add_trajectory_features(panel_base)
    panel = add_enquiry_roll_features(panel)
    panel = add_repayment_dynamics_from_ncb(panel)
    panel = add_cardx_vs_others_triggers(panel)

    # exposure dynamics extras
    w = Window.partitionBy("cust_id").orderBy("as_of_month")
    panel = (panel
        .withColumn("owed_jump_m", (F.col("bureau_owed_sum") - F.lag("bureau_owed_sum", 1).over(w)))
        .withColumn("limit_jump_m", (F.col("bureau_limit_sum") - F.lag("bureau_limit_sum", 1).over(w)))
        .withColumn("util_lag1", F.lag("bureau_util", 1).over(w))
        .withColumn("util_jump_m", (F.col("bureau_util") - F.col("util_lag1")))
    )
    for m in CFG["windows_months"]:
        ww = w.rowsBetween(-m + 1, 0)
        panel = (panel
            .withColumn(f"owed_growth_{m}m",
                        _safe_div(F.col("bureau_owed_sum") - F.first("bureau_owed_sum").over(ww),
                                  F.first("bureau_owed_sum").over(ww), default=F.lit(None)))
            .withColumn(f"util_avg_{m}m", F.avg("bureau_util").over(ww))
            .withColumn(f"util_vol_{m}m", F.stddev_pop("bureau_util").over(ww))
        )

    # ========================================================================
    # BUCKET A: ORIGINAL ENGINE FEATURES (128 features after dedup)
    # Adds unique features not present in template
    # ========================================================================
    print("\n" + "="*80)
    print("INTEGRATING BUCKET A: ORIGINAL BEHAVIORAL PHYSICS ENGINES")
    print("="*80)

    # Initialize engines
    state_builder = StateBuilder(spark)
    trajectory_engine = TrajectoryEngine(spark)
    lender_ecology = LenderEcologyEngine(spark)
    enquiries_engine = EnquiriesEngine(spark)
    cardx_interactions = CardXBureauInteractions(spark)
    legal_actions = LegalActionsEngine(spark)
    tdr_restructuring = TDRRestructuringEngine(spark)

    # 1. StateBuilder - 7 unique features (state transition metadata)
    print("\n1/7: StateBuilder (7 features)...")
    # StateBuilder needs hist_joined_df converted to trade format
    # Most state features already in template, only get transition metadata
    state_df = state_builder.compute_state_transitions(panel)
    state_unique_cols = ["prev_month_state", "prev_month_regime", "current_state_streak",
                        "max_s3_streak", "max_s4_streak", "state_change_indicator", "state_group"]
    for col in state_unique_cols:
        if col in state_df.columns and col not in panel.columns:
            panel = panel.join(
                state_df.select("cust_id", "as_of_month", col),
                on=["cust_id", "as_of_month"],
                how="left"
            )
    print("✓ StateBuilder features integrated")

    # 2. TrajectoryEngine - 60 features (velocity, acceleration, transitions, entropy)
    print("\n2/7: TrajectoryEngine (60 features)...")
    # Reconstruct bureau_trade_df from hist_joined for trajectory engine
    bureau_trade_for_traj = hist_joined_df.select(
        "cust_id", "as_of_month", "dpd_proxy_final", "creditlimit_hs", "amountowed_hs"
    ).withColumnRenamed("dpd_proxy_final", "dpd") \
     .withColumnRenamed("creditlimit_hs", "credit_limit") \
     .withColumnRenamed("amountowed_hs", "balance")

    traj_df = trajectory_engine.compute_all_features(panel, bureau_trade_for_traj)
    # Join trajectory features (exclude duplicates already in template)
    traj_unique_cols = [c for c in traj_df.columns if c not in panel.columns
                       and c not in ["cust_id", "as_of_month"]]
    if traj_unique_cols:
        panel = panel.join(
            traj_df.select("cust_id", "as_of_month", *traj_unique_cols),
            on=["cust_id", "as_of_month"],
            how="left"
        )
    print(f"✓ TrajectoryEngine features integrated ({len(traj_unique_cols)} unique)")

    # 3. LenderEcology - 15 specific features
    print("\n3/7: LenderEcology (15 features)...")
    lender_df = lender_ecology.compute_all_features(bureau_trade_for_traj, panel)
    # Only keep the 15 user-specified features
    lender_keep_cols = [
        "hhi_lender_concentration", "gini_lender_concentration",
        "shannon_entropy_lender_tiers", "network_degree_m",
        "tier_downgrade_flag", "nano_entry_ever_flag", "nano_entry_3m_flag",
        "new_lender_velocity_3m", "new_lender_velocity_6m",
        "lender_age_mean", "lender_age_diversity",
        "formal_to_informal_ratio", "top1_lender_balance_share",
        "top3_lender_balance_share", "lender_tier_mix_index"
    ]
    lender_available = [c for c in lender_keep_cols if c in lender_df.columns]
    if lender_available:
        panel = panel.join(
            lender_df.select("cust_id", "as_of_month", *lender_available),
            on=["cust_id", "as_of_month"],
            how="left"
        )
    print(f"✓ LenderEcology features integrated ({len(lender_available)} of 15)")

    # 4. EnquiriesEngine - 2 unique features (diversity, secured share)
    print("\n4/7: EnquiriesEngine (2 features)...")
    # Reconstruct enquiry_df from enq_m
    enq_df = enquiries_engine.compute_all_features(enq_m, bureau_trade_for_traj)
    enq_unique_cols = ["enquiry_type_diversity", "secured_enquiry_share"]
    enq_available = [c for c in enq_unique_cols if c in enq_df.columns]
    if enq_available:
        panel = panel.join(
            enq_df.select("cust_id", "as_of_month", *enq_available),
            on=["cust_id", "as_of_month"],
            how="left"
        )
    print(f"✓ EnquiriesEngine features integrated ({len(enq_available)} of 2)")

    # 5. CardXBureauInteractions - 15 unique features (Lead-Lag, Divergence, Util Spread)
    print("\n5/7: CardXBureauInteractions (15 features)...")
    # Needs bureau + cardx data - extract from panel
    cardx_df = cardx_interactions.compute_all_features(bureau_trade_for_traj, panel, panel)
    # Exclude trigger features (duplicates), keep Lead-Lag, Divergence, Util Spread
    cardx_exclude = ["others_then_cardx_trigger", "cardx_then_others_trigger"]
    cardx_unique_cols = [c for c in cardx_df.columns if c not in panel.columns
                        and c not in cardx_exclude and c not in ["cust_id", "as_of_month"]]
    if cardx_unique_cols:
        panel = panel.join(
            cardx_df.select("cust_id", "as_of_month", *cardx_unique_cols),
            on=["cust_id", "as_of_month"],
            how="left"
        )
    print(f"✓ CardXBureauInteractions features integrated ({len(cardx_unique_cols)} unique)")

    # 6. LegalActions - 18 features
    print("\n6/7: LegalActions (18 features)...")
    legal_df = legal_actions.compute_all_features(bureau_trade_for_traj)
    legal_cols = [c for c in legal_df.columns if c not in ["cust_id", "as_of_month"]]
    if legal_cols:
        panel = panel.join(
            legal_df.select("cust_id", "as_of_month", *legal_cols),
            on=["cust_id", "as_of_month"],
            how="left"
        )
    print(f"✓ LegalActions features integrated ({len(legal_cols)} features)")

    # 7. TDRRestructuring - 22 features
    print("\n7/7: TDRRestructuring (22 features)...")
    tdr_df = tdr_restructuring.compute_all_features(bureau_trade_for_traj)
    tdr_cols = [c for c in tdr_df.columns if c not in ["cust_id", "as_of_month"]]
    if tdr_cols:
        panel = panel.join(
            tdr_df.select("cust_id", "as_of_month", *tdr_cols),
            on=["cust_id", "as_of_month"],
            how="left"
        )
    print(f"✓ TDRRestructuring features integrated ({len(tdr_cols)} features)")

    print("\n" + "="*80)
    print("✅ BUCKET A INTEGRATION COMPLETE")
    print("="*80 + "\n")

    # ========================================================================
    # BUCKET C: ADVANCED BEHAVIORAL PHYSICS FEATURES (87 features after review)
    # Momentum, energy dynamics, thermodynamics, stress tensor, network topology,
    # field theory, phase transitions - state-of-the-art physics features
    # ========================================================================
    panel = add_advanced_behavioral_physics_features(panel)

    # ========================================================================
    # 7 PHYSICS FAMILIES (41 features - user-specified)
    # Additions to the 335 existing features (Buckets A+B+C)
    # ========================================================================
    panel = add_all_physics_families(panel, hist_joined_df)

    key_cols = ["cust_id", "as_of_month"]
    monthly_feature_df = panel.select(*key_cols, *[c for c in panel.columns if c not in key_cols]).repartition("cust_id")

    audit_log_df = build_audit_log(monthly_feature_df) if run_audit else None

    # free cache
    panel_base.unpersist()

    return monthly_feature_df, hist_joined_df, audit_log_df

# -------------------------
# Incremental Delta writer (MERGE)
# Keep latest by dl_data_dt then created_ts
# -------------------------
def _ensure_output_table_exists(sample_df: DataFrame):
    if not spark.catalog.tableExists(OUTPUT_TABLE):
        (sample_df
         .write.format("delta")
         .mode("overwrite")
         .partitionBy("as_of_month")
         .saveAsTable(OUTPUT_TABLE))

def _merge_into_output(src_df: DataFrame):
    dt = DeltaTable.forName(spark, OUTPUT_TABLE)

    # pre-dedup inside batch: keep latest dl_data_dt, then created_ts
    w = Window.partitionBy("cust_id", "as_of_month").orderBy(F.col("dl_data_dt").desc(), F.col("created_ts").desc())
    src = (src_df
           .withColumn("_rn", F.row_number().over(w))
           .where(F.col("_rn") == 1)
           .drop("_rn"))

    (dt.alias("t")
      .merge(src.alias("s"),
             "t.cust_id = s.cust_id AND t.as_of_month = s.as_of_month")
      .whenMatchedUpdateAll(condition="(s.dl_data_dt > t.dl_data_dt) OR (s.dl_data_dt = t.dl_data_dt AND s.created_ts >= t.created_ts)")
      .whenNotMatchedInsertAll()
      .execute())

# -------------------------
# Loader: one DL_DATA_DT
# (column pruning to reduce IO)
# -------------------------
def _load_one_snapshot(dl_data_dt: str):
    f = (F.col("DL_DATA_DT") == F.lit(dl_data_dt))

    id_dummy_df = (spark.table(ID_DUMMY_TBL)
                   .where(f)
                   .select("ID_NO", "REF_NO", "RECEIVE_DT", "SEQ_ID", "DL_DATA_DT"))  # CHANGE 1: CHECK_D → RECEIVE_DT

    s_account_df = (spark.table(S_ACCOUNT_TBL)
                    .where(f)
                    .select(
                        "REF_NO","SEQ_TL","MEMBERCODE","MEMBERSHORTNAME",
                        "DEFAULTDATE","ACCOUNTTYPE","ACCOUNTSTATUS",
                        "PAYMENTHISTORY1","PAYMENTHISTORY2","PAYMENTHISTORYENDDATE","ASOFDATE",
                        "DL_DATA_DT"
                    ))

    s_history_df = (spark.table(S_HISTORY_TBL)
                    .where(f)
                    .select(
                        "REF_NO","SEQ_TL","ASOFDATE",
                        "OVERDUEMONTHS","CREDITLIMIT","AMOUNTOWED",
                        "DL_DATA_DT"
                    ))

    s_enquiry_df = (spark.table(S_ENQUIRY_TBL)
                    .where(f)
                    .select(
                        "REF_NO","DATEOFENQUIRY",
                        "MEMBERCODE","MEMBERSHORTNAME","ENQUIRYAMOUNT",
                        "DL_DATA_DT"
                    ))

    return id_dummy_df, s_account_df, s_history_df, s_enquiry_df

# -------------------------
# Driver: process a list of DL_DATA_DT sequentially
# -------------------------
def run_ncb_incremental(
    dl_data_dt_start: str,
    dl_data_dt_end: str,
    cardx_membercodes: Optional[List[str]] = None,
    use_latest_report_only: bool = True,
    run_audit: bool = False,
):
    cardx_membercodes = cardx_membercodes or ["CARDX"]

    # list DL_DATA_DT to run (from id_dummy driver)
    dl_rows = (spark.table(ID_DUMMY_TBL)
               .select(F.to_date("DL_DATA_DT").alias("DL_DATA_DT"))
               .where((F.col("DL_DATA_DT") >= F.lit(dl_data_dt_start)) &
                      (F.col("DL_DATA_DT") <= F.lit(dl_data_dt_end)))
               .distinct()
               .orderBy("DL_DATA_DT")
               .collect())

    dl_values = [r["DL_DATA_DT"].strftime("%Y-%m-%d") for r in dl_rows]
    if not dl_values:
        print(f"No DL_DATA_DT found between {dl_data_dt_start} and {dl_data_dt_end}")
        return

    print("DL_DATA_DT to process:", dl_values)

    # process one month at a time
    for dl_data_dt in dl_values:
        print(f"\n=== Processing DL_DATA_DT = {dl_data_dt} ===")

        id_dummy_df, s_account_df, s_history_df, s_enquiry_df = _load_one_snapshot(dl_data_dt)

        monthly_feature_df, _, audit_df = build_ncb_behavioral_feature_store(
            id_dummy_df=id_dummy_df,
            s_account_df=s_account_df,
            s_history_df=s_history_df,
            s_enquiry_df=s_enquiry_df,
            cardx_membercodes=cardx_membercodes,
            use_latest_report_only=use_latest_report_only,
            run_audit=run_audit
        )

        out_df = (monthly_feature_df
                  .withColumn("dl_data_dt", F.lit(dl_data_dt).cast("date"))
                  .withColumn("created_ts", F.current_timestamp()))

        _ensure_output_table_exists(out_df)
        _merge_into_output(out_df)

        if run_audit and audit_df is not None:
            print("Audit:")
            audit_df.show(truncate=False)

        # be nice to memory between months
        spark.catalog.clearCache()

    print("\nDONE. Output table:", OUTPUT_TABLE)

# ============================================================
# USAGE (ONLY THING YOU DECLARE)
# ============================================================

# Example: process month-by-month from Oct 2025 to Jan 2026
DL_START = "2024-01-31"
DL_END   = "2026-01-31"

run_ncb_incremental(
    dl_data_dt_start=DL_START,
    dl_data_dt_end=DL_END,
    cardx_membercodes=["CARDX"],
    use_latest_report_only=True,
    run_audit=False
)
