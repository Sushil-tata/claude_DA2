"""
Bureau Schema Adapter - Maps Actual Bureau Schema to Behavioral Physics Schema
==============================================================================

Handles schema mapping for:
- CIBIL/NCB bureau data (mnf_cra_rvw_* tables)
- Payment history string parsing (PAYMENTHISTORY1/2)
- DPD bucket conversion (OVERDUEMONTHS)
- Lender type classification (MEMBERSHORTNAME)

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import ArrayType, StringType
from typing import Dict, Optional
import re


class BureauSchemaAdapter:
    """
    Adapts actual bureau schema to expected behavioral physics schema.

    Supports:
    - CIBIL/NCB format (Thailand)
    - Column name mapping
    - Payment history parsing
    - DPD bucket conversion
    """

    # Schema mappings: Bureau column → Expected column
    ACCOUNT_SCHEMA_MAP = {
        "REF_NO": "ref_no",  # Keep original REF_NO (bureau report reference)
        "SEQ_TL": "seq_tl",  # Tradeline sequence number (unique per report)
        "ACCOUNTNUMBER": "account_id",
        "ASOFDATE": "as_of_month",
        "MEMBERSHORTNAME": "lender_name",
        # MEMBERCODE removed - only MEMBERSHORTNAME used
        "ACCOUNTTYPE": "account_type",
        "CREDITLIMIT": "credit_limit",
        "AMOUNTOWED": "balance",
        "AMOUNTPASTDUE": "past_due_amount",
        "ACCOUNTSTATUS": "account_status",
        "DATEACCOUNTOPENED": "account_open_date",
        "DATEACCOUNTCLOSED": "account_close_date",
        "DATEOFLASTPAYMENT": "last_payment_date",
        "DATEOFLASTDEBTRESTRUCTURE": "last_tdr_date",
        "INSTALLMENTAMOUNT": "emi_amount",
        "INSTALLMENTNUMBEROFPAYMENTS": "tenure_months",
        "PAYMENTHISTORYSTARTDATE": "payment_history_start",
        "PAYMENTHISTORYENDDATE": "payment_history_end",
        "PAYMENTHISTORY1": "payment_history_1",
        "PAYMENTHISTORY2": "payment_history_2",
    }

    HISTORY_SCHEMA_MAP = {
        "REF_NO": "ref_no",  # Keep original REF_NO (bureau report reference)
        "SEQ_TL": "seq_tl",  # Tradeline sequence number (unique per report)
        "ASOFDATE": "as_of_month",
        "RECEIVE_DT": "receive_dt",
        "DL_DATA_DT": "dl_data_dt",
        "CREDITLIMIT": "credit_limit",
        "AMOUNTOWED": "balance",
        "OVERDUEMONTHS": "dpd_bucket",
    }

    ENQUIRY_SCHEMA_MAP = {
        "REF_NO": "ref_no",  # Keep original REF_NO (bureau report reference)
        "DATEOFENQUIRY": "enquiry_date",
        "ENQUIRYPURPOSE": "enquiry_purpose",
        "ENQUIRYAMOUNT": "enquiry_amount",
        "MEMBERSHORTNAME": "lender_name",
        # MEMBERCODE removed - only MEMBERSHORTNAME (lender_name) used
    }

    # OVERDUEMONTHS to DPD conversion
    # NCB Thailand uses OVERDUEMONTHS field (integer months overdue)
    # Convert to DPD by multiplying by 30 days/month
    ODM_TO_DPD_MULTIPLIER = 30

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def adapt_bureau_trade_data(
        self,
        bureau_account_df: DataFrame,
        bureau_history_df: DataFrame,
        bridge_df: DataFrame,
        as_of_month: str
    ) -> DataFrame:
        """
        Adapt bureau account + history tables to behavioral physics schema.

        Creates monthly snapshots at (cust_id, account_id, as_of_month) grain
        with DPD extracted from history table and RECEIVE_DT from bridge.

        Args:
            bureau_account_df: mnf_cra_rvw_s_account (account master)
            bureau_history_df: mnf_cra_rvw_s_history (monthly snapshots)
            bridge_df: Point-in-time bridge with RECEIVE_DT (from build_bridge_df)
            as_of_month: Observation month (YYYY-MM-DD format)

        Returns:
            DataFrame with expected schema for behavioral physics
        """
        print("📊 Adapting bureau schema to behavioral physics format...")

        # 1. Map history table (monthly snapshots)
        history_mapped = self._map_columns(bureau_history_df, self.HISTORY_SCHEMA_MAP)

        # 2. Parse OVERDUEMONTHS to DPD days
        # The dpd_bucket column contains OVERDUEMONTHS values (months overdue)
        history_mapped = history_mapped.withColumn(
            "dpd",
            self._parse_overduemonths_to_dpd(F.col("dpd_bucket"))
        )

        # 3. Wire RECEIVE_DT from bridge (Bug Fix #2)
        # RECEIVE_DT exists in bridge (from mnf_cra_rvw_id_dummy), NOT in history table
        # Join bridge to attach RECEIVE_DT and DL_DATA_DT to each history row
        # Join on ref_no (normalized from REF_NO) to avoid silent type mismatches
        bridge_recv = bridge_df.select(
            F.col("REF_NO").alias("ref_no"),  # Normalize to lowercase for join
            "RECEIVE_DT",
            "DL_DATA_DT"
        ).distinct()

        history_mapped = history_mapped.join(
            bridge_recv,
            on="ref_no",  # Explicit ref_no join (REF_NO on both sides)
            how="left"
        )

        # Map RECEIVE_DT and DL_DATA_DT to internal schema
        history_mapped = history_mapped.withColumnRenamed("RECEIVE_DT", "receive_dt")
        history_mapped = history_mapped.withColumnRenamed("DL_DATA_DT", "dl_data_dt")

        # FIX 2: Create cust_id alias ONCE (after bridge join, before any other joins)
        # PRINCIPLE: cust_id is created HERE and propagates to all downstream joins
        # NO other DataFrame should create its own cust_id
        history_mapped = history_mapped.select(
            F.col("ref_no"),
            F.col("ref_no").alias("cust_id"),   # Compatibility alias - ONE instance only
            F.col("seq_tl"),
            F.col("as_of_month"),
            F.col("dpd"),
            F.col("dpd_bucket"),
            F.col("credit_limit"),
            F.col("balance"),
            F.col("receive_dt"),
            F.col("dl_data_dt")
        )

        print(f"✓ Wired RECEIVE_DT from bridge to {history_mapped.count():,} history rows")
        print(f"✓ Created cust_id alias ONCE (single source of truth)")

        # 4. Map account table
        account_mapped = self._map_columns(bureau_account_df, self.ACCOUNT_SCHEMA_MAP)

        # FIX 2: DO NOT create cust_id here - it comes from history_mapped join
        # account_mapped uses ref_no only

        # Note: Only lender_name (from MEMBERSHORTNAME) is used
        # No lender_id column created

        # 5. Parse payment history strings to get monthly DPD
        account_with_dpd = self._parse_payment_history(account_mapped)

        # 6. Join account master with history
        # Use history as primary source for monthly snapshots
        # Enrich with account master attributes
        # Unique tradeline identifier: (ref_no, seq_tl) = (REF_NO, SEQ_TL)
        # FIX 2: DO NOT select cust_id from account_static (avoid duplicate column)
        account_static = account_mapped.select(
            "ref_no", "seq_tl", "account_id", "lender_name", "account_type",
            "account_open_date", "account_close_date", "last_tdr_date",
            "emi_amount", "tenure_months"
        ).dropDuplicates(["ref_no", "seq_tl"])

        # Join history with account attributes on unique tradeline key
        # FIX 2: cust_id comes from history_mapped only (single source of truth)
        bureau_trade = history_mapped.join(
            account_static,
            on=["ref_no", "seq_tl"],  # Unique tradeline = (REF_NO, SEQ_TL)
            how="left"
        )

        # 7. Calculate utilization
        bureau_trade = bureau_trade.withColumn(
            "utilization",
            F.when(
                F.col("credit_limit") > 0,
                F.col("balance") / F.col("credit_limit")
            ).otherwise(0.0)
        )

        # 8. Add derived fields
        bureau_trade = bureau_trade.withColumn(
            "account_age_months",
            F.months_between(F.col("as_of_month"), F.col("account_open_date"))
        )

        # 9. TDR flag
        bureau_trade = bureau_trade.withColumn(
            "has_tdr",
            F.col("last_tdr_date").isNotNull().cast("int")
        )

        bureau_trade = bureau_trade.withColumn(
            "months_since_tdr",
            F.when(
                F.col("has_tdr") == 1,
                F.months_between(F.col("as_of_month"), F.col("last_tdr_date"))
            )
        ).withColumn(
            # Mark history table rows for differentiation
            "source", F.lit("history_table")
        ).withColumn(
            # Reporting gap features (null for history table - only from payment history)
            "has_reporting_gap", F.lit(None).cast("int")
        ).withColumn(
            "reporting_gap_count_12m", F.lit(None).cast("int")
        )

        # 10. Extend DPD time series with payment history (PAYMENTHISTORY1/2)
        # This adds up to 48 months of historical DPD data for months
        # not covered by the history table
        try:
            payment_history_snapshots = self.create_monthly_snapshots_from_payment_history(bureau_account_df)

            # Get existing (cust_id, account_id, as_of_month) combinations from history
            existing_months = bureau_trade.select("cust_id", "account_id", "as_of_month").distinct()

            # Filter payment history to only NEW months (extend backwards)
            # Use anti-join to exclude months already in history table
            extended_months = payment_history_snapshots.join(
                existing_months,
                on=["cust_id", "account_id", "as_of_month"],
                how="left_anti"
            )

            extended_rows = extended_months.count()

            if extended_rows > 0:
                # Union with existing bureau_trade
                # Align schemas by adding missing columns with nulls
                bureau_trade = bureau_trade.unionByName(
                    extended_months,
                    allowMissingColumns=True
                )

                print(f"✓ Extended DPD time series: added {extended_rows:,} payment history rows")
            else:
                print(f"✓ No additional payment history months to add (history table has full coverage)")

        except Exception as e:
            print(f"⚠️  Warning: Could not parse payment history: {e}")
            print(f"   Continuing with history table only ({bureau_trade.count():,} rows)")

        print(f"✓ Adapted bureau trade data: {bureau_trade.count():,} rows")

        return bureau_trade

    def adapt_bureau_enquiry_data(
        self,
        bureau_enquiry_df: DataFrame
    ) -> DataFrame:
        """
        Adapt bureau enquiry table to behavioral physics schema.

        Args:
            bureau_enquiry_df: mnf_cra_rvw_s_enquiry

        Returns:
            DataFrame with expected schema
        """
        print("📊 Adapting enquiry schema...")

        # Map columns
        enquiry_mapped = self._map_columns(bureau_enquiry_df, self.ENQUIRY_SCHEMA_MAP)

        # FIX 2: DO NOT create cust_id here
        # When this gets joined to panel, cust_id comes from panel (which gets it from history_mapped)
        # enquiry_mapped uses ref_no only for joins

        # Note: Only lender_name (from MEMBERSHORTNAME) is used
        # No lender_id column created

        # Parse enquiry date (ensure it's date type)
        enquiry_mapped = enquiry_mapped.withColumn(
            "enquiry_date",
            F.to_date(F.col("enquiry_date"))
        )

        print(f"✓ Adapted enquiry data: {enquiry_mapped.count()} rows")

        return enquiry_mapped

    def _map_columns(
        self,
        df: DataFrame,
        schema_map: Dict[str, str]
    ) -> DataFrame:
        """Map bureau column names to expected names"""
        select_expr = []

        for bureau_col, expected_col in schema_map.items():
            if bureau_col in df.columns:
                select_expr.append(F.col(bureau_col).alias(expected_col))

        # Add unmapped columns that might be useful
        for col in df.columns:
            if col not in schema_map and col not in ["SEGMENT", "TAG", "CHECK_D",
                                                       "CHECK_T", "DL_LOAD_TS"]:
                select_expr.append(F.col(col))

        return df.select(*select_expr)

    @staticmethod
    def _parse_overduemonths_to_dpd(odm_col):
        """
        Parse OVERDUEMONTHS field to DPD days.

        OVERDUEMONTHS is an integer representing months overdue.
        Convert to DPD by multiplying by 30 days/month.

        Args:
            odm_col: Column with OVERDUEMONTHS values (can be int or string)

        Returns:
            Column with DPD days (int), null if invalid
        """
        # Extract numeric digits from the column (handles both int and string)
        digits = F.regexp_extract(odm_col.cast("string"), r"(-?\d+)", 1)
        months_i = F.when(digits == "", F.lit(None)).otherwise(digits.cast("int"))

        # Convert months to DPD days (months * 30)
        # Use greatest(0, ...) to ensure non-negative DPD
        return (
            F.when(months_i.isNull(), F.lit(None).cast("int"))
             .otherwise(F.greatest(F.lit(0), months_i * F.lit(30)))
        )

    @staticmethod
    def ph_code_to_dpd(col_code):
        """
        Convert NCB Thailand payment history code to DPD days.

        NCB uses numeric codes for payment history:
        - "000" = 0 DPD (current)
        - "001" = 1-30 DPD → 30
        - "002" = 31-60 DPD → 60
        - "003" = 61-90 DPD → 90
        - "004" = 91-120 DPD → 120
        - "005" = 121-150 DPD → 150
        - "006" = 151-180 DPD → 180
        - "007" = 181-210 DPD → 210
        - "008" = 211-240 DPD → 240
        - "009" = 241-270 DPD → 270
        - Codes ending with "F" = Foreclosure → 300
        - "Y", "N" = Yes/No indicators → null
        - null/empty = null

        Args:
            col_code: Column with payment history codes (3-char strings)

        Returns:
            Column with DPD days (int), null if invalid
        """
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

    def _parse_payment_history(self, df: DataFrame) -> DataFrame:
        """
        Parse PAYMENTHISTORY1 and PAYMENTHISTORY2 strings.

        Format: "000000030060090..." where each 3-char segment is a month's DPD bucket
        Example: "000000030060" = [0, 0, 30, 60] → [0 DPD, 0 DPD, 31-60 DPD, 61-90 DPD]

        Extract monthly DPD values for trajectory analysis.
        """
        # This is complex - payment history is a string like "000000030060090"
        # Each 3 characters represents one month's DPD bucket
        # For now, we'll create a flag indicating if payment history exists
        # Full parsing would require exploding the string into monthly records

        df = df.withColumn(
            "has_payment_history",
            (F.col("payment_history_1").isNotNull() |
             F.col("payment_history_2").isNotNull()).cast("int")
        )

        # Extract max DPD from payment history string
        # Look for highest bucket code in the string
        df = df.withColumn(
            "max_dpd_from_history",
            F.when(
                F.col("payment_history_1").contains("180"), 270
            ).when(
                F.col("payment_history_1").contains("150"), 165
            ).when(
                F.col("payment_history_1").contains("090"), 120
            ).when(
                F.col("payment_history_1").contains("060"), 75
            ).when(
                F.col("payment_history_1").contains("030"), 45
            ).when(
                F.col("payment_history_1").contains("001"), 15
            ).otherwise(0)
        )

        return df

    def create_monthly_snapshots_from_payment_history(
        self,
        account_df: DataFrame
    ) -> DataFrame:
        """
        ADVANCED: Explode payment history strings into monthly records.

        This creates a time-series by parsing the payment history string.
        Each 3-character segment becomes a separate month.

        Args:
            account_df: Account data with PAYMENTHISTORY1/2

        Returns:
            DataFrame with monthly snapshots from payment history
        """
        print("📊 Exploding payment history into monthly snapshots...")

        # Map schema first
        df = self._map_columns(account_df, self.ACCOUNT_SCHEMA_MAP)

        # UDF to parse payment history string into array of 3-char monthly codes
        def split_string_into_3_chars(input_string):
            """
            Parse payment history string into 3-character chunks.
            Example: '000030060090' -> ['000', '030', '060', '090']

            Handles:
            - Null/empty strings → []
            - Zero-width spaces and regular spaces (strip them)
            """
            if input_string is None:
                return []
            # Remove zero-width spaces and regular spaces
            s = str(input_string).replace("\u200b", "").replace(" ", "")
            if len(s) == 0:
                return []
            # Split into 3-character chunks
            return [s[i:i+3] for i in range(0, len(s), 3)]

        split_string_udf = F.udf(split_string_into_3_chars, ArrayType(StringType()))

        # Parse payment history into array
        df = df.withColumn(
            "payment_history_array",
            split_string_udf(F.col("payment_history_1"))
        )

        # Add reporting gap features (account-level)
        # has_reporting_gap = 1 if "XXX" exists anywhere in payment history
        df = df.withColumn(
            "has_reporting_gap",
            F.when(
                F.col("payment_history_1").contains("XXX"),
                1
            ).otherwise(0)
        )

        # reporting_gap_count_12m = count of "XXX" in last 12 positions (most recent 12 months)
        # UDF to count "XXX" in last 12 positions
        def count_xxx_in_last_12_udf(payment_history_str):
            """Count 'XXX' in last 12 3-char segments (most recent 12 months)"""
            if not payment_history_str or len(payment_history_str) < 3:
                return 0

            # Get last 36 characters (12 segments * 3 chars each)
            last_12_months = payment_history_str[-36:] if len(payment_history_str) >= 36 else payment_history_str

            # Split into 3-character chunks
            chunks = [last_12_months[i:i+3] for i in range(0, len(last_12_months), 3)]

            # Count "XXX" occurrences
            return sum(1 for chunk in chunks if chunk == "XXX")

        count_xxx_udf = F.udf(count_xxx_in_last_12_udf, "int")

        df = df.withColumn(
            "reporting_gap_count_12m",
            count_xxx_udf(F.col("payment_history_1"))
        )

        # Explode array with position (month offset)
        # Use selectExpr with posexplode to properly extract pos and value
        df = df.selectExpr(
            "*",
            "posexplode(payment_history_array) as (months_back, dpd_bucket)"
        )

        # Calculate as_of_month by subtracting months from payment_history_end
        df = df.withColumn(
            "as_of_month",
            F.add_months(
                F.col("payment_history_end"),
                -F.col("months_back")
            )
        )

        # Convert payment history code to DPD days
        # dpd_bucket contains NCB payment history codes ("000", "001", "002", ...")
        df = df.withColumn(
            "dpd",
            self.ph_code_to_dpd(F.col("dpd_bucket"))
        )

        # FIX 2: Create cust_id as alias of ref_no (for schema compatibility with bureau_trade)
        # Payment history snapshots will be union'd with bureau_trade, which has cust_id
        df = df.withColumn("cust_id", F.col("ref_no"))

        # Select relevant columns (DPD-only - no balance/credit_limit from payment strings)
        monthly_snapshots = df.select(
            "ref_no", "cust_id", "account_id", "as_of_month",
            "lender_name", "account_type",
            "dpd_bucket", "dpd",
            "account_open_date", "last_tdr_date",
            "has_reporting_gap",           # Account-level: 1 if any XXX in payment history
            "reporting_gap_count_12m"      # Account-level: count of XXX in last 12 months
        ).withColumn(
            # Mark as payment history source (for debugging/validation)
            "source", F.lit("payment_history")
        ).withColumn(
            # Nulls for balance/credit_limit (not available in payment strings)
            "balance", F.lit(None).cast("double")
        ).withColumn(
            "credit_limit", F.lit(None).cast("double")
        ).withColumn(
            "utilization", F.lit(None).cast("double")
        )

        print(f"✓ Created {monthly_snapshots.count()} monthly DPD snapshots from payment history")

        return monthly_snapshots


# ============================================================================
# Example Usage & Testing
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date

    spark = SparkSession.builder \
        .appName("BureauSchemaAdapterTest") \
        .master("local[*]") \
        .getOrCreate()

    print("\n" + "="*70)
    print("BUREAU SCHEMA ADAPTER - TEST")
    print("="*70)

    # Create sample data matching actual bureau schema

    # Sample mnf_cra_rvw_s_history
    history_data = [
        ("CUST001", date(2024, 1, 31), 50000, 10000, "000", 1),
        ("CUST001", date(2024, 2, 29), 50000, 12000, "000", 1),
        ("CUST001", date(2024, 3, 31), 50000, 15000, "030", 1),
        ("CUST001", date(2024, 4, 30), 50000, 17000, "060", 1),
        ("CUST002", date(2024, 1, 31), 100000, 20000, "000", 2),
        ("CUST002", date(2024, 2, 29), 100000, 25000, "090", 2),
    ]

    history_df = spark.createDataFrame(
        history_data,
        ["REF_NO", "ASOFDATE", "CREDITLIMIT", "AMOUNTOWED", "OVERDUEMONTHS", "SEQ_TL"]
    )

    # Sample mnf_cra_rvw_s_account
    account_data = [
        ("CUST001", "ACC001", "BANGKOK BANK", "AUTO LOAN", date(2023, 1, 1),
         None, "000000030060", 5000, 60),
        ("CUST002", "ACC002", "KASIKORN BANK", "CREDIT CARD", date(2022, 6, 1),
         date(2024, 1, 15), "000090", 0, 0),
    ]

    account_df = spark.createDataFrame(
        account_data,
        ["REF_NO", "ACCOUNTNUMBER", "MEMBERSHORTNAME", "ACCOUNTTYPE",
         "DATEACCOUNTOPENED", "DATEOFLASTDEBTRESTRUCTURE", "PAYMENTHISTORY1",
         "INSTALLMENTAMOUNT", "INSTALLMENTNUMBEROFPAYMENTS"]
    )

    # Sample mnf_cra_rvw_s_enquiry
    enquiry_data = [
        ("CUST001", date(2024, 1, 15), "PERSONAL LOAN", 50000, "SCB"),
        ("CUST001", date(2024, 3, 20), "CREDIT CARD", 0, "KRUNGSRI"),
    ]

    enquiry_df = spark.createDataFrame(
        enquiry_data,
        ["REF_NO", "DATEOFENQUIRY", "ENQUIRYPURPOSE", "ENQUIRYAMOUNT", "MEMBERSHORTNAME"]
    )

    # Test adapter
    adapter = BureauSchemaAdapter(spark)

    print("\n1. Testing Bureau Trade Adaptation:")
    bureau_trade = adapter.adapt_bureau_trade_data(account_df, history_df)
    bureau_trade.select(
        "cust_id", "as_of_month", "dpd", "balance", "credit_limit",
        "utilization", "lender_name"
    ).show()

    print("\n2. Testing Enquiry Adaptation:")
    bureau_enquiry = adapter.adapt_bureau_enquiry_data(enquiry_df)
    bureau_enquiry.select(
        "cust_id", "enquiry_date", "enquiry_purpose", "lender_name"
    ).show()

    print("\n3. Testing Payment History Parsing:")
    monthly_snapshots = adapter.create_monthly_snapshots_from_payment_history(account_df)
    monthly_snapshots.select(
        "cust_id", "as_of_month", "dpd_bucket", "dpd"
    ).orderBy("cust_id", "as_of_month").show()

    print("\n✅ Schema adapter test complete")
    print("\nColumn Mappings:")
    print("  - REF_NO → cust_id")
    print("  - ASOFDATE → as_of_month")
    print("  - OVERDUEMONTHS → dpd (converted to days)")
    print("  - AMOUNTOWED → balance")
    print("  - CREDITLIMIT → credit_limit")
    print("  - MEMBERSHORTNAME → lender_name")

    spark.stop()
