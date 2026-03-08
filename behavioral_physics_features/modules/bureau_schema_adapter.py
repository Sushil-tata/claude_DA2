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
        "REF_NO": "cust_id",
        "ACCOUNTNUMBER": "account_id",
        "ASOFDATE": "as_of_month",
        "MEMBERSHORTNAME": "lender_name",
        "MEMBERCODE": "lender_id",  # Changed from lender_code to lender_id
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
        "REF_NO": "cust_id",
        "ASOFDATE": "as_of_month",
        "RECEIVE_DT": "receive_dt",
        "DL_DATA_DT": "dl_data_dt",
        "CREDITLIMIT": "credit_limit",
        "AMOUNTOWED": "balance",
        "OVERDUEMONTHS": "dpd_bucket",
        "SEQ_TL": "account_seq",
    }

    ENQUIRY_SCHEMA_MAP = {
        "REF_NO": "cust_id",
        "DATEOFENQUIRY": "enquiry_date",
        "ENQUIRYPURPOSE": "enquiry_purpose",
        "ENQUIRYAMOUNT": "enquiry_amount",
        "MEMBERSHORTNAME": "lender_name",
        "MEMBERCODE": "lender_id",  # Changed from lender_code to lender_id
    }

    # DPD bucket to numeric mapping (for behavioral physics features)
    DPD_BUCKET_MAP = {
        "000": 0,
        "001": 15,   # 1-30 days → midpoint 15
        "030": 45,   # 31-60 days → midpoint 45
        "060": 75,   # 61-90 days → midpoint 75
        "090": 120,  # 91-150 days → midpoint 120
        "150": 165,  # 151-180 days → midpoint 165
        "180": 270,  # 181+ days → 270
        "XXX": 0,    # No data
        "STD": 0,    # Standard (no overdue)
        "SUB": 45,   # Substandard
        "DBT": 120,  # Doubtful
        "LSS": 270,  # Loss
    }

    # DPD bucket to ordinal mapping (for scorecard WOE binning)
    # Preserves natural risk categories without midpoint conversion
    DPD_BUCKET_ORDINAL_MAP = {
        "000": 0,    # Current (0 DPD)
        "001": 1,    # 1-30 DPD
        "030": 2,    # 31-60 DPD
        "060": 3,    # 61-90 DPD
        "090": 4,    # 91-150 DPD
        "150": 5,    # 151-180 DPD
        "180": 6,    # 181+ DPD
        # Note: "XXX" (not reported) maps to null, not 0
    }

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def adapt_bureau_trade_data(
        self,
        account_df: DataFrame,
        history_df: DataFrame
    ) -> DataFrame:
        """
        Adapt bureau account + history tables to behavioral physics schema.

        Creates monthly snapshots at (cust_id, account_id, as_of_month) grain
        with DPD extracted from history table.

        Args:
            account_df: mnf_cra_rvw_s_account (account master)
            history_df: mnf_cra_rvw_s_history (monthly snapshots)

        Returns:
            DataFrame with expected schema for behavioral physics
        """
        print("📊 Adapting bureau schema to behavioral physics format...")

        # 1. Map history table (monthly snapshots)
        history_mapped = self._map_columns(history_df, self.HISTORY_SCHEMA_MAP)

        # 2. Convert DPD bucket to numeric
        history_mapped = self._convert_dpd_bucket(history_mapped)

        # 3. Map account table
        account_mapped = self._map_columns(account_df, self.ACCOUNT_SCHEMA_MAP)

        # 4. Parse payment history strings to get monthly DPD
        account_with_dpd = self._parse_payment_history(account_mapped)

        # 5. Join account master with history
        # Use history as primary source for monthly snapshots
        # Enrich with account master attributes
        account_static = account_mapped.select(
            "cust_id", "account_id", "lender_name", "lender_id", "account_type",
            "account_open_date", "account_close_date", "last_tdr_date",
            "emi_amount", "tenure_months"
        ).dropDuplicates(["cust_id", "account_id"])

        # Join history with account attributes
        # Note: history table has account_seq instead of account_id
        # We'll need to join by (cust_id, as_of_month) and use seq to match
        bureau_trade = history_mapped.join(
            account_static,
            on=["cust_id"],
            how="left"
        )

        # 6. Calculate utilization
        bureau_trade = bureau_trade.withColumn(
            "utilization",
            F.when(
                F.col("credit_limit") > 0,
                F.col("balance") / F.col("credit_limit")
            ).otherwise(0.0)
        )

        # 7. Add derived fields
        bureau_trade = bureau_trade.withColumn(
            "account_age_months",
            F.months_between(F.col("as_of_month"), F.col("account_open_date"))
        )

        # 8. TDR flag
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

        # 9. Extend DPD time series with payment history (PAYMENTHISTORY1/2)
        # This adds up to 48 months of historical DPD data for months
        # not covered by the history table
        try:
            payment_history_snapshots = self.create_monthly_snapshots_from_payment_history(account_df)

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
        enquiry_df: DataFrame
    ) -> DataFrame:
        """
        Adapt bureau enquiry table to behavioral physics schema.

        Args:
            enquiry_df: mnf_cra_rvw_s_enquiry

        Returns:
            DataFrame with expected schema
        """
        print("📊 Adapting enquiry schema...")

        # Map columns
        enquiry_mapped = self._map_columns(enquiry_df, self.ENQUIRY_SCHEMA_MAP)

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

    def _convert_dpd_bucket(self, df: DataFrame) -> DataFrame:
        """
        Convert DPD bucket to both numeric DPD and ordinal category.

        Creates two columns:
        - dpd: Numeric midpoint (for behavioral physics features)
        - dpd_bucket_ordinal: Ordinal 0-6 (for scorecard WOE binning)

        OVERDUEMONTHS format: '030' = 31-60 days, '060' = 61-90 days, etc.

        Special handling:
        - "XXX" (not reported) → null for both dpd and dpd_bucket_ordinal
        - null → 0 (assume current if no data)
        """
        # Create numeric DPD mapping (midpoints for velocity/acceleration)
        # "XXX" (not reported) should map to null, not 0
        dpd_map_expr = F.when(F.col("dpd_bucket") == "XXX", F.lit(None).cast("int"))
        dpd_map_expr = dpd_map_expr.when(F.col("dpd_bucket").isNull(), 0)

        for bucket, dpd_value in self.DPD_BUCKET_MAP.items():
            dpd_map_expr = dpd_map_expr.when(
                F.col("dpd_bucket") == bucket, dpd_value
            )

        # Default to 0 if bucket not recognized (but not XXX)
        dpd_map_expr = dpd_map_expr.otherwise(0)

        df = df.withColumn("dpd", dpd_map_expr)

        # Create ordinal DPD mapping (0-6 categories for WOE binning)
        # "XXX" (not reported) should map to null, not 0
        dpd_ordinal_expr = F.when(F.col("dpd_bucket") == "XXX", F.lit(None).cast("int"))
        dpd_ordinal_expr = dpd_ordinal_expr.when(F.col("dpd_bucket").isNull(), 0)

        for bucket, ordinal_value in self.DPD_BUCKET_ORDINAL_MAP.items():
            dpd_ordinal_expr = dpd_ordinal_expr.when(
                F.col("dpd_bucket") == bucket, ordinal_value
            )

        # Default to 0 if bucket not recognized (but not XXX)
        dpd_ordinal_expr = dpd_ordinal_expr.otherwise(0)

        df = df.withColumn("dpd_bucket_ordinal", dpd_ordinal_expr)

        return df

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

        # UDF to parse payment history string into array of monthly buckets
        def parse_payment_history_udf(payment_history_str):
            """Parse '000030060090' -> ['000', '030', '060', '090']"""
            if not payment_history_str:
                return []

            # Split into 3-character chunks
            chunks = [payment_history_str[i:i+3] for i in range(0, len(payment_history_str), 3)]
            return chunks

        parse_udf = F.udf(parse_payment_history_udf, "array<string>")

        # Parse payment history into array
        df = df.withColumn(
            "payment_history_array",
            parse_udf(F.col("payment_history_1"))
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

        # Convert DPD bucket to numeric and ordinal
        df = self._convert_dpd_bucket(df)

        # Select relevant columns (DPD-only - no balance/credit_limit from payment strings)
        monthly_snapshots = df.select(
            "cust_id", "account_id", "as_of_month",
            "lender_name", "lender_id", "account_type",
            "dpd_bucket", "dpd", "dpd_bucket_ordinal",
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
        "utilization", "lender_name", "lender_id"
    ).show()

    print("\n2. Testing Enquiry Adaptation:")
    bureau_enquiry = adapter.adapt_bureau_enquiry_data(enquiry_df)
    bureau_enquiry.select(
        "cust_id", "enquiry_date", "enquiry_purpose", "lender_name", "lender_id"
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
    print("  - OVERDUEMONTHS → dpd (converted from bucket)")
    print("  - AMOUNTOWED → balance")
    print("  - CREDITLIMIT → credit_limit")
    print("  - MEMBERSHORTNAME → lender_name")
    print("  - MEMBERCODE → lender_id")

    spark.stop()
