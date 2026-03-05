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
        "MEMBERCODE": "lender_code",
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
        "MEMBERCODE": "lender_code",
    }

    # DPD bucket to numeric mapping
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
            "cust_id", "account_id", "lender_name", "account_type",
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
        )

        print(f"✓ Adapted bureau trade data: {bureau_trade.count()} rows")

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
                                                       "CHECK_T", "RECEIVE_DT",
                                                       "DL_LOAD_TS", "DL_DATA_DT"]:
                select_expr.append(F.col(col))

        return df.select(*select_expr)

    def _convert_dpd_bucket(self, df: DataFrame) -> DataFrame:
        """
        Convert DPD bucket (e.g., '030', '060') to numeric DPD.

        OVERDUEMONTHS format: '030' = 31-60 days, '060' = 61-90 days, etc.
        """
        # Create mapping UDF
        dpd_map_expr = F.when(F.col("dpd_bucket").isNull(), 0)

        for bucket, dpd_value in self.DPD_BUCKET_MAP.items():
            dpd_map_expr = dpd_map_expr.when(
                F.col("dpd_bucket") == bucket, dpd_value
            )

        # Default to 0 if bucket not recognized
        dpd_map_expr = dpd_map_expr.otherwise(0)

        df = df.withColumn("dpd", dpd_map_expr)

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

        # Explode array with position (month offset)
        df = df.withColumn(
            "month_data",
            F.posexplode(F.col("payment_history_array"))
        )

        # Extract position (months back from end date) and DPD bucket
        df = df.withColumn("months_back", F.col("month_data.pos"))
        df = df.withColumn("dpd_bucket", F.col("month_data.col"))

        # Calculate as_of_month by subtracting months from payment_history_end
        df = df.withColumn(
            "as_of_month",
            F.add_months(
                F.col("payment_history_end"),
                -F.col("months_back")
            )
        )

        # Convert DPD bucket to numeric
        df = self._convert_dpd_bucket(df)

        # Select relevant columns
        monthly_snapshots = df.select(
            "cust_id", "account_id", "as_of_month",
            "lender_name", "account_type",
            "dpd_bucket", "dpd",
            "credit_limit", "balance",
            "account_open_date", "last_tdr_date"
        )

        print(f"✓ Created {monthly_snapshots.count()} monthly snapshots from payment history")

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
    print("  - OVERDUEMONTHS → dpd (converted from bucket)")
    print("  - AMOUNTOWED → balance")
    print("  - CREDITLIMIT → credit_limit")
    print("  - MEMBERSHORTNAME → lender_name")

    spark.stop()
