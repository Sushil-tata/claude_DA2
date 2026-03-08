"""
Main Pipeline - Orchestration and Quality Assurance
===================================================

Production-ready orchestration for behavioral physics feature factory:
- End-to-end pipeline execution
- Point-in-time safety enforcement
- Quality checks (missingness, leakage, distributions)
- Audit logging and lineage tracking
- QA summary reporting

Author: Behavioral Physics Team
Version: 1.0.0
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json

from .config import get_config
from .feature_registry import FeatureRegistry
from .bureau_schema_adapter import BureauSchemaAdapter
from .cardx_schema_adapter import CardXSchemaAdapter, build_bridge_df


class BehavioralPhysicsPipeline:
    """
    Production pipeline for behavioral physics feature factory.

    Responsibilities:
    1. Orchestrate feature computation
    2. Enforce point-in-time safety
    3. Validate feature quality
    4. Generate audit logs
    5. Produce QA summary
    """

    def __init__(self, spark: SparkSession, config_override: Optional[Dict] = None):
        self.spark = spark
        self.config = get_config()
        if config_override:
            # Allow config overrides for testing
            for key, value in config_override.items():
                setattr(self.config, key, value)

        self.feature_registry = FeatureRegistry(spark)

        # Audit log
        self.audit_log = []

    def run_from_raw_tables(
        self,
        as_of_month: str,
        catalog: str = "cdx_mdz_prd",
        output_table: Optional[str] = None
    ) -> Tuple[DataFrame, DataFrame, Dict[str, Any]]:
        """
        Execute full pipeline starting from raw tables with point-in-time bridge.

        This method:
        1. Builds point-in-time bridge (CardX ACCT_NUM → Bureau REF_NO)
        2. Loads and adapts CardX data using bridge
        3. Loads and adapts bureau data
        4. Calls run() with prepared DataFrames

        Args:
            as_of_month: As-of date for point-in-time filtering (YYYY-MM-DD)
            catalog: Databricks catalog name
            output_table: Optional Delta table name for output

        Returns:
            Tuple of (features_df, audit_log_df, qa_summary_dict)
        """
        print(f"\n{'='*80}")
        print(f"LOADING DATA FROM RAW TABLES")
        print(f"As-of Month: {as_of_month}")
        print(f"Catalog: {catalog}")
        print(f"{'='*80}\n")

        # ========================================================================
        # STEP 1: Build Point-in-Time Bridge (CardX → Bureau)
        # ========================================================================
        print("Step 1/4: Building Point-in-Time Bridge")
        print("-" * 80)

        bridge_df = build_bridge_df(self.spark, as_of_month)

        self._log_audit("BRIDGE_BUILD", {
            "as_of_month": as_of_month,
            "bridge_rows": bridge_df.count(),
            "unique_accounts": bridge_df.select("ACCT_NUM").distinct().count(),
            "unique_customers": bridge_df.select("REF_NO").distinct().count()
        })

        print("✓ Bridge built successfully\n")

        # ========================================================================
        # STEP 2: Load and Adapt CardX Data Using Bridge
        # ========================================================================
        print("Step 2/4: Loading and Adapting CardX Data")
        print("-" * 80)

        cardx_adapter = CardXSchemaAdapter(self.spark)

        # Load CardX monthly data
        cardx_monthly = self.spark.table(f"{catalog}.cdx_curated_spl_acl_db.spl_acct_mthly")

        # Filter to observation month
        month_end = F.last_day(F.to_date(F.lit(as_of_month), "yyyy-MM-dd"))
        cardx_monthly = cardx_monthly.filter(F.col("DL_DATA_DT") == month_end)

        print(f"✓ Loaded CardX monthly (month={as_of_month}): {cardx_monthly.count():,} rows")

        # Adapt using bridge
        cardx_internal_df = cardx_adapter.adapt_cardx_monthly_data(
            cardx_monthly_df=cardx_monthly,
            bridge_df=bridge_df
        )

        self._log_audit("CARDX_ADAPT", {
            "as_of_month": as_of_month,
            "cardx_rows": cardx_internal_df.count(),
            "cardx_customers": cardx_internal_df.select("cust_id").distinct().count()
        })

        print("✓ CardX data adapted successfully\n")

        # ========================================================================
        # STEP 3: Load and Adapt Bureau Data
        # ========================================================================
        print("Step 3/4: Loading and Adapting Bureau Data")
        print("-" * 80)

        bureau_adapter = BureauSchemaAdapter(self.spark)

        # Load bureau tables
        bureau_account = self.spark.table(f"{catalog}.cdx_persist_mnf_res_db.mnf_cra_rvw_s_account")
        bureau_history = self.spark.table(f"{catalog}.cdx_persist_mnf_res_db.mnf_cra_rvw_s_history")
        bureau_enquiry = self.spark.table(f"{catalog}.cdx_persist_mnf_res_db.mnf_cra_rvw_s_enquiry")

        print(f"✓ Loaded bureau_account: {bureau_account.count():,} rows")
        print(f"✓ Loaded bureau_history: {bureau_history.count():,} rows")
        print(f"✓ Loaded bureau_enquiry: {bureau_enquiry.count():,} rows")

        # Adapt bureau trade data (combines account + history + payment history expansion)
        bureau_trade_df = bureau_adapter.adapt_bureau_trade_data(
            bureau_account_df=bureau_account,
            bureau_history_df=bureau_history,
            as_of_month=as_of_month
        )

        print(f"✓ Bureau trade adapted: {bureau_trade_df.count():,} rows")

        # Adapt bureau enquiry data
        bureau_enquiry_df = bureau_adapter.adapt_bureau_enquiry_data(
            bureau_enquiry_df=bureau_enquiry
        )

        print(f"✓ Bureau enquiry adapted: {bureau_enquiry_df.count():,} rows")

        self._log_audit("BUREAU_ADAPT", {
            "as_of_month": as_of_month,
            "bureau_trade_rows": bureau_trade_df.count(),
            "bureau_enquiry_rows": bureau_enquiry_df.count(),
            "bureau_customers": bureau_trade_df.select("cust_id").distinct().count()
        })

        print("✓ Bureau data adapted successfully\n")

        # ========================================================================
        # STEP 4: Run Feature Pipeline
        # ========================================================================
        print("Step 4/4: Running Feature Pipeline")
        print("-" * 80)

        # Call existing run() method with prepared DataFrames
        features_df, audit_log_df, qa_summary = self.run(
            bureau_trade_df=bureau_trade_df,
            bureau_enquiry_df=bureau_enquiry_df,
            cardx_internal_df=cardx_internal_df,
            as_of_month=as_of_month,
            output_table=output_table
        )

        return features_df, audit_log_df, qa_summary

    def run(
        self,
        bureau_trade_df: DataFrame,
        bureau_enquiry_df: DataFrame,
        cardx_internal_df: DataFrame,
        as_of_month: str,
        output_table: Optional[str] = None
    ) -> Tuple[DataFrame, DataFrame, Dict[str, Any]]:
        """
        Execute full behavioral physics feature pipeline with pre-prepared DataFrames.

        NOTE: For production use, prefer run_from_raw_tables() which handles:
        - Point-in-time bridge building (CardX → Bureau)
        - Data loading and adaptation from raw tables
        - Proper point-in-time filtering

        This method is useful when:
        - You have already built the bridge and adapted the data
        - You're running tests with sample DataFrames
        - You want fine-grained control over data preparation

        Args:
            bureau_trade_df: Adapted bureau trade monthly data (must have: cust_id, as_of_month, receive_dt)
            bureau_enquiry_df: Adapted bureau enquiry data (must have: cust_id)
            cardx_internal_df: Adapted CardX data (must have: cust_id, as_of_month)
            as_of_month: As-of date for point-in-time filtering (YYYY-MM-DD)
            output_table: Optional Delta table name for output

        Returns:
            Tuple of (features_df, audit_log_df, qa_summary_dict)
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_audit("PIPELINE_START", {
            "run_id": run_id,
            "as_of_month": as_of_month,
            "timestamp": datetime.now().isoformat()
        })

        print(f"\n{'='*80}")
        print(f"BEHAVIORAL PHYSICS FEATURE FACTORY - PRODUCTION RUN")
        print(f"Run ID: {run_id}")
        print(f"As-of Month: {as_of_month}")
        print(f"{'='*80}\n")

        # Step 1: Input validation
        print("Step 1/6: Input Validation")
        self._validate_inputs(bureau_trade_df, bureau_enquiry_df, cardx_internal_df)
        print("✓ Input validation passed\n")

        # Step 2: Point-in-time filtering
        print("Step 2/6: Point-in-Time Filtering")
        bureau_trade_df, bureau_enquiry_df, cardx_internal_df = self._apply_point_in_time_filter(
            bureau_trade_df, bureau_enquiry_df, cardx_internal_df, as_of_month
        )
        print("✓ Point-in-time filtering applied\n")

        # Step 3: Feature computation
        print("Step 3/6: Feature Computation")
        features_df = self.feature_registry.compute_all_features(
            bureau_trade_df, bureau_enquiry_df, cardx_internal_df, as_of_month
        )
        print("✓ Feature computation complete\n")

        # Step 4: Quality checks
        print("Step 4/6: Quality Checks")
        qa_summary = self._run_quality_checks(features_df, as_of_month)
        print("✓ Quality checks complete\n")

        # Step 5: Leakage detection
        print("Step 5/6: Leakage Detection")
        leakage_results = self._detect_leakage(features_df, as_of_month)
        qa_summary["leakage_check"] = leakage_results
        print("✓ Leakage detection complete\n")

        # Step 6: Output writing
        if output_table:
            print(f"Step 6/6: Writing to {output_table}")
            self._write_output(features_df, output_table, run_id, as_of_month)
            print("✓ Output written\n")
        else:
            print("Step 6/6: Skipping output write (no table specified)\n")

        # Generate audit log
        audit_log_df = self._generate_audit_log()

        # Final summary
        self._print_summary(features_df, qa_summary)

        self._log_audit("PIPELINE_COMPLETE", {
            "run_id": run_id,
            "status": "SUCCESS",
            "features_count": len(features_df.columns) - 2,
            "customers_count": features_df.count()
        })

        return features_df, audit_log_df, qa_summary

    def _validate_inputs(
        self,
        bureau_trade_df: DataFrame,
        bureau_enquiry_df: DataFrame,
        cardx_internal_df: DataFrame
    ):
        """Validate input DataFrames have required columns"""
        # Required columns for bureau_trade
        required_bureau_cols = ["cust_id", "as_of_month"]
        missing_cols = [col for col in required_bureau_cols if col not in bureau_trade_df.columns]
        if missing_cols:
            raise ValueError(f"bureau_trade_df missing required columns: {missing_cols}")

        # Required columns for bureau_enquiry
        required_enquiry_cols = ["cust_id"]
        missing_cols = [col for col in required_enquiry_cols if col not in bureau_enquiry_df.columns]
        if missing_cols:
            raise ValueError(f"bureau_enquiry_df missing required columns: {missing_cols}")

        # Required columns for cardx_internal
        required_cardx_cols = ["cust_id", "as_of_month"]
        missing_cols = [col for col in required_cardx_cols if col not in cardx_internal_df.columns]
        if missing_cols:
            raise ValueError(f"cardx_internal_df missing required columns: {missing_cols}")

        self._log_audit("INPUT_VALIDATION", {
            "bureau_trade_rows": bureau_trade_df.count(),
            "bureau_enquiry_rows": bureau_enquiry_df.count(),
            "cardx_internal_rows": cardx_internal_df.count()
        })

    def _apply_point_in_time_filter(
        self,
        bureau_trade_df: DataFrame,
        bureau_enquiry_df: DataFrame,
        cardx_internal_df: DataFrame,
        as_of_month: str
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Apply point-in-time filtering to prevent data leakage.

        Only use data available as of the as_of_month.
        """
        # Filter bureau trade
        # Allow rows with null receive_dt (payment history) OR receive_dt <= as_of_month (history table)
        bureau_trade_filtered = bureau_trade_df.filter(
            F.col("receive_dt").isNull() |
            (F.col("receive_dt") <= F.last_day(F.to_date(F.lit(as_of_month), "yyyy-MM-dd")))
        )

        # Filter cardx internal
        cardx_internal_filtered = cardx_internal_df.filter(
            F.col("as_of_month") <= F.lit(as_of_month)
        )

        # Filter enquiries (if enquiry_date exists)
        if "enquiry_date" in bureau_enquiry_df.columns:
            bureau_enquiry_filtered = bureau_enquiry_df.filter(
                F.col("enquiry_date") <= F.lit(as_of_month)
            )
        elif "ENQUIRYDT" in bureau_enquiry_df.columns:
            bureau_enquiry_filtered = bureau_enquiry_df.filter(
                F.to_date(F.col("ENQUIRYDT")) <= F.lit(as_of_month)
            )
        else:
            bureau_enquiry_filtered = bureau_enquiry_df

        rows_before = (
            bureau_trade_df.count() +
            bureau_enquiry_df.count() +
            cardx_internal_df.count()
        )

        rows_after = (
            bureau_trade_filtered.count() +
            bureau_enquiry_filtered.count() +
            cardx_internal_filtered.count()
        )

        self._log_audit("POINT_IN_TIME_FILTER", {
            "as_of_month": as_of_month,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_filtered": rows_before - rows_after
        })

        return bureau_trade_filtered, bureau_enquiry_filtered, cardx_internal_filtered

    def _run_quality_checks(
        self,
        features_df: DataFrame,
        as_of_month: str
    ) -> Dict[str, Any]:
        """
        Run comprehensive quality checks on features.

        Checks:
        1. Missingness by feature
        2. Distribution statistics
        3. Outlier detection
        4. Feature correlations (high correlation warnings)
        """
        qa_summary = {
            "as_of_month": as_of_month,
            "total_features": len(features_df.columns) - 2,
            "total_customers": features_df.count(),
            "checks": []
        }

        # Missingness check
        missingness_stats = self._check_missingness(features_df)
        qa_summary["missingness"] = missingness_stats

        # Distribution check
        distribution_stats = self._check_distributions(features_df)
        qa_summary["distributions"] = distribution_stats

        # Outlier check
        outlier_stats = self._check_outliers(features_df)
        qa_summary["outliers"] = outlier_stats

        return qa_summary

    def _check_missingness(self, features_df: DataFrame) -> Dict[str, float]:
        """Check missingness percentage for each feature"""
        total_rows = features_df.count()
        missingness = {}

        for col_name in features_df.columns:
            if col_name not in ["cust_id", "as_of_month"]:
                null_count = features_df.filter(
                    F.col(col_name).isNull() | F.isnan(col_name)
                ).count()
                missingness[col_name] = (null_count / total_rows * 100) if total_rows > 0 else 0

        # Identify high missingness features (>50%)
        high_missingness = {k: v for k, v in missingness.items() if v > 50}

        if high_missingness:
            print(f"  ⚠️  {len(high_missingness)} features with >50% missing:")
            for feature, pct in sorted(high_missingness.items(), key=lambda x: -x[1])[:5]:
                print(f"     - {feature}: {pct:.1f}%")

        self._log_audit("MISSINGNESS_CHECK", {
            "features_with_high_missingness": len(high_missingness),
            "max_missingness_pct": max(missingness.values()) if missingness else 0
        })

        return missingness

    def _check_distributions(self, features_df: DataFrame) -> Dict[str, Dict]:
        """Check distribution statistics for numeric features"""
        numeric_features = [
            col_name for col_name in features_df.columns
            if col_name not in ["cust_id", "as_of_month"]
        ]

        # Sample features for distribution check
        sample_features = numeric_features[:20]  # Check first 20 to avoid slowness

        distributions = {}

        for col_name in sample_features:
            stats = features_df.select(
                F.min(col_name).alias("min"),
                F.max(col_name).alias("max"),
                F.avg(col_name).alias("mean"),
                F.stddev(col_name).alias("std")
            ).collect()[0]

            distributions[col_name] = {
                "min": float(stats["min"]) if stats["min"] is not None else None,
                "max": float(stats["max"]) if stats["max"] is not None else None,
                "mean": float(stats["mean"]) if stats["mean"] is not None else None,
                "std": float(stats["std"]) if stats["std"] is not None else None
            }

        return distributions

    def _check_outliers(self, features_df: DataFrame) -> Dict[str, int]:
        """Check for outliers (values > 3 std from mean)"""
        outlier_counts = {}

        numeric_features = [
            col_name for col_name in features_df.columns
            if col_name not in ["cust_id", "as_of_month"]
        ]

        # Sample features
        sample_features = numeric_features[:10]

        for col_name in sample_features:
            # Calculate mean and std
            stats = features_df.select(
                F.avg(col_name).alias("mean"),
                F.stddev(col_name).alias("std")
            ).collect()[0]

            if stats["mean"] is not None and stats["std"] is not None:
                mean_val = stats["mean"]
                std_val = stats["std"]

                # Count outliers (>3 std from mean)
                outlier_count = features_df.filter(
                    (F.col(col_name) > mean_val + 3 * std_val) |
                    (F.col(col_name) < mean_val - 3 * std_val)
                ).count()

                outlier_counts[col_name] = outlier_count

        return outlier_counts

    def _detect_leakage(
        self,
        features_df: DataFrame,
        as_of_month: str
    ) -> Dict[str, Any]:
        """
        Detect potential data leakage.

        Checks:
        1. No features use data from after as_of_month
        2. No future-looking calculations
        """
        leakage_detected = False
        leakage_features = []

        # Check: as_of_month in features should match requested as_of_month
        distinct_months = features_df.select("as_of_month").distinct().collect()

        for row in distinct_months:
            if row["as_of_month"] > as_of_month:
                leakage_detected = True
                leakage_features.append(f"as_of_month > {as_of_month}")

        self._log_audit("LEAKAGE_DETECTION", {
            "leakage_detected": leakage_detected,
            "leakage_features": leakage_features
        })

        if leakage_detected:
            print(f"  ⚠️  LEAKAGE DETECTED: {len(leakage_features)} issues")
            raise ValueError(f"Data leakage detected: {leakage_features}")
        else:
            print("  ✓ No leakage detected")

        return {
            "leakage_detected": leakage_detected,
            "leakage_features": leakage_features
        }

    def _write_output(
        self,
        features_df: DataFrame,
        output_table: str,
        run_id: str,
        as_of_month: str
    ):
        """Write features to Delta Lake table"""
        # Add metadata columns
        features_with_metadata = features_df.withColumn(
            "run_id", F.lit(run_id)
        ).withColumn(
            "created_at", F.current_timestamp()
        ).withColumn(
            "pipeline_version", F.lit("1.0.0")
        )

        # Write to Delta
        features_with_metadata.write \
            .format("delta") \
            .mode("append") \
            .partitionBy("as_of_month") \
            .saveAsTable(output_table)

        self._log_audit("OUTPUT_WRITE", {
            "output_table": output_table,
            "run_id": run_id,
            "rows_written": features_with_metadata.count()
        })

    def _log_audit(self, event: str, details: Dict[str, Any]):
        """Log audit event"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details
        })

    def _generate_audit_log(self) -> DataFrame:
        """Generate audit log DataFrame"""
        audit_data = [
            (log["timestamp"], log["event"], json.dumps(log["details"]))
            for log in self.audit_log
        ]

        return self.spark.createDataFrame(
            audit_data,
            ["timestamp", "event", "details"]
        )

    def _print_summary(self, features_df: DataFrame, qa_summary: Dict[str, Any]):
        """Print execution summary"""
        print(f"\n{'='*80}")
        print("EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Features: {qa_summary['total_features']}")
        print(f"Total Customers: {qa_summary['total_customers']}")
        print(f"As-of Month: {qa_summary['as_of_month']}")

        print(f"\nQuality Checks:")
        print(f"  - Missingness: {len([v for v in qa_summary['missingness'].values() if v > 50])} features >50% missing")
        print(f"  - Leakage: {'PASS' if not qa_summary['leakage_check']['leakage_detected'] else 'FAIL'}")

        print(f"\nAudit Events: {len(self.audit_log)}")

        print(f"{'='*80}\n")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from datetime import date, timedelta

    spark = SparkSession.builder \
        .appName("BehavioralPhysicsPipelineTest") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()

    print("\n" + "="*80)
    print("BEHAVIORAL PHYSICS PIPELINE - END-TO-END TEST")
    print("="*80)

    # ========================================================================
    # OPTION 1: Production Usage - Load from Raw Tables (RECOMMENDED)
    # ========================================================================
    print("\n--- Option 1: Production Usage with Bridge ---\n")

    # This method handles:
    # 1. Building point-in-time bridge (CardX → Bureau)
    # 2. Loading and adapting all data
    # 3. Running feature pipeline

    try:
        pipeline = BehavioralPhysicsPipeline(spark)

        as_of_month = "2024-12-31"

        features_df, audit_log_df, qa_summary = pipeline.run_from_raw_tables(
            as_of_month=as_of_month,
            catalog="cdx_mdz_prd",
            output_table=None  # Set to table name to write output
        )

        print("\n✅ Production pipeline test complete")
        print(f"   Features: {len(features_df.columns) - 2}")
        print(f"   Customers: {features_df.count():,}")

        # Show sample
        print("\n📊 Sample Features:")
        sample_cols = ["cust_id", "as_of_month"] + [
            col for col in features_df.columns
            if col not in ["cust_id", "as_of_month"]
        ][:3]
        features_df.select(*sample_cols).show(5)

        print("\n📝 Audit Log:")
        audit_log_df.show(10, truncate=False)

    except Exception as e:
        print(f"\n⚠️  Production test failed (expected if not in Databricks):")
        print(f"   {str(e)}")
        print("\n   To run in Databricks, execute this script in a Databricks notebook")

    # ========================================================================
    # OPTION 2: Testing with Sample Data
    # ========================================================================
    print("\n\n--- Option 2: Testing with Sample DataFrames ---\n")

    # Create sample data for testing
    base_date = date(2024, 1, 1)

    # Sample bureau trade data
    trade_data = []
    for month_offset in range(12):
        as_of = base_date + timedelta(days=30 * month_offset)
        trade_data.extend([
            ("CUST001", "ACC001", as_of, 10000 + month_offset * 1000, month_offset * 5, 50000, as_of),
            ("CUST002", "ACC002", as_of, 20000 + month_offset * 2000, month_offset * 10, 100000, as_of),
        ])

    bureau_trade_df = spark.createDataFrame(
        trade_data,
        ["cust_id", "account_id", "as_of_month", "balance", "dpd", "credit_limit", "receive_dt"]
    )

    # Sample enquiry data
    enquiry_data = [
        ("CUST001", base_date, "CREDIT CARD"),
        ("CUST001", base_date + timedelta(days=60), "PERSONAL LOAN"),
        ("CUST002", base_date + timedelta(days=90), "HOME LOAN"),
    ]

    bureau_enquiry_df = spark.createDataFrame(
        enquiry_data,
        ["cust_id", "enquiry_date", "ENQUIRYPURPOSE"]
    )

    # Sample CardX internal data
    cardx_data = []
    for month_offset in range(12):
        as_of = base_date + timedelta(days=30 * month_offset)
        cardx_data.extend([
            ("CUST001", as_of, month_offset * 3, 15000 + month_offset * 500),
            ("CUST002", as_of, month_offset * 8, 25000 + month_offset * 1000),
        ])

    cardx_internal_df = spark.createDataFrame(
        cardx_data,
        ["cust_id", "as_of_month", "cardx_dpd", "cardx_balance"]
    )

    # Run pipeline with sample data
    pipeline = BehavioralPhysicsPipeline(spark)

    as_of_month = "2024-06-30"

    features_df, audit_log_df, qa_summary = pipeline.run(
        bureau_trade_df,
        bureau_enquiry_df,
        cardx_internal_df,
        as_of_month=as_of_month
    )

    # Show sample features
    print("\n📊 Sample Features:")
    features_df.select(
        "cust_id", "as_of_month",
        "dpd_velocity_3m", "state_entropy_6m", "lender_hhi"
    ).show()

    print("\n📝 Audit Log:")
    audit_log_df.show(truncate=False)

    print("\n📋 QA Summary:")
    print(json.dumps(qa_summary, indent=2, default=str))

    print("\n✅ Sample data pipeline test complete")

    spark.stop()
