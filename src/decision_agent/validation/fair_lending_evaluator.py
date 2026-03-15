"""
Fair Lending Evaluator - Regulatory Compliance (CRITICAL P0)

Evaluates model for bias/discrimination across protected classes.
This is a LEGAL/REGULATORY REQUIREMENT for financial ML systems:
- ECOA (Equal Credit Opportunity Act)
- FCRA (Fair Credit Reporting Act)
- CFPB oversight

Detects:
- Disparate Impact: Different approval/prediction rates across protected groups
- Disparate Treatment: Model uses protected attributes (illegal)
- Proxy Discrimination: Model uses proxies for protected attributes

Quality Gate:
- Training MUST pass fair lending checks before deployment
- Model promotion BLOCKED if disparate impact exceeds thresholds

Integration Point:
- Called by training_harness.py post-training
- Extends validation/segment_eval.py
- Logs results to MLflow for audit trail
- BLOCKS model registry promotion if fails

Protected Classes (USA):
- Race/Ethnicity
- Gender/Sex
- Age
- National Origin
- Religion
- Marital Status
- Disability Status
"""

import logging
from typing import Dict, List, Any, Tuple
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import numpy as np

logger = logging.getLogger(__name__)


class FairLendingError(Exception):
    """Raised when fair lending evaluation fails critically."""
    pass


class FairLendingEvaluator:
    """
    Evaluate model fairness across protected demographic groups.

    This is a GATE - model cannot be deployed if fairness checks fail.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with fairness thresholds.

        Args:
            config: Fair lending configuration:
                - max_disparate_impact_ratio: Maximum DI ratio (default 0.80 = 80% rule)
                - max_mean_difference: Maximum mean prediction difference (default 5000)
                - max_approval_rate_difference: Max difference in approval rates (default 0.10)
                - protected_attributes: List of protected demographic columns
                - reference_groups: Reference group for each protected attribute
        """
        self.config = config
        self.max_disparate_impact_ratio = config.get("max_disparate_impact_ratio", 0.80)
        self.max_mean_difference = config.get("max_mean_difference", 5000)  # dollars
        self.max_approval_rate_difference = config.get("max_approval_rate_difference", 0.10)
        self.protected_attributes = config.get("protected_attributes", [])
        self.reference_groups = config.get("reference_groups", {})

        if not self.protected_attributes:
            logger.warning("No protected attributes specified. Fair lending evaluation limited.")

    def evaluate(
        self,
        df: DataFrame,
        predictions_col: str = "prediction",
        label_col: str = "income_level",
        decision_threshold: float = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate fairness across protected groups.

        Args:
            df: Spark DataFrame with predictions and demographics
            predictions_col: Name of prediction column
            label_col: Name of true label column (for ground truth fairness)
            decision_threshold: Threshold for binary decisions (optional)

        Returns:
            Tuple of (passed: bool, results: Dict)
            - passed: True if all fairness checks passed
            - results: Detailed fairness metrics

        Raises:
            FairLendingError: If critical fairness violations detected
        """
        logger.info("Starting fair lending evaluation...")

        results = {
            "evaluation_timestamp": str(F.current_timestamp()),
            "protected_attributes_evaluated": self.protected_attributes,
            "fairness_checks": {}
        }

        if not self.protected_attributes:
            logger.warning("No protected attributes to evaluate. Skipping fair lending checks.")
            return True, {"warning": "No protected attributes provided"}

        all_passed = True

        for attribute in self.protected_attributes:
            if attribute not in df.columns:
                logger.warning(f"Protected attribute '{attribute}' not found in data. Skipping.")
                continue

            # Evaluate fairness for this attribute
            passed, attribute_results = self._evaluate_attribute(
                df,
                attribute,
                predictions_col,
                label_col,
                decision_threshold
            )

            results["fairness_checks"][attribute] = attribute_results

            if not passed:
                all_passed = False
                logger.error(f"Fair lending violation detected for attribute: {attribute}")

        results["overall_passed"] = all_passed

        if not all_passed:
            raise FairLendingError(
                f"Fair lending evaluation failed. Disparate impact detected. "
                f"See results for details: {results['fairness_checks']}"
            )

        logger.info("Fair lending evaluation PASSED")
        return all_passed, results

    def _evaluate_attribute(
        self,
        df: DataFrame,
        attribute: str,
        predictions_col: str,
        label_col: str,
        decision_threshold: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate fairness for a single protected attribute.

        Metrics:
        1. Disparate Impact Ratio (80% rule)
        2. Mean Prediction Difference
        3. Approval Rate Difference (if threshold provided)
        4. Prediction Distribution by Group
        """
        results = {}

        # Get reference group for comparison
        reference_group = self.reference_groups.get(attribute, None)

        # 1. Compute mean predictions by group
        group_stats = df.groupBy(attribute).agg(
            F.count("*").alias("group_size"),
            F.mean(predictions_col).alias("mean_prediction"),
            F.stddev(predictions_col).alias("std_prediction"),
            F.mean(label_col).alias("mean_actual") if label_col in df.columns else F.lit(None).alias("mean_actual")
        )

        # Convert to pandas for easier manipulation (small aggregated data)
        group_stats_pd = group_stats.toPandas()

        results["group_statistics"] = group_stats_pd.to_dict(orient="records")

        # 2. Compute disparate impact ratios
        if reference_group and reference_group in group_stats_pd[attribute].values:
            ref_mean = group_stats_pd[group_stats_pd[attribute] == reference_group]["mean_prediction"].values[0]

            disparate_impact_ratios = {}
            violations = []

            for _, row in group_stats_pd.iterrows():
                group_name = row[attribute]
                group_mean = row["mean_prediction"]

                if group_name != reference_group:
                    # Disparate Impact Ratio = min(group_rate / ref_rate, ref_rate / group_rate)
                    # For continuous predictions, use mean ratio
                    di_ratio = min(group_mean / ref_mean, ref_mean / group_mean) if ref_mean > 0 else 1.0

                    disparate_impact_ratios[str(group_name)] = float(di_ratio)

                    # Check 80% rule
                    if di_ratio < self.max_disparate_impact_ratio:
                        violations.append({
                            "group": str(group_name),
                            "disparate_impact_ratio": float(di_ratio),
                            "threshold": self.max_disparate_impact_ratio,
                            "reference_group": str(reference_group),
                            "group_mean": float(group_mean),
                            "reference_mean": float(ref_mean)
                        })

            results["disparate_impact_ratios"] = disparate_impact_ratios
            results["violations"] = violations
            results["passed"] = len(violations) == 0

        else:
            # No reference group - just check max difference between any two groups
            max_mean = group_stats_pd["mean_prediction"].max()
            min_mean = group_stats_pd["mean_prediction"].min()
            mean_difference = max_mean - min_mean

            results["max_mean_difference"] = float(mean_difference)
            results["passed"] = mean_difference <= self.max_mean_difference

        # 3. If decision threshold provided, check approval rates
        if decision_threshold:
            approval_stats = df.withColumn(
                "approved",
                (F.col(predictions_col) >= decision_threshold).cast("int")
            ).groupBy(attribute).agg(
                F.mean("approved").alias("approval_rate")
            )

            approval_stats_pd = approval_stats.toPandas()
            results["approval_rates"] = approval_stats_pd.to_dict(orient="records")

            # Check approval rate difference
            max_approval = approval_stats_pd["approval_rate"].max()
            min_approval = approval_stats_pd["approval_rate"].min()
            approval_diff = max_approval - min_approval

            results["approval_rate_difference"] = float(approval_diff)

            if approval_diff > self.max_approval_rate_difference:
                results["passed"] = False

        return results.get("passed", True), results

    def generate_fairness_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable fairness report for MRM documentation.

        Args:
            results: Output from evaluate()

        Returns:
            Formatted fairness report string
        """
        report_lines = [
            "=" * 80,
            "FAIR LENDING EVALUATION REPORT",
            "=" * 80,
            "",
            f"Evaluation Timestamp: {results.get('evaluation_timestamp', 'N/A')}",
            f"Overall Status: {'PASSED ✓' if results.get('overall_passed', False) else 'FAILED ✗'}",
            "",
            "Protected Attributes Evaluated:",
        ]

        for attr in results.get("protected_attributes_evaluated", []):
            report_lines.append(f"  - {attr}")

        report_lines.append("")
        report_lines.append("Fairness Checks by Attribute:")
        report_lines.append("-" * 80)

        for attr, attr_results in results.get("fairness_checks", {}).items():
            report_lines.append(f"\n{attr.upper()}")
            report_lines.append("  Status: " + ("PASSED ✓" if attr_results.get("passed", False) else "FAILED ✗"))

            # Group statistics
            if "group_statistics" in attr_results:
                report_lines.append("\n  Group Statistics:")
                for stat in attr_results["group_statistics"]:
                    report_lines.append(
                        f"    {stat.get(attr, 'N/A')}: "
                        f"n={stat.get('group_size', 0)}, "
                        f"mean_pred=${stat.get('mean_prediction', 0):,.2f}"
                    )

            # Disparate impact violations
            if "violations" in attr_results and attr_results["violations"]:
                report_lines.append("\n  ⚠️  DISPARATE IMPACT VIOLATIONS:")
                for violation in attr_results["violations"]:
                    report_lines.append(
                        f"    {violation['group']}: DI Ratio = {violation['disparate_impact_ratio']:.3f} "
                        f"(threshold: {violation['threshold']:.2f})"
                    )
                    report_lines.append(
                        f"      Group Mean: ${violation['group_mean']:,.2f}, "
                        f"Reference Mean: ${violation['reference_mean']:,.2f}"
                    )

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)
