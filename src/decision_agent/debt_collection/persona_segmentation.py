"""
Debt Collection Persona & Segmentation Module

Creates debtor personas combining rule-based business knowledge
and ML-based clustering for optimal collection strategies.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import DataFrame, functions as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mlflow


class DebtorPersonaSegmentation:
    """
    Create debtor personas using hierarchical segmentation:
    1. Rule-based segments (business knowledge)
    2. ML-based clustering within segments
    3. Combined persona assignment
    """

    def __init__(self, config: Dict):
        self.config = config
        self.rule_segments = config['segmentation']['rule_based_segments']
        self.ml_config = config['segmentation']['ml_segments']

    def create_personas(
        self,
        accounts_df: DataFrame,
        features_df: DataFrame
    ) -> DataFrame:
        """
        Create final personas combining rule-based and ML segments.

        Args:
            accounts_df: Account data (balance, days past due, etc.)
            features_df: Engineered features (behavioral, financial)

        Returns:
            DataFrame with persona assignments and strategies
        """
        # Step 1: Rule-based segmentation
        segmented_df = self._apply_rule_based_segments(accounts_df, features_df)

        # Step 2: ML clustering within each rule segment
        final_df = self._apply_ml_clustering(segmented_df)

        # Step 3: Assign strategies
        strategy_df = self._assign_strategies(final_df)

        return strategy_df

    def _apply_rule_based_segments(
        self,
        accounts_df: DataFrame,
        features_df: DataFrame
    ) -> DataFrame:
        """Apply business rule-based segmentation"""

        # Join accounts and features
        combined_df = accounts_df.join(features_df, on="account_id", how="inner")

        # Initialize segment column
        combined_df = combined_df.withColumn("rule_segment", F.lit("unclassified"))

        # Apply each rule-based segment
        for segment in self.rule_segments:
            segment_name = segment['name']
            rules = segment['rules']

            # Build condition expression
            condition = self._build_condition(rules)

            # Update segment for matching rows
            combined_df = combined_df.withColumn(
                "rule_segment",
                F.when(
                    condition & (F.col("rule_segment") == "unclassified"),
                    F.lit(segment_name)
                ).otherwise(F.col("rule_segment"))
            )

        return combined_df

    def _build_condition(self, rules: List[str]):
        """Build Spark SQL condition from rule strings"""
        conditions = []

        for rule in rules:
            # Parse rule string and convert to Spark expression
            # Example: "current_balance > 5000" -> F.col("current_balance") > 5000

            if ">" in rule:
                col_name, value = rule.split(">")
                col_name = col_name.strip()
                value = float(value.strip())
                conditions.append(F.col(col_name) > value)

            elif "<" in rule:
                col_name, value = rule.split("<")
                col_name = col_name.strip()
                value = float(value.strip())
                conditions.append(F.col(col_name) < value)

            elif "BETWEEN" in rule:
                parts = rule.replace("BETWEEN", "").replace("AND", ",").split(",")
                col_name = parts[0].strip()
                lower = float(parts[1].strip())
                upper = float(parts[2].strip())
                conditions.append(F.col(col_name).between(lower, upper))

            elif "=" in rule:
                col_name, value = rule.split("=")
                col_name = col_name.strip()
                value = value.strip()
                if value.lower() == "true":
                    conditions.append(F.col(col_name) == True)
                elif value.lower() == "false":
                    conditions.append(F.col(col_name) == False)
                else:
                    conditions.append(F.col(col_name) == value)

        # Combine all conditions with AND
        if len(conditions) == 0:
            return F.lit(False)

        result = conditions[0]
        for cond in conditions[1:]:
            result = result & cond

        return result

    def _apply_ml_clustering(self, df: DataFrame) -> DataFrame:
        """Apply ML clustering within each rule segment"""

        # Convert to pandas for sklearn
        pdf = df.toPandas()

        # Get feature columns for clustering
        feature_cols = self.ml_config['features']

        # Cluster within each rule segment
        pdf['ml_segment'] = 'unknown'

        for segment_name in pdf['rule_segment'].unique():
            segment_mask = pdf['rule_segment'] == segment_name

            if segment_mask.sum() < self.ml_config['n_clusters']:
                # Too few samples, skip clustering
                pdf.loc[segment_mask, 'ml_segment'] = 'singleton'
                continue

            # Get features for this segment
            X = pdf.loc[segment_mask, feature_cols].fillna(0)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # K-means clustering
            n_clusters = min(self.ml_config['n_clusters'], segment_mask.sum() // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            # Assign cluster IDs
            pdf.loc[segment_mask, 'ml_segment'] = [
                f"{segment_name}_cluster_{c}" for c in clusters
            ]

        # Convert back to Spark DataFrame
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        result_df = spark.createDataFrame(pdf)

        return result_df

    def _assign_strategies(self, df: DataFrame) -> DataFrame:
        """Assign collection strategies to each persona"""

        # Create final persona ID
        df = df.withColumn(
            "persona",
            F.concat(F.col("rule_segment"), F.lit("_"), F.col("ml_segment"))
        )

        # Assign strategies based on rule segment
        strategy_map = {
            segment['name']: segment['strategy']
            for segment in self.rule_segments
        }

        # Create mapping expression
        strategy_expr = F.create_map(
            [F.lit(x) for pair in strategy_map.items() for x in pair]
        )

        df = df.withColumn(
            "strategy",
            strategy_expr[F.col("rule_segment")]
        )

        return df

    def analyze_personas(self, df: DataFrame) -> pd.DataFrame:
        """Analyze persona characteristics for reporting"""

        # Convert to pandas for analysis
        pdf = df.toPandas()

        # Compute statistics per persona
        persona_stats = pdf.groupby('persona').agg({
            'account_id': 'count',
            'current_balance': ['mean', 'sum'],
            'days_past_due': 'mean',
            'contact_response_rate': 'mean',
            'payment_capacity_score': 'mean',
            'ltp_score': 'mean'
        }).round(2)

        persona_stats.columns = [
            'count', 'avg_balance', 'total_balance',
            'avg_days_past_due', 'avg_response_rate',
            'avg_payment_capacity', 'avg_likelihood_to_pay'
        ]

        # Add percentage
        persona_stats['pct_of_portfolio'] = (
            persona_stats['count'] / persona_stats['count'].sum() * 100
        ).round(1)

        return persona_stats.reset_index()


class PersonaStrategyRecommender:
    """
    Recommend collection strategies for each persona based on:
    - Historical performance
    - Cost-effectiveness
    - Regulatory constraints
    - Expected recovery
    """

    def __init__(self, config: Dict):
        self.config = config

    def recommend_strategy(
        self,
        persona: str,
        account_features: Dict,
        historical_performance: pd.DataFrame
    ) -> Dict:
        """
        Recommend optimal collection strategy for a persona.

        Returns:
            {
                'channel': 'sms',
                'offer_type': 'settlement',
                'discount_pct': 40,
                'expected_recovery': 2500,
                'expected_cost': 50,
                'expected_roi': 50.0
            }
        """

        # Get historical performance for this persona
        persona_history = historical_performance[
            historical_performance['persona'] == persona
        ]

        if len(persona_history) == 0:
            # No history, use defaults
            return self._default_strategy(persona, account_features)

        # Find best performing strategy
        best_strategy = persona_history.sort_values(
            'recovery_rate', ascending=False
        ).iloc[0]

        recommendation = {
            'channel': best_strategy['best_channel'],
            'offer_type': best_strategy['best_offer_type'],
            'discount_pct': best_strategy['avg_discount'],
            'expected_recovery': account_features['current_balance'] * best_strategy['recovery_rate'],
            'expected_cost': self._estimate_cost(best_strategy['best_channel']),
            'timing': best_strategy['best_time_of_day']
        }

        recommendation['expected_roi'] = (
            (recommendation['expected_recovery'] - recommendation['expected_cost']) /
            recommendation['expected_cost'] * 100
        )

        return recommendation

    def _default_strategy(self, persona: str, account_features: Dict) -> Dict:
        """Default strategy when no historical data"""

        # Simple rules
        if "high_value" in persona:
            channel = "call"
            offer_type = "settlement"
            discount = 50
        elif "low_value" in persona:
            channel = "email"
            offer_type = "payment_reminder"
            discount = 0
        else:
            channel = "sms"
            offer_type = "payment_plan"
            discount = 20

        return {
            'channel': channel,
            'offer_type': offer_type,
            'discount_pct': discount,
            'expected_recovery': account_features['current_balance'] * 0.15,
            'expected_cost': self._estimate_cost(channel),
            'expected_roi': 200,
            'timing': '10:00 AM'
        }

    def _estimate_cost(self, channel: str) -> float:
        """Estimate cost per contact by channel"""
        cost_map = self.config['constraints']['cost']['channel_costs']
        return cost_map.get(channel, 1.0)


# Example usage
if __name__ == "__main__":
    import yaml
    from pyspark.sql import SparkSession

    # Load config
    with open("conf/use_cases/debt_collection_optimization.yaml") as f:
        config = yaml.safe_load(f)

    # Create Spark session
    spark = SparkSession.builder.appName("DebtorPersonas").getOrCreate()

    # Load data (example)
    accounts_df = spark.table("debt_collection.accounts")
    features_df = spark.table("debt_collection.features")

    # Create personas
    segmenter = DebtorPersonaSegmentation(config)
    personas_df = segmenter.create_personas(accounts_df, features_df)

    # Analyze
    stats = segmenter.analyze_personas(personas_df)
    print("\nPersona Analysis:")
    print(stats)

    # Log to MLflow
    with mlflow.start_run(run_name="persona_segmentation"):
        mlflow.log_param("n_personas", stats.shape[0])
        mlflow.log_metric("avg_balance_per_persona", stats['avg_balance'].mean())

        # Log stats table
        stats.to_csv("persona_stats.csv", index=False)
        mlflow.log_artifact("persona_stats.csv")

    print("\n✓ Persona segmentation complete")
