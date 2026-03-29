"""
Fraud Detection Example

Demonstrates the complete fraud detection pipeline with synthetic data.
This example shows training, real-time scoring, fraud ring detection,
and risk propagation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Note: Uncomment when dependencies are installed
# from src.use_cases.fraud_detection import (
#     FraudDetectionPipeline,
#     FraudClassifier,
#     EnsembleAnomalyDetector,
#     Node2VecEmbeddings,
#     PageRankRiskPropagation
# )


def generate_synthetic_data(n_samples: int = 10000, fraud_rate: float = 0.001):
    """
    Generate synthetic transaction data for demonstration.
    
    Args:
        n_samples: Number of transactions
        fraud_rate: Proportion of fraudulent transactions
        
    Returns:
        DataFrame with synthetic transactions
    """
    print(f"Generating {n_samples} synthetic transactions...")
    
    # Generate account IDs (some accounts will be fraudulent)
    n_accounts = n_samples // 10
    account_ids = [f'acc_{i}' for i in range(n_accounts)]
    
    # Create transactions
    data = {
        'transaction_id': [f'txn_{i}' for i in range(n_samples)],
        'account_id': np.random.choice(account_ids, n_samples),
        'merchant_id': np.random.choice([f'merch_{i}' for i in range(100)], n_samples),
        'device_id': np.random.choice([f'device_{i}' for i in range(500)], n_samples),
        'ip_address': [f'192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}' 
                       for _ in range(n_samples)],
        'amount': np.random.lognormal(mean=3, sigma=1, size=n_samples),
        'timestamp': [
            datetime.now() - timedelta(days=np.random.randint(0, 30), 
                                      hours=np.random.randint(0, 24))
            for _ in range(n_samples)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels
    # Fraudulent transactions have:
    # - Higher amounts
    # - Concentrated in certain accounts
    # - Higher velocity
    
    n_fraud = int(n_samples * fraud_rate)
    fraud_accounts = np.random.choice(account_ids, size=50, replace=False)
    
    df['is_fraud'] = 0
    
    # Mark some transactions from fraud accounts as fraud
    fraud_mask = df['account_id'].isin(fraud_accounts)
    fraud_candidates = df[fraud_mask].index
    
    if len(fraud_candidates) > n_fraud:
        fraud_indices = np.random.choice(fraud_candidates, size=n_fraud, replace=False)
        df.loc[fraud_indices, 'is_fraud'] = 1
    else:
        df.loc[fraud_candidates, 'is_fraud'] = 1
    
    # Make fraud transactions more extreme
    df.loc[df['is_fraud'] == 1, 'amount'] *= 2.5
    
    print(f"Generated {len(df)} transactions ({df['is_fraud'].sum()} fraudulent, "
          f"{df['is_fraud'].mean():.4%} fraud rate)")
    
    return df


def example_basic_pipeline():
    """Example: Basic fraud detection pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Fraud Detection Pipeline")
    print("="*70 + "\n")
    
    # Generate data
    transactions = generate_synthetic_data(n_samples=5000, fraud_rate=0.01)
    
    print("\nThis example demonstrates:")
    print("1. Training the fraud detection pipeline")
    print("2. Real-time transaction scoring")
    print("3. Batch scoring")
    print("4. Model monitoring")
    
    print("\nCode:")
    print("""
    from src.use_cases.fraud_detection import FraudDetectionPipeline
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    # Train pipeline
    pipeline.fit(transactions)
    
    # Real-time scoring
    result = pipeline.predict_single_transaction(transactions.iloc[0])
    print(f"Fraud Score: {result['fraud_score']:.3f}")
    print(f"Is Fraud: {result['is_fraud']}")
    print(f"Latency: {result['latency_ms']:.1f}ms")
    
    # Batch scoring
    scores = pipeline.predict_batch(transactions.head(100))
    print(f"Average fraud score: {scores['fraud_score'].mean():.3f}")
    
    # Monitoring
    metrics = pipeline.get_monitoring_metrics()
    print(f"Average latency: {metrics['avg_latency_ms']:.1f}ms")
    """)


def example_fraud_rings():
    """Example: Fraud ring detection."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Fraud Ring Detection")
    print("="*70 + "\n")
    
    print("This example demonstrates:")
    print("1. Detecting organized fraud rings")
    print("2. Analyzing ring characteristics")
    print("3. Risk propagation through networks")
    
    print("\nCode:")
    print("""
    # Detect fraud rings
    rings = pipeline.detect_fraud_rings()
    
    print(f"Detected {len(rings)} fraud rings")
    
    # Analyze top rings
    for i, ring in enumerate(rings[:5], 1):
        print(f"Ring {i}:")
        print(f"  Size: {ring['size']} accounts")
        print(f"  Average Risk: {ring['avg_risk']:.3f}")
        print(f"  Density: {ring['density']:.3f}")
        print(f"  Known Fraud: {ring['fraud_count']}")
    """)


def example_risk_propagation():
    """Example: Risk propagation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Risk Propagation")
    print("="*70 + "\n")
    
    print("This example demonstrates:")
    print("1. Propagating risk from known fraud accounts")
    print("2. Identifying high-risk accounts")
    print("3. Network-based fraud detection")
    
    print("\nCode:")
    print("""
    # Known fraud accounts
    known_fraud = {'acc_123', 'acc_456', 'acc_789'}
    
    # Propagate risk using PageRank
    risk_scores = pipeline.propagate_risk(
        known_fraud, 
        method='pagerank'
    )
    
    # Get top risky accounts
    top_risk = sorted(
        risk_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:20]
    
    print("Top 20 highest risk accounts:")
    for account, risk in top_risk:
        print(f"{account}: {risk:.4f}")
    """)


def example_anomaly_detection():
    """Example: Anomaly detection."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Unsupervised Anomaly Detection")
    print("="*70 + "\n")
    
    print("This example demonstrates:")
    print("1. Using ensemble anomaly detection")
    print("2. Detecting novel fraud patterns")
    print("3. Combining supervised and unsupervised methods")
    
    print("\nCode:")
    print("""
    from src.use_cases.fraud_detection import EnsembleAnomalyDetector
    
    # Create ensemble detector
    detector = EnsembleAnomalyDetector(
        use_isolation_forest=True,
        use_lof=True,
        use_autoencoder=True,
        aggregation='weighted'
    )
    
    # Train on normal transactions
    normal_transactions = transactions[transactions['is_fraud'] == 0]
    detector.fit(X_normal)
    
    # Detect anomalies
    anomaly_scores = detector.predict_scores(X_test)
    anomalies = detector.predict(X_test, contamination=0.01)
    
    print(f"Detected {anomalies.sum()} anomalies")
    print(f"Average anomaly score: {anomaly_scores.mean():.3f}")
    """)


def example_ab_testing():
    """Example: A/B testing."""
    print("\n" + "="*70)
    print("EXAMPLE 5: A/B Testing Framework")
    print("="*70 + "\n")
    
    print("This example demonstrates:")
    print("1. Setting up A/B tests for model variants")
    print("2. Traffic splitting")
    print("3. Statistical analysis")
    
    print("\nCode:")
    print("""
    from src.use_cases.fraud_detection import ABTestingFramework
    
    # Create A/B test
    ab_test = ABTestingFramework()
    
    # Add variants
    ab_test.add_variant('control', pipeline_v1, traffic_proportion=0.5)
    ab_test.add_variant('treatment', pipeline_v2, traffic_proportion=0.5)
    
    # Route traffic
    for transaction in new_transactions:
        user_id = transaction['account_id']
        variant_name, pipeline = ab_test.get_variant(user_id)
        
        # Score transaction
        result = pipeline.predict_single_transaction(transaction)
        
        # Log result (with ground truth if available)
        ab_test.log_result(variant_name, result, ground_truth=None)
    
    # Analyze results
    analysis = ab_test.analyze_results()
    
    print("A/B Test Results:")
    for variant, metrics in analysis.items():
        if variant != 'comparison':
            print(f"{variant}:")
            print(f"  Avg Score: {metrics['avg_score']:.3f}")
            print(f"  Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    
    if 'comparison' in analysis:
        comp = analysis['comparison']
        print(f"Statistical Significance: {comp['significant']}")
        print(f"P-value: {comp['p_value']:.4f}")
    """)


def example_custom_features():
    """Example: Custom feature engineering."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Feature Engineering")
    print("="*70 + "\n")
    
    print("This example demonstrates:")
    print("1. Creating velocity features")
    print("2. Computing anomaly scores")
    print("3. Integrating graph features")
    
    print("\nCode:")
    print("""
    from src.use_cases.fraud_detection import FraudFeatureEngineer
    
    engineer = FraudFeatureEngineer()
    
    # Velocity features (transaction frequency/volume)
    velocity_features = engineer.create_velocity_features(
        transactions,
        account_col='account_id',
        timestamp_col='timestamp',
        amount_col='amount'
    )
    
    print(f"Created {velocity_features.shape[1]} velocity features")
    print(velocity_features.columns.tolist())
    
    # Anomaly scores
    anomaly_features = engineer.create_anomaly_scores(
        transactions,
        amount_col='amount',
        merchant_col='merchant_id'
    )
    
    print(f"Created {anomaly_features.shape[1]} anomaly features")
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("FRAUD DETECTION EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate the fraud detection use case.")
    print("To run the actual code, install dependencies:")
    print("  pip install numpy pandas scikit-learn networkx lightgbm")
    print("  pip install node2vec torch  # optional")
    
    # Run examples
    example_basic_pipeline()
    example_fraud_rings()
    example_risk_propagation()
    example_anomaly_detection()
    example_ab_testing()
    example_custom_features()
    
    print("\n" + "="*70)
    print("For more information, see:")
    print("  src/use_cases/fraud_detection/README.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
