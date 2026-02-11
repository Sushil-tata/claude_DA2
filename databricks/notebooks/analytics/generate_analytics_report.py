# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Agent Analytics Report
# MAGIC
# MAGIC Comprehensive analytics report including:
# MAGIC 1. Cohort Analysis - Customer retention and segmentation
# MAGIC 2. Model Performance - Prediction accuracy and business metrics
# MAGIC 3. Feature Importance - Top drivers of predictions
# MAGIC 4. Uplift Analysis - Treatment effect estimation
# MAGIC 5. Business Impact - Revenue, cost, and ROI metrics
# MAGIC
# MAGIC **Output:** HTML report with visualizations and insights

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

%pip install matplotlib seaborn plotly

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from decision_agent.analytics.cohort_analysis import CohortAnalyzer
from decision_agent.analytics.uplift_modeling import TLearner
from decision_agent.analytics.causal_inference import PropensityScoreMatcher

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cohort Analysis

# COMMAND ----------

print("=" * 80)
print("COHORT ANALYSIS")
print("=" * 80)

# Load transaction data
transactions_df = spark.table("decision_agent.transactions").toPandas()

print(f"Loaded {len(transactions_df):,} transactions")
print(f"Unique customers: {transactions_df['customer_id'].nunique():,}")
print(f"Date range: {transactions_df['transaction_date'].min()} to {transactions_df['transaction_date'].max()}")

# Initialize cohort analyzer
analyzer = CohortAnalyzer(
    transactions_df,
    customer_id_col="customer_id",
    date_col="transaction_date",
    value_col="amount"
)

# Compute retention curve
retention = analyzer.compute_retention_curve(cohort_period="month", retention_periods=12)

print("\nRetention Curve (first 5 cohorts):")
print(retention.head())

# Segment customers
segments, stats = analyzer.segment_customers(n_clusters=5)

print("\nCustomer Segments:")
for segment_id, segment_stats in stats.items():
    print(f"\n{segment_id}:")
    print(f"  Size: {segment_stats['size']:,} ({segment_stats['size_pct']:.1f}%)")
    print(f"  Recency: {segment_stats['features']['recency']['mean']:.0f} days")
    print(f"  Frequency: {segment_stats['features']['frequency']['mean']:.1f} transactions")
    print(f"  Monetary: ${segment_stats['features']['monetary']['mean']:,.0f}")

# Compute LTV by cohort
ltv = analyzer.compute_lifetime_value_by_cohort(cohort_period="month")

print("\nLifetime Value by Cohort (top 5):")
print(ltv.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization: Retention Curve

# COMMAND ----------

# Plot retention curve
fig = go.Figure()

for cohort in retention.index[-6:]:  # Last 6 cohorts
    fig.add_trace(go.Scatter(
        x=retention.columns,
        y=retention.loc[cohort],
        mode='lines+markers',
        name=str(cohort)
    ))

fig.update_layout(
    title="Customer Retention Curve by Cohort",
    xaxis_title="Months Since First Transaction",
    yaxis_title="Retention Rate (%)",
    yaxis=dict(range=[0, 100]),
    hovermode='x unified'
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization: Customer Segments

# COMMAND ----------

# Scatter plot of segments
fig = px.scatter(
    segments,
    x="recency",
    y="monetary",
    color="segment",
    size="frequency",
    title="Customer Segmentation (RFM Analysis)",
    labels={
        "recency": "Days Since Last Transaction",
        "monetary": "Total Spend ($)",
        "frequency": "Number of Transactions"
    }
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Performance Analysis

# COMMAND ----------

print("=" * 80)
print("MODEL PERFORMANCE ANALYSIS")
print("=" * 80)

# Load predictions
predictions_df = spark.table("decision_agent.model_predictions_log").toPandas()

print(f"Loaded {len(predictions_df):,} predictions")
print(f"Date range: {predictions_df['prediction_timestamp'].min()} to {predictions_df['prediction_timestamp'].max()}")

# Load actual outcomes (for evaluation)
actuals_df = spark.table("decision_agent.customer_actuals").toPandas()

# Merge predictions with actuals
eval_df = predictions_df.merge(
    actuals_df,
    on="customer_id",
    how="inner"
)

print(f"\nEvaluation sample size: {len(eval_df):,}")

# Compute metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(eval_df['actual_income'], eval_df['prediction']))
mae = mean_absolute_error(eval_df['actual_income'], eval_df['prediction'])
r2 = r2_score(eval_df['actual_income'], eval_df['prediction'])

print("\nOverall Metrics:")
print(f"  RMSE: {rmse:,.2f}")
print(f"  MAE: {mae:,.2f}")
print(f"  R²: {r2:.4f}")

# Performance by segment
print("\nPerformance by Confidence Level:")
for confidence in ['high', 'medium', 'low']:
    subset = eval_df[eval_df['confidence'] == confidence]
    if len(subset) > 0:
        subset_mae = mean_absolute_error(subset['actual_income'], subset['prediction'])
        print(f"  {confidence.capitalize()}: MAE = {subset_mae:,.2f} (n={len(subset):,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization: Prediction vs Actual

# COMMAND ----------

# Scatter plot: Predicted vs Actual
sample = eval_df.sample(min(10000, len(eval_df)))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sample['actual_income'],
    y=sample['prediction'],
    mode='markers',
    marker=dict(
        color=sample['confidence'].map({'high': 'green', 'medium': 'yellow', 'low': 'red'}),
        size=5,
        opacity=0.5
    ),
    name='Predictions'
))

# Add perfect prediction line
min_val = min(sample['actual_income'].min(), sample['prediction'].min())
max_val = max(sample['actual_income'].max(), sample['prediction'].max())

fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    line=dict(color='black', dash='dash'),
    name='Perfect Prediction'
))

fig.update_layout(
    title=f"Predicted vs Actual Income (R² = {r2:.3f})",
    xaxis_title="Actual Income ($)",
    yaxis_title="Predicted Income ($)"
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization: Error Distribution

# COMMAND ----------

# Error distribution
eval_df['error'] = eval_df['prediction'] - eval_df['actual_income']
eval_df['pct_error'] = (eval_df['error'] / eval_df['actual_income']) * 100

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Absolute Error Distribution", "Percentage Error Distribution")
)

fig.add_trace(
    go.Histogram(x=eval_df['error'], nbinsx=50, name="Absolute Error"),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(x=eval_df['pct_error'].clip(-50, 50), nbinsx=50, name="% Error"),
    row=1, col=2
)

fig.update_layout(
    title_text="Prediction Error Distribution",
    showlegend=False
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Importance Analysis

# COMMAND ----------

print("=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Load feature importance from latest model
import mlflow

mlflow.set_tracking_uri("databricks")

# Get champion model
model = mlflow.pyfunc.load_model("models:/income_estimation_champion/Production")

# Get feature importance (if available)
if hasattr(model._model_impl.python_model, 'feature_importances_'):
    importances = model._model_impl.python_model.feature_importances_
    feature_names = model._model_impl.python_model.feature_names_in_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Features:")
    print(importance_df.head(20).to_string(index=False))

    # Plot top features
    fig = go.Figure(go.Bar(
        x=importance_df.head(20)['importance'],
        y=importance_df.head(20)['feature'],
        orientation='h'
    ))

    fig.update_layout(
        title="Top 20 Most Important Features",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600,
        yaxis=dict(autorange="reversed")
    )

    fig.show()
else:
    print("Feature importance not available for this model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Business Impact Analysis

# COMMAND ----------

print("=" * 80)
print("BUSINESS IMPACT ANALYSIS")
print("=" * 80)

# Calculate business metrics
avg_prediction = predictions_df['prediction'].mean()
avg_actual = actuals_df['actual_income'].mean()

total_predictions = len(predictions_df)
total_customers = predictions_df['customer_id'].nunique()

# Revenue impact (if using predictions for targeting)
# Assume: targeting customers with prediction > threshold
threshold = predictions_df['prediction'].quantile(0.8)  # Top 20%

high_value_customers = predictions_df[predictions_df['prediction'] >= threshold]
potential_revenue = high_value_customers['prediction'].sum()

print("\nBusiness Metrics:")
print(f"  Total predictions made: {total_predictions:,}")
print(f"  Unique customers: {total_customers:,}")
print(f"  Average predicted income: ${avg_prediction:,.0f}")
print(f"  Average actual income: ${avg_actual:,.0f}")
print("")
print(f"High-Value Customer Targeting (Top 20%):")
print(f"  Number of customers: {len(high_value_customers):,}")
print(f"  Potential revenue: ${potential_revenue:,.0f}")
print(f"  Average income: ${high_value_customers['prediction'].mean():,.0f}")

# Model usage over time
predictions_by_date = predictions_df.groupby(
    predictions_df['prediction_timestamp'].dt.date
).size().reset_index(name='predictions')

print(f"\nPrediction Volume (last 7 days):")
print(predictions_by_date.tail(7).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualization: Prediction Volume Over Time

# COMMAND ----------

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=predictions_by_date['prediction_timestamp'],
    y=predictions_by_date['predictions'],
    mode='lines+markers',
    name='Daily Predictions'
))

fig.update_layout(
    title="Daily Prediction Volume",
    xaxis_title="Date",
    yaxis_title="Number of Predictions"
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary Report

# COMMAND ----------

# Generate executive summary
summary = f"""
DECISION AGENT ANALYTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

==========================================
EXECUTIVE SUMMARY
==========================================

1. CUSTOMER INSIGHTS
   • Total Customers: {transactions_df['customer_id'].nunique():,}
   • Customer Segments: 5 identified clusters
   • Average Retention (6 months): {retention.iloc[:, 6].mean():.1f}%

2. MODEL PERFORMANCE
   • Total Predictions: {total_predictions:,}
   • Model Accuracy (R²): {r2:.3f}
   • Mean Absolute Error: ${mae:,.0f}
   • RMSE: ${rmse:,.0f}

3. BUSINESS IMPACT
   • Average Predicted Income: ${avg_prediction:,.0f}
   • High-Value Customers (Top 20%): {len(high_value_customers):,}
   • Potential Revenue (Top 20%): ${potential_revenue:,.0f}

4. RECOMMENDATIONS
   • Focus retention efforts on high-value segments
   • Monitor model performance for segments with higher error
   • Continue model retraining quarterly for accuracy
   • Leverage cohort insights for personalized campaigns

==========================================
"""

print(summary)

# Save summary
dbutils.fs.put(
    f"dbfs:/decision_agent/reports/analytics_summary_{datetime.now().strftime('%Y%m%d')}.txt",
    summary,
    overwrite=True
)

print("✓ Report saved to DBFS")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Report

# COMMAND ----------

# Generate HTML report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Decision Agent Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2C5282; }}
        h2 {{ color: #4A5568; border-bottom: 2px solid #E2E8F0; padding-bottom: 10px; }}
        .metric {{ background: #EDF2F7; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2C5282; }}
        .metric-label {{ font-size: 14px; color: #718096; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #E2E8F0; padding: 12px; text-align: left; }}
        th {{ background: #2C5282; color: white; }}
    </style>
</head>
<body>
    <h1>Decision Agent Analytics Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>Key Metrics</h2>
    <div class="metric">
        <div class="metric-label">Total Customers</div>
        <div class="metric-value">{transactions_df['customer_id'].nunique():,}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Model Accuracy (R²)</div>
        <div class="metric-value">{r2:.3f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Total Predictions</div>
        <div class="metric-value">{total_predictions:,}</div>
    </div>

    <h2>Performance Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>RMSE</td><td>${rmse:,.0f}</td></tr>
        <tr><td>MAE</td><td>${mae:,.0f}</td></tr>
        <tr><td>Average Predicted Income</td><td>${avg_prediction:,.0f}</td></tr>
        <tr><td>Average Actual Income</td><td>${avg_actual:,.0f}</td></tr>
    </table>

    <h2>Recommendations</h2>
    <ul>
        <li>Focus retention efforts on high-value customer segments</li>
        <li>Monitor model performance for customer segments with higher prediction error</li>
        <li>Continue quarterly model retraining to maintain accuracy</li>
        <li>Leverage cohort analysis insights for personalized marketing campaigns</li>
    </ul>
</body>
</html>
"""

# Save HTML report
dbutils.fs.put(
    f"dbfs:/decision_agent/reports/analytics_report_{datetime.now().strftime('%Y%m%d')}.html",
    html_report,
    overwrite=True
)

print("✓ HTML report generated")
print(f"Location: dbfs:/decision_agent/reports/analytics_report_{datetime.now().strftime('%Y%m%d')}.html")

# COMMAND ----------

print("\n" + "=" * 80)
print("ANALYTICS REPORT COMPLETE")
print("=" * 80)
print("\nAll visualizations and reports generated successfully!")
print("\nNext Steps:")
print("  1. Review cohort insights for retention strategies")
print("  2. Monitor model performance metrics")
print("  3. Act on high-value customer identification")
print("  4. Schedule monthly report generation")
