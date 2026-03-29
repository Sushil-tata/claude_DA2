# Recommender Systems Module

Comprehensive recommendation algorithms for the Principal Data Science Decision Agent, optimized for financial services applications including next best action, treatment optimization, and personalized product recommendations.

## Overview

This module provides three core components:

1. **Contextual Bandits** - Real-time decision making with exploration-exploitation tradeoffs
2. **Uplift Models** - Causal treatment effect estimation for targeted interventions
3. **Ranking Models** - Learning-to-rank for personalized recommendations

## Installation

All dependencies are included in the main `requirements.txt`. Key packages:

```bash
pip install lightgbm xgboost scikit-learn scipy econml loguru
```

## Module Structure

```
src/recommender/
├── contextual_bandits.py  # Multi-armed bandit algorithms
├── uplift_model.py        # Causal uplift modeling
├── ranking_model.py       # Learning-to-rank models
├── __init__.py           # Module exports
└── README.md             # This file
```

---

## 1. Contextual Bandits

Multi-armed bandit algorithms for sequential decision making with online learning.

### Algorithms

- **ThompsonSamplingBandit** - Bayesian approach with Beta priors
- **UCBBandit** - Upper Confidence Bound strategy
- **LinUCBBandit** - Linear contextual UCB with feature context
- **EpsilonGreedyBandit** - Baseline ε-greedy exploration
- **ContextualBanditOrchestrator** - Multi-bandit management with A/B testing

### Quick Start

```python
from src.recommender.contextual_bandits import LinUCBBandit, ContextualBanditOrchestrator

# Initialize contextual bandit
bandit = LinUCBBandit(
    n_actions=5,        # Number of actions (e.g., products, treatments)
    n_features=10,      # Context feature dimension
    alpha=1.0           # Exploration parameter
)

# Select action based on context
context = np.random.randn(10)
action = bandit.select_action(context)

# Update with observed reward
reward = 1.0  # Binary or continuous reward
bandit.update(action, reward, context)

# Get performance metrics
metrics = bandit.get_metrics()
print(f"Cumulative reward: {metrics['cumulative_reward']}")
print(f"Average reward: {metrics['avg_reward']}")
```

### A/B Testing with Orchestrator

```python
from src.recommender.contextual_bandits import (
    ThompsonSamplingBandit,
    UCBBandit,
    LinUCBBandit,
    ContextualBanditOrchestrator
)

# Create multiple bandits
bandits = {
    'thompson': ThompsonSamplingBandit(n_actions=5),
    'ucb': UCBBandit(n_actions=5, c=2.0),
    'linucb': LinUCBBandit(n_actions=5, n_features=10)
}

# Setup orchestrator
orchestrator = ContextualBanditOrchestrator(
    bandits=bandits,
    default_bandit='linucb'
)

# Configure A/B test splits
orchestrator.setup_ab_test({
    'thompson': 0.3,
    'ucb': 0.3,
    'linucb': 0.4
})

# Select action (bandit chosen based on A/B split)
action, bandit_name = orchestrator.select_action(context=context)
orchestrator.update(bandit_name, action, reward, context)

# Compare performance
comparison = orchestrator.get_comparison_metrics()
print(comparison)

# Identify best performer
best = orchestrator.get_best_bandit(metric='avg_reward')
print(f"Best bandit: {best}")
```

### Online Learning

```python
from src.recommender.contextual_bandits import OnlineBanditTrainer

# Setup online trainer
trainer = OnlineBanditTrainer(
    bandit=bandit,
    batch_size=32,
    performance_window=100
)

# Stream observations
for context, action, reward in data_stream:
    batch_completed = trainer.add_observation(action, reward, context)
    
    if batch_completed:
        # Get rolling metrics
        metrics = trainer.get_rolling_metrics()
        print(f"Rolling avg reward: {metrics['rolling_avg_reward']:.3f}")
```

### Save/Load Models

```python
# Save bandit
bandit.save('models/linucb_bandit.pkl')

# Load bandit
loaded_bandit = LinUCBBandit(n_actions=5, n_features=10)
loaded_bandit.load('models/linucb_bandit.pkl')
```

---

## 2. Uplift Models

Causal treatment effect estimation for optimizing interventions and personalized treatments.

### Algorithms

- **TLearner** - Two-model approach (separate models for treatment/control)
- **SLearner** - Single-model with treatment as feature
- **XLearner** - Advanced meta-learner with propensity weighting
- **CausalForestUplift** - Random forest for heterogeneous effects (requires EconML)
- **UpliftEnsemble** - Ensemble of multiple uplift models

### Quick Start

```python
from src.recommender.uplift_model import TLearner
from sklearn.ensemble import RandomForestClassifier

# Initialize T-Learner
uplift = TLearner(
    base_model=RandomForestClassifier(n_estimators=100),
    task='classification'
)

# Fit model
# treatment: binary indicator (1=treated, 0=control)
# y: outcome (e.g., conversion, payment)
uplift.fit(X_train, treatment_train, y_train)

# Predict uplift (CATE - Conditional Average Treatment Effect)
cate = uplift.predict_uplift(X_test)

# Identify high-uplift segments
top_10_percent = np.percentile(cate, 90)
high_uplift_customers = X_test[cate >= top_10_percent]
print(f"Target {len(high_uplift_customers)} customers with high uplift")
```

### Advanced: X-Learner with Propensity

```python
from src.recommender.uplift_model import XLearner

# X-Learner accounts for treatment assignment bias
uplift = XLearner(
    base_model=RandomForestRegressor(),
    propensity_model=LogisticRegression()
)

uplift.fit(X_train, treatment_train, y_train)
cate = uplift.predict_uplift(X_test)

# Get propensity scores
propensity = uplift.get_propensity_scores(X_test)
```

### Causal Forest

```python
from src.recommender.uplift_model import CausalForestUplift

# Requires econml: pip install econml
uplift = CausalForestUplift(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10
)

uplift.fit(X_train, treatment_train, y_train)
cate = uplift.predict_uplift(X_test)

# Get confidence intervals
lower, upper = uplift.get_confidence_intervals(X_test, alpha=0.05)
```

### Ensemble Multiple Models

```python
from src.recommender.uplift_model import UpliftEnsemble

# Create ensemble
models = [
    TLearner(RandomForestClassifier()),
    SLearner(LGBMClassifier()),
    XLearner(RandomForestRegressor())
]

ensemble = UpliftEnsemble(
    models=models,
    weights='auto'  # Optimize weights on validation set
)

# Fit with validation set for weight optimization
ensemble.fit(
    X_train, treatment_train, y_train,
    X_val=X_val, treatment_val=treatment_val, y_val=y_val
)

# Predict weighted uplift
cate = ensemble.predict_uplift(X_test)
```

### Validation Metrics

```python
from src.recommender.uplift_model import UpliftValidator

validator = UpliftValidator()

# Comprehensive evaluation
metrics = validator.evaluate(
    model=uplift,
    X=X_test,
    treatment=treatment_test,
    y=y_test
)

print(f"Qini AUC: {metrics['qini_auc']:.3f}")
print(f"ATE (top decile): {metrics['ate_top_decile']:.3f}")
print(f"Mean predicted uplift: {metrics['mean_predicted_uplift']:.3f}")

# Calculate Qini curve
qini = validator.calculate_qini_curve(
    uplift_pred=cate,
    treatment=treatment_test,
    y=y_test,
    n_bins=10
)
```

---

## 3. Ranking Models

Learning-to-rank algorithms for personalized recommendations and search.

### Algorithms

- **LambdaMARTRanker** - Gradient boosted ranking (LightGBM)
- **PairwiseRanker** - Pairwise comparison (XGBoost)
- **ListwiseRanker** - Listwise optimization (XGBoost)
- **NDCGOptimizer** - Direct NDCG optimization
- **RankingEnsemble** - Ensemble of ranking models

### Quick Start

```python
from src.recommender.ranking_model import LambdaMARTRanker

# Initialize ranker
ranker = LambdaMARTRanker(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    ndcg_at=[1, 3, 5, 10]  # Evaluate NDCG at these positions
)

# Prepare data
# X: features (user-item pairs)
# y: relevance labels (e.g., 0=not relevant, 1=relevant, 2=highly relevant)
# group: query identifiers or group sizes

# Fit with validation set
ranker.fit(
    X_train, y_train, group_train,
    X_val=X_val, y_val=y_val, group_val=group_val,
    early_stopping_rounds=50
)

# Predict ranking scores
scores = ranker.predict(X_test)

# Rank items within groups
rankings = ranker.rank(X_test, group_test, k=10)
# Returns list of top-10 indices for each query

# Get feature importance
importance = ranker.get_feature_importance(importance_type='gain')
print(importance.head(10))
```

### Position Bias Correction

```python
from src.recommender.ranking_model import PositionBiasCorrector

# Train position bias model
corrector = PositionBiasCorrector(method='inverse_propensity')
corrector.fit(positions=positions, clicks=clicks, max_position=20)

# Correct relevance labels
corrected_relevance = corrector.correct(
    relevance=y_train,
    positions=positions_train
)

# Use corrected labels for training
ranker.fit(X_train, corrected_relevance, group_train)
```

### Ranking Ensemble

```python
from src.recommender.ranking_model import (
    LambdaMARTRanker,
    PairwiseRanker,
    ListwiseRanker,
    RankingEnsemble
)

# Create ensemble
models = [
    LambdaMARTRanker(n_estimators=100),
    PairwiseRanker(n_estimators=100),
    ListwiseRanker(n_estimators=100)
]

ensemble = RankingEnsemble(
    models=models,
    weights=[0.5, 0.3, 0.2]  # Or None for uniform
)

ensemble.fit(X_train, y_train, group_train)
scores = ensemble.predict(X_test, group_test)
```

### Evaluation Metrics

```python
from src.recommender.ranking_model import RankingMetrics

metrics = RankingMetrics()

# NDCG@10
ndcg_10 = metrics.ndcg(y_true, y_pred, k=10, group=group_test)
print(f"NDCG@10: {ndcg_10:.3f}")

# Mean Average Precision
map_score = metrics.mean_average_precision(y_true, y_pred, group=group_test)
print(f"MAP: {map_score:.3f}")

# Mean Reciprocal Rank
mrr = metrics.mean_reciprocal_rank(y_true, y_pred, group=group_test)
print(f"MRR: {mrr:.3f}")

# Precision@5
p_at_5 = metrics.precision_at_k(y_true, y_pred, k=5, group=group_test)
print(f"Precision@5: {p_at_5:.3f}")
```

---

## Use Cases

### 1. Collections Next Best Action (NBA)

Recommend optimal collection strategy for each account.

```python
from src.recommender.contextual_bandits import LinUCBBandit

# Actions: [email, call, sms, letter, legal]
bandit = LinUCBBandit(n_actions=5, n_features=20, alpha=1.0)

# For each customer
for customer in customers:
    # Extract features: payment history, balance, demographics
    context = extract_features(customer)
    
    # Select best action
    action = bandit.select_action(context)
    
    # Execute action and observe reward (e.g., payment made)
    reward = execute_action(customer, action)
    
    # Update model
    bandit.update(action, reward, context)
```

### 2. Product Recommendation with Uplift

Target customers most likely to respond to product offer.

```python
from src.recommender.uplift_model import XLearner

# Historical data: customers offered vs not offered
uplift = XLearner()
uplift.fit(X_historical, treatment=offer_flag, y=purchase)

# Predict uplift for new customers
new_customers_uplift = uplift.predict_uplift(X_new_customers)

# Target top 20% with highest predicted uplift
threshold = np.percentile(new_customers_uplift, 80)
target_customers = X_new_customers[new_customers_uplift >= threshold]
```

### 3. Personalized Content Ranking

Rank content items for each user.

```python
from src.recommender.ranking_model import LambdaMARTRanker

# Features: user-item pairs with engagement signals
# Labels: relevance scores (clicks, time spent, etc.)
# Groups: user IDs

ranker = LambdaMARTRanker(ndcg_at=[5, 10])
ranker.fit(X_train, relevance_train, user_groups_train)

# For each user, rank all available items
user_id = 12345
user_items = get_user_item_pairs(user_id)
scores = ranker.predict(user_items)

# Show top 10
top_10_idx = np.argsort(-scores)[:10]
recommended_items = user_items.iloc[top_10_idx]
```

### 4. Treatment Heterogeneity Analysis

Identify which customer segments benefit most from treatment.

```python
from src.recommender.uplift_model import CausalForestUplift, UpliftValidator

# Fit causal forest
uplift = CausalForestUplift(n_estimators=200)
uplift.fit(X, treatment, y)

# Predict with confidence intervals
cate, (lower, upper) = uplift.predict_uplift(X_test), \
                       uplift.get_confidence_intervals(X_test)

# Segment analysis
for segment in ['high_balance', 'low_balance', 'delinquent']:
    segment_mask = get_segment_mask(X_test, segment)
    segment_uplift = cate[segment_mask]
    print(f"{segment}: Mean uplift = {segment_uplift.mean():.3f}")
```

---

## Best Practices

### 1. Contextual Bandits

- **Cold Start**: Use epsilon-greedy or UCB initially to gather data
- **Feature Engineering**: Include temporal features, customer history
- **Regret Tracking**: Monitor cumulative regret vs. oracle policy
- **A/B Testing**: Compare multiple algorithms before deployment
- **Exploration Tuning**: Balance exploration-exploitation (alpha parameter)

### 2. Uplift Models

- **Randomization**: Ensure treatment assignment is randomized or use propensity models
- **Validation**: Use Qini curves and uplift curves, not standard accuracy
- **Segment Analysis**: Analyze uplift by customer segments
- **Confidence Intervals**: Report uncertainty in treatment effects
- **Negative Uplift**: Watch for "sleeping dogs" - customers harmed by treatment

### 3. Ranking Models

- **Position Bias**: Correct for position bias in click data
- **Group Normalization**: Normalize features within query groups
- **NDCG@k**: Optimize for your actual use case (k=5, 10, etc.)
- **Feature Importance**: Regularly review to ensure model makes sense
- **Online Metrics**: Track online metrics (CTR, engagement) post-deployment

---

## Performance Considerations

### Memory Efficiency

- Use mini-batch updates for bandits with streaming data
- For ranking, process groups in chunks if memory-constrained
- Save models periodically during long training runs

### Speed Optimization

- **Bandits**: LinUCB requires matrix inversion (O(d³)), cache when possible
- **Uplift**: T-Learner faster than X-Learner, but less accurate
- **Ranking**: LightGBM generally faster than XGBoost for large datasets

### Production Deployment

```python
# Example: Real-time API endpoint
from fastapi import FastAPI
import joblib

app = FastAPI()

# Load trained model
bandit = joblib.load('models/production_bandit.pkl')

@app.post("/recommend")
def recommend(customer_features: dict):
    context = extract_context(customer_features)
    action = bandit.select_action(context)
    return {"recommended_action": action}

@app.post("/feedback")
def feedback(action: int, reward: float, context: list):
    bandit.update(action, reward, np.array(context))
    return {"status": "updated"}
```

---

## Testing

```python
# Run module tests
pytest tests/test_recommender/

# Run specific test file
pytest tests/test_recommender/test_contextual_bandits.py -v

# Run with coverage
pytest tests/test_recommender/ --cov=src.recommender --cov-report=html
```

---

## References

### Contextual Bandits
- Li et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- Agrawal & Goyal (2013). "Thompson Sampling for Contextual Bandits with Linear Payoffs"
- Auer et al. (2002). "Using Confidence Bounds for Exploitation-Exploration Trade-offs"

### Uplift Modeling
- Künzel et al. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning"
- Athey & Imbens (2016). "Recursive partitioning for heterogeneous causal effects"
- Radcliffe & Surry (2011). "Real-World Uplift Modelling with Significance-Based Uplift Trees"

### Learning to Rank
- Burges (2010). "From RankNet to LambdaRank to LambdaMART: An Overview"
- Cao et al. (2007). "Learning to Rank: From Pairwise Approach to Listwise Approach"
- Järvelin & Kekäläinen (2002). "Cumulated gain-based evaluation of IR techniques"

---

## Support

For questions or issues:
- Check documentation in docstrings
- Review examples in `examples/recommender/`
- See integration tests in `tests/test_recommender/`

## License

Internal use for Principal Data Science Decision Agent project.

---

**Version**: 1.0.0  
**Last Updated**: 2024
