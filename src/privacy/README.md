# Privacy-Preserving Machine Learning Layer

Comprehensive privacy-preserving ML framework for multi-institution collaboration in financial services.

## Overview

This module provides production-ready privacy-preserving machine learning capabilities that enable multiple financial institutions to collaborate on model training while maintaining strict data privacy and regulatory compliance.

### Key Features

- **Federated Learning**: Train models across institutions without sharing raw data
- **Differential Privacy**: Mathematical privacy guarantees with ε-δ privacy budget tracking
- **Transfer Learning**: Cross-institution knowledge transfer with domain adaptation
- **Secure Aggregation**: Protect model updates during aggregation
- **Byzantine Robustness**: Defend against malicious participants
- **Compliance Support**: Built-in GDPR, CCPA, and HIPAA compliance features

## Core Concepts

### Federated Learning

Federated learning enables multiple institutions to collaboratively train a shared model without exchanging their private data. Each institution:

1. Receives the current global model from the server
2. Trains the model locally on their private data
3. Sends only model updates (not data) back to the server
4. Server aggregates updates to create an improved global model

**Mathematical Foundation:**

The FedAvg algorithm aggregates client updates as:

```
w_{t+1} = Σ(n_k/n) * w_k^{t+1}
```

where:
- `w_{t+1}`: Global model at round t+1
- `w_k^{t+1}`: Model update from client k
- `n_k`: Number of samples at client k
- `n`: Total samples across all clients

**Key Papers:**
- McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Li et al. (2020): "Federated Optimization in Heterogeneous Networks" (FedProx)

### Differential Privacy

Differential privacy provides mathematical guarantees that individual data points cannot be identified from model outputs.

**Definition:**

A mechanism M satisfies (ε, δ)-differential privacy if for all datasets D₁ and D₂ differing in one element, and all outcomes S:

```
P[M(D₁) ∈ S] ≤ e^ε * P[M(D₂) ∈ S] + δ
```

where:
- `ε` (epsilon): Privacy loss parameter (lower = more private)
- `δ` (delta): Probability of privacy breach (typically < 1/n²)

**Implementation:**

We implement DP-SGD (Differentially Private Stochastic Gradient Descent):

1. **Gradient Clipping**: Clip gradients to bound sensitivity
   ```
   g̃ᵢ = gᵢ / max(1, ||gᵢ||₂ / C)
   ```

2. **Noise Addition**: Add calibrated Gaussian noise
   ```
   g̃ = (1/L) * Σ g̃ᵢ + N(0, σ²C²I)
   ```

3. **Privacy Accounting**: Track cumulative privacy loss using RDP composition

**Key Papers:**
- Abadi et al. (2016): "Deep Learning with Differential Privacy"
- Dwork & Roth (2014): "The Algorithmic Foundations of Differential Privacy"

### Transfer Learning

Transfer learning enables knowledge transfer from source institutions to target institutions, useful for:
- Cold start problems (new institutions with limited data)
- Domain adaptation (different but related data distributions)
- Multi-institution collaboration

**Domain Adaptation Methods:**

1. **Feature-Based (CORAL)**: Align feature distributions
   ```
   A_coral = C_s^{-1/2} * C_t^{1/2}
   ```

2. **Instance-Based**: Reweight source samples
   ```
   w_i = p_t(x_i) / p_s(x_i)
   ```

3. **Model-Based**: Fine-tune pre-trained models

**Key Papers:**
- Pan & Yang (2010): "A Survey on Transfer Learning"
- Sun et al. (2016): "Return of Frustratingly Easy Domain Adaptation" (CORAL)

## Quick Start

### Federated Learning Example

```python
from src.privacy import (
    FederatedLearningServer,
    FederatedLearningClient,
    DifferentialPrivacyConfig
)
import numpy as np

# Server setup
server = FederatedLearningServer(
    model_config={
        'input_dim': 10,
        'hidden_dim': 64,
        'output_dim': 1
    },
    aggregation_strategy='fedavg',
    privacy_budget={
        'epsilon': 1.0,
        'delta': 1e-5,
        'method': 'rdp'
    },
    min_clients=2
)

# Client setup (for each institution)
clients = []
for i, (X_train, y_train) in enumerate(institution_data):
    client = FederatedLearningClient(
        client_id=f'institution_{i}',
        local_data=(X_train, y_train),
        dp_config=DifferentialPrivacyConfig(
            noise_multiplier=1.1,
            l2_norm_clip=1.0,
            enable_dp=True
        ),
        learning_rate=0.01,
        batch_size=32
    )
    clients.append(client)

# Federated training
num_rounds = 100
for round_num in range(num_rounds):
    # Select clients for this round
    selected_clients = server.select_clients(
        clients,
        fraction=0.3,  # 30% of clients per round
        min_clients=2
    )
    
    # Get global model
    global_model = server.get_global_model()
    
    # Train locally and collect updates
    client_updates = []
    for client in selected_clients:
        update = client.train_local_model(
            global_model=global_model,
            epochs=5,
            verbose=False
        )
        client_updates.append(update)
    
    # Aggregate updates
    server.aggregate_updates(client_updates, round_num)
    
    # Check convergence
    if server.has_converged(window=5):
        print(f"Converged at round {round_num}")
        break
    
    # Check privacy budget
    if server.privacy_tracker:
        remaining = server.privacy_tracker.get_remaining_epsilon()
        if remaining < 0.1:
            print(f"Privacy budget nearly exhausted: {remaining:.4f} remaining")
            break

# Get final model
final_model = server.get_global_model()

# Save checkpoint with privacy audit
server.save_checkpoint('checkpoints/federated_model.json')
if server.privacy_tracker:
    server.privacy_tracker.save_audit_log('audit/privacy_log.json')
```

### Transfer Learning Example

```python
from src.privacy import TransferLearner, DomainAdapter
import numpy as np

# Source institution (has lots of data)
X_source = np.random.randn(10000, 20)
y_source = np.random.randn(10000, 1)

# Target institution (limited data - cold start)
X_target = np.random.randn(100, 20)
y_target = np.random.randn(100, 1)

# Initialize transfer learner
transfer_learner = TransferLearner(
    source_data=(X_source, y_source),
    target_data=(X_target, y_target),
    transfer_method='feature_based',
    adaptation_config={'method': 'coral'}
)

# Assess transferability
scores = transfer_learner.compute_transferability()
print(f"Expected transfer gain: {scores[0].expected_transfer_gain:.3f}")
print(f"Negative transfer risk: {scores[0].negative_transfer_risk:.3f}")

# Perform adaptation
adapted_model = transfer_learner.adapt(
    source_model=pretrained_model,  # Can be None to train from scratch
    adaptation_epochs=20,
    fine_tune_layers=None  # None = fine-tune all layers
)

# Evaluate on target domain
target_metrics = transfer_learner.evaluate_on_target()
print(f"Target MSE: {target_metrics['mse']:.4f}")
print(f"Target R²: {target_metrics['r2']:.4f}")

# Detect negative transfer
neg_transfer = transfer_learner.detect_negative_transfer()
if neg_transfer['has_negative_transfer']:
    print("Warning: Negative transfer detected!")
    print(f"Performance difference: {neg_transfer['relative_improvement']:.2%}")
else:
    print(f"Positive transfer: {neg_transfer['relative_improvement']:.2%} improvement")

# Save adapted model
transfer_learner.save_adapted_model('models/adapted_model.json')
```

### Multi-Source Transfer Learning

```python
# Multiple source institutions
source_institutions = [
    (X_source1, y_source1),  # Institution 1
    (X_source2, y_source2),  # Institution 2
    (X_source3, y_source3),  # Institution 3
]

# Target institution (new/limited data)
target_data = (X_target, y_target)

# Multi-source transfer
multi_transfer = TransferLearner(
    source_data=source_institutions,
    target_data=target_data,
    transfer_method='multi_source'
)

# Compute transferability for each source
scores = multi_transfer.compute_transferability()
for i, score in enumerate(scores):
    print(f"Source {i}: gain={score.expected_transfer_gain:.3f}")

# Adapt (automatically weights sources by transferability)
adapted_model = multi_transfer.adapt(adaptation_epochs=30)

# Evaluate
metrics = multi_transfer.evaluate_on_target()
print(f"Multi-source R²: {metrics['r2']:.4f}")
```

## Privacy Budget Management

### Understanding Privacy Budget

The privacy budget (ε, δ) represents the cumulative privacy loss:

- **ε (epsilon)**: Privacy loss parameter
  - ε = 0: Perfect privacy (no learning possible)
  - ε = 1: Strong privacy (recommended for sensitive data)
  - ε = 10: Moderate privacy (common in practice)
  - ε > 10: Weak privacy

- **δ (delta)**: Probability of privacy breach
  - Typically set to 1/n² where n is dataset size
  - Example: n=10,000 → δ=1e-8

### Privacy Budget Calculator

```python
from src.privacy import PrivacyBudgetTracker

# Initialize tracker
tracker = PrivacyBudgetTracker(
    max_epsilon=10.0,
    delta=1e-5,
    composition_method='rdp'  # Renyi Differential Privacy
)

# Account for each training round
for round_num in range(100):
    epsilon_spent = tracker.account_for_round(
        noise_multiplier=1.1,
        sampling_rate=0.01,  # 1% of data per batch
        steps=100,  # 100 gradient steps
        round_num=round_num
    )
    
    print(f"Round {round_num}: spent ε={epsilon_spent:.4f}, "
          f"total={tracker.get_spent_epsilon():.4f}")
    
    if not tracker.can_continue():
        print("Privacy budget exhausted!")
        break

# Save audit log for compliance
tracker.save_audit_log('audit/privacy_budget.json')

# Get history
history = tracker.get_history()
for entry in history[-5:]:
    print(f"Round {entry['round']}: ε={entry['epsilon']:.4f}")
```

### Recommended Privacy Budgets by Use Case

| Use Case | ε | δ | Notes |
|----------|---|---|-------|
| Medical records | 0.1-1.0 | 1e-6 | Highest privacy |
| Financial transactions | 1.0-3.0 | 1e-5 | High privacy |
| Credit risk models | 3.0-8.0 | 1e-5 | Moderate privacy |
| Marketing analytics | 8.0-15.0 | 1e-4 | Lower privacy |

## Regulatory Compliance

### GDPR Compliance

The General Data Protection Regulation (EU) requires:

**✓ Right to Privacy**: Implemented via differential privacy
**✓ Data Minimization**: Only model updates shared, not raw data
**✓ Purpose Limitation**: Models used only for specified purposes
**✓ Transparency**: Audit logs track all data usage
**✓ Right to be Forgotten**: Can retrain without specific users

**Compliance Checklist:**

```python
# GDPR Compliance for Federated Learning
gdpr_compliance = {
    'lawful_basis': 'legitimate_interest',  # or 'consent'
    'data_minimization': True,  # Only gradients shared
    'privacy_by_design': True,  # Differential privacy enabled
    'audit_trail': 'audit/privacy_log.json',
    'data_retention': '90_days',
    'right_to_erasure': True,  # Can exclude users and retrain
    'privacy_impact_assessment': 'docs/pia.pdf'
}
```

### CCPA Compliance

California Consumer Privacy Act requirements:

**✓ Consumer Rights**: Right to know, delete, opt-out
**✓ Privacy Notices**: Clear disclosure of data practices
**✓ Data Security**: Encryption and secure aggregation
**✓ Non-Discrimination**: Equal service regardless of privacy choices

### HIPAA Compliance (Healthcare)

Health Insurance Portability and Accountability Act:

**✓ Privacy Rule**: Protected Health Information (PHI) safeguards
**✓ Security Rule**: Administrative, physical, technical safeguards
**✓ De-identification**: Differential privacy provides statistical de-identification
**✓ Audit Controls**: Comprehensive logging of all access

**Safe Harbor Method:**
- Differential privacy with ε ≤ 1.0 provides strong de-identification
- Combined with federated learning (no raw data sharing)
- Regular privacy audits and risk assessments

## Advanced Features

### Byzantine-Robust Aggregation

Protect against malicious or faulty clients:

```python
from src.privacy import FederatedLearningServer

server = FederatedLearningServer(
    model_config=config,
    aggregation_strategy='krum',  # or 'median', 'trimmed_mean'
    num_byzantine=2  # Expected number of malicious clients
)
```

**Methods:**
- **Krum**: Select clients with smallest distance sum
- **Median**: Coordinate-wise median (robust to outliers)
- **Trimmed Mean**: Remove extreme values before averaging

### Communication Efficiency

Reduce communication costs via gradient compression:

```python
from src.privacy.federated_learning import compress_gradients

# Compress gradients before sending
compressed, metadata = compress_gradients(
    gradients,
    compression_ratio=0.1,  # Keep top 10%
    method='topk'  # or 'random'
)

# Decompress on server
gradients = decompress_gradients(compressed, metadata)
```

**Compression Ratios:**
- 0.01 (1%): 100x reduction, minimal accuracy loss
- 0.1 (10%): 10x reduction, negligible accuracy loss
- 0.5 (50%): 2x reduction, no accuracy loss

### Privacy-Preserving Evaluation

Evaluate models without revealing test data:

```python
# Each client evaluates locally
client_metrics = client.evaluate_model(
    model=global_model,
    X_test=local_test_data,
    y_test=local_test_labels
)

# Server aggregates metrics (not data)
# Add noise to metrics for privacy
noisy_metrics = add_laplace_noise(
    client_metrics,
    sensitivity=0.1,
    epsilon=0.5
)
```

## Multi-Institution Setup Guide

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Federated Learning Server                 │
│  - Global model management                                   │
│  - Client selection & aggregation                            │
│  - Privacy budget tracking                                   │
│  - Convergence monitoring                                    │
└─────────────────────────────────────────────────────────────┘
                            ▲ ▼
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐      ┌────▼─────┐      ┌────▼─────┐
   │ Client 1 │      │ Client 2 │      │ Client 3 │
   │ Bank A   │      │ Bank B   │      │ Bank C   │
   ├──────────┤      ├──────────┤      ├──────────┤
   │ Local    │      │ Local    │      │ Local    │
   │ Data     │      │ Data     │      │ Data     │
   │ Training │      │ Training │      │ Training │
   │ DP-SGD   │      │ DP-SGD   │      │ DP-SGD   │
   └──────────┘      └──────────┘      └──────────┘
```

### Setup Steps

1. **Server Setup** (Trusted coordinator)
   ```python
   server = FederatedLearningServer(
       model_config=model_config,
       aggregation_strategy='fedavg',
       privacy_budget={'epsilon': 10.0, 'delta': 1e-5}
   )
   ```

2. **Client Registration** (Each institution)
   ```python
   client = FederatedLearningClient(
       client_id='institution_unique_id',
       local_data=(X_local, y_local),
       dp_config=DifferentialPrivacyConfig(
           noise_multiplier=1.1,
           l2_norm_clip=1.0
       )
   )
   ```

3. **Coordination Protocol**
   - Server broadcasts global model
   - Clients train locally and return updates
   - Server aggregates and updates global model
   - Repeat until convergence or budget exhausted

### Security Considerations

**Network Security:**
- Use TLS/SSL for all communications
- Authenticate clients using certificates
- Encrypt model updates in transit

**Model Security:**
- Validate client updates (check for poisoning)
- Use secure aggregation protocols
- Implement Byzantine-robust aggregation

**Privacy Audit:**
- Log all model accesses
- Track privacy budget consumption
- Regular privacy impact assessments
- Document compliance measures

## Performance Tuning

### Hyperparameter Guidelines

**Differential Privacy:**
```python
# Strong privacy (financial/medical)
dp_config = DifferentialPrivacyConfig(
    noise_multiplier=1.5,    # Higher noise
    l2_norm_clip=0.5,        # Tighter clipping
    target_epsilon=1.0
)

# Moderate privacy (general business)
dp_config = DifferentialPrivacyConfig(
    noise_multiplier=1.1,
    l2_norm_clip=1.0,
    target_epsilon=5.0
)
```

**Federated Learning:**
```python
# Communication-efficient (many clients, limited bandwidth)
server = FederatedLearningServer(
    client_selection='random',
    fraction=0.1,              # 10% clients per round
    min_clients=10,
    local_epochs=5             # More local training
)

# Convergence-focused (fewer clients, good bandwidth)
server = FederatedLearningServer(
    client_selection='importance',
    fraction=0.5,              # 50% clients per round
    min_clients=5,
    local_epochs=1             # More frequent aggregation
)
```

### Monitoring & Debugging

```python
# Monitor convergence
history = server.get_round_history()
for round_info in history[-5:]:
    print(f"Round {round_info['round']}: "
          f"loss={round_info['avg_loss']:.4f}, "
          f"change={round_info['model_change']:.6f}")

# Check privacy budget
if server.privacy_tracker:
    print(f"Privacy spent: {server.privacy_tracker.get_spent_epsilon():.4f}")
    print(f"Privacy remaining: {server.privacy_tracker.get_remaining_epsilon():.4f}")

# Visualize training progress
import matplotlib.pyplot as plt

rounds = [h['round'] for h in history]
losses = [h['avg_loss'] for h in history]

plt.plot(rounds, losses)
plt.xlabel('Round')
plt.ylabel('Average Loss')
plt.title('Federated Learning Convergence')
plt.savefig('convergence.png')
```

## API Reference

### FederatedLearningServer

**Methods:**
- `select_clients(clients, fraction, min_clients)`: Select clients for round
- `aggregate_updates(client_updates, round_num)`: Aggregate client updates
- `get_global_model()`: Get current global model
- `has_converged(window)`: Check convergence
- `save_checkpoint(path)`: Save server state

### FederatedLearningClient

**Methods:**
- `train_local_model(global_model, epochs, verbose)`: Train locally
- `evaluate_model(model, X_test, y_test)`: Evaluate model
- `get_data_size()`: Get local dataset size

### TransferLearner

**Methods:**
- `compute_transferability()`: Assess transfer potential
- `adapt(source_model, adaptation_epochs, fine_tune_layers)`: Adapt model
- `evaluate_on_target(model, X_test, y_test)`: Evaluate on target
- `detect_negative_transfer(baseline_model)`: Detect negative transfer
- `save_adapted_model(path)`: Save adapted model

### PrivacyBudgetTracker

**Methods:**
- `account_for_round(noise_multiplier, sampling_rate, steps)`: Account for privacy
- `get_spent_epsilon()`: Get total spent epsilon
- `get_remaining_epsilon()`: Get remaining budget
- `can_continue()`: Check if training can continue
- `save_audit_log(path)`: Save privacy audit

## References

### Foundational Papers

1. **Federated Learning:**
   - McMahan et al. (2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
   - Li et al. (2020): "Federated Optimization in Heterogeneous Networks"
   - Kairouz et al. (2021): "Advances and Open Problems in Federated Learning"

2. **Differential Privacy:**
   - Dwork & Roth (2014): "The Algorithmic Foundations of Differential Privacy"
   - Abadi et al. (2016): "Deep Learning with Differential Privacy"
   - Mironov (2017): "Renyi Differential Privacy"

3. **Transfer Learning:**
   - Pan & Yang (2010): "A Survey on Transfer Learning"
   - Ganin et al. (2016): "Domain-Adversarial Training of Neural Networks"
   - Sun et al. (2016): "Return of Frustratingly Easy Domain Adaptation"

4. **Secure Aggregation:**
   - Bonawitz et al. (2017): "Practical Secure Aggregation for Privacy-Preserving Machine Learning"

### Regulatory Resources

- **GDPR**: https://gdpr.eu/
- **CCPA**: https://oag.ca.gov/privacy/ccpa
- **HIPAA**: https://www.hhs.gov/hipaa/
- **NIST Privacy Framework**: https://www.nist.gov/privacy-framework

## License

This module is part of the Principal Data Science Decision Agent and is subject to the same license terms.

## Support

For questions, issues, or contributions:
- GitHub Issues: [Project Issues](https://github.com/yourorg/yourrepo/issues)
- Documentation: [Full Documentation](https://docs.yourproject.com)
- Email: privacy-ml@yourproject.com

## Changelog

### v1.0.0 (2024)
- Initial release
- Federated learning with FedAvg and FedProx
- Differential privacy with DP-SGD
- Transfer learning with domain adaptation
- Privacy budget tracking
- Multi-institution coordination
- Regulatory compliance features
