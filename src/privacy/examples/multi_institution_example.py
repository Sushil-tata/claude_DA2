"""
Example: Multi-Institution Federated Learning Setup

This example demonstrates setting up federated learning across
multiple financial institutions for credit risk modeling.

Scenario:
- 5 banks want to collaboratively train a credit risk model
- Each bank has proprietary customer data (cannot be shared)
- Need to comply with GDPR and maintain strong privacy
- Target: Train better model than individual banks can alone
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger

from src.privacy import (
    FederatedLearningServer,
    FederatedLearningClient,
    DifferentialPrivacyConfig,
    PrivacyBudgetTracker
)


def generate_synthetic_institution_data(
    institution_id: int,
    num_samples: int = 1000,
    num_features: int = 20,
    heterogeneity: float = 0.3
) -> tuple:
    """Generate synthetic data for an institution.
    
    Simulates heterogeneous data across institutions
    (non-IID distribution).
    
    Args:
        institution_id: Institution identifier
        num_samples: Number of samples
        num_features: Number of features
        heterogeneity: Degree of heterogeneity (0=identical, 1=very different)
    
    Returns:
        (X, y): Features and labels
    """
    np.random.seed(institution_id * 42)
    
    # Each institution has slightly different data distribution
    mean_shift = heterogeneity * np.random.randn(num_features)
    std_scale = 1.0 + heterogeneity * np.random.rand()
    
    X = np.random.randn(num_samples, num_features) * std_scale + mean_shift
    
    # True model with institution-specific bias
    true_weights = np.random.randn(num_features, 1)
    institution_bias = heterogeneity * np.random.randn()
    
    y = X @ true_weights + institution_bias + 0.1 * np.random.randn(num_samples, 1)
    
    return X, y


def setup_institutions(
    num_institutions: int = 5,
    samples_per_institution: int = 1000,
    enable_dp: bool = True
) -> list:
    """Setup clients for each institution.
    
    Args:
        num_institutions: Number of participating institutions
        samples_per_institution: Samples per institution
        enable_dp: Whether to enable differential privacy
    
    Returns:
        List of FederatedLearningClient instances
    """
    clients = []
    
    logger.info(f"Setting up {num_institutions} institutions...")
    
    for i in range(num_institutions):
        # Generate institution data
        X, y = generate_synthetic_institution_data(
            institution_id=i,
            num_samples=samples_per_institution,
            heterogeneity=0.3
        )
        
        # Create client with differential privacy
        if enable_dp:
            dp_config = DifferentialPrivacyConfig(
                noise_multiplier=1.1,
                l2_norm_clip=1.0,
                delta=1e-5,
                target_epsilon=1.0,
                enable_dp=True
            )
        else:
            dp_config = DifferentialPrivacyConfig(enable_dp=False)
        
        client = FederatedLearningClient(
            client_id=f'bank_{i+1}',
            local_data=(X, y),
            dp_config=dp_config,
            learning_rate=0.01,
            batch_size=32
        )
        
        clients.append(client)
        
        logger.info(
            f"  Bank {i+1}: {len(X)} samples, "
            f"DP={'enabled' if enable_dp else 'disabled'}"
        )
    
    return clients


def run_federated_training(
    clients: list,
    num_rounds: int = 100,
    client_fraction: float = 0.6,
    local_epochs: int = 5,
    privacy_budget: dict = None
) -> tuple:
    """Run federated learning training."""
    logger.info("Initializing federated learning server...")
    
    server = FederatedLearningServer(
        model_config={
            'input_dim': 20,
            'hidden_dim': 64,
            'output_dim': 1
        },
        aggregation_strategy='fedavg',
        privacy_budget=privacy_budget,
        client_selection='random',
        min_clients=2,
        convergence_threshold=1e-4
    )
    
    logger.info(f"Starting federated training for {num_rounds} rounds...")
    
    history = {
        'rounds': [],
        'losses': [],
        'model_changes': [],
        'privacy_spent': [],
        'num_clients': []
    }
    
    for round_num in range(num_rounds):
        selected_clients = server.select_clients(
            clients,
            fraction=client_fraction,
            min_clients=2
        )
        
        global_model = server.get_global_model()
        
        client_updates = []
        for client in selected_clients:
            update = client.train_local_model(
                global_model=global_model,
                epochs=local_epochs,
                verbose=False
            )
            client_updates.append(update)
        
        server.aggregate_updates(client_updates, round_num)
        
        round_info = server.get_round_history()[-1]
        history['rounds'].append(round_num)
        history['losses'].append(round_info['avg_loss'])
        history['model_changes'].append(round_info['model_change'])
        history['num_clients'].append(len(selected_clients))
        
        if server.privacy_tracker:
            history['privacy_spent'].append(
                server.privacy_tracker.get_spent_epsilon()
            )
        
        if (round_num + 1) % 10 == 0:
            log_msg = (
                f"Round {round_num + 1}/{num_rounds}: "
                f"loss={round_info['avg_loss']:.4f}"
            )
            
            if server.privacy_tracker:
                epsilon_spent = server.privacy_tracker.get_spent_epsilon()
                log_msg += f", ε_spent={epsilon_spent:.4f}"
            
            logger.info(log_msg)
        
        if server.has_converged(window=10):
            logger.info(f"Training converged at round {round_num + 1}")
            break
        
        if server.privacy_tracker and not server.privacy_tracker.can_continue():
            logger.warning(f"Privacy budget exhausted at round {round_num + 1}")
            break
    
    logger.info("Federated training complete!")
    
    return server, history


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Multi-Institution Federated Learning Example")
    logger.info("=" * 80)
    
    clients = setup_institutions(
        num_institutions=5,
        samples_per_institution=1000,
        enable_dp=True
    )
    
    server, history = run_federated_training(
        clients=clients,
        num_rounds=100,
        client_fraction=0.6,
        local_epochs=5,
        privacy_budget={'epsilon': 10.0, 'delta': 1e-5, 'method': 'rdp'}
    )
    
    output_dir = Path('outputs/federated_learning')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    server.save_checkpoint(output_dir / 'federated_model.json')
    
    if server.privacy_tracker:
        server.privacy_tracker.save_audit_log(
            output_dir / 'privacy_audit.json'
        )
    
    logger.info("=" * 80)
    logger.info("Example complete!")
    logger.info("=" * 80)
    
    return server, clients, history


if __name__ == '__main__':
    main()
