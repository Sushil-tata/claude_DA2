"""
Privacy Budget Calculator and Utilities.

Provides tools for computing, analyzing, and visualizing privacy budgets
for differential privacy in federated learning.

Usage:
    >>> calculator = PrivacyBudgetCalculator()
    >>> 
    >>> # Calculate epsilon for training scenario
    >>> epsilon = calculator.calculate_epsilon(
    ...     noise_multiplier=1.1,
    ...     sampling_rate=0.01,
    ...     steps=1000,
    ...     delta=1e-5
    ... )
    >>> 
    >>> # Recommend parameters for target epsilon
    >>> params = calculator.recommend_parameters(
    ...     target_epsilon=1.0,
    ...     dataset_size=10000,
    ...     num_epochs=10,
    ...     batch_size=32
    ... )
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


@dataclass
class PrivacyParameters:
    """Privacy parameters for differential privacy.
    
    Attributes:
        epsilon: Privacy loss parameter
        delta: Probability of privacy breach
        noise_multiplier: Gaussian noise multiplier
        l2_norm_clip: Gradient clipping threshold
        sampling_rate: Sampling rate per step
        steps: Number of training steps
    """
    epsilon: float
    delta: float
    noise_multiplier: float
    l2_norm_clip: float
    sampling_rate: float
    steps: int


class PrivacyBudgetCalculator:
    """Calculate and analyze privacy budgets for differential privacy.
    
    Provides utilities for:
    - Computing epsilon given DP parameters
    - Recommending parameters for target epsilon
    - Analyzing privacy-utility tradeoffs
    - Comparing different privacy regimes
    
    Example:
        >>> calc = PrivacyBudgetCalculator()
        >>> epsilon = calc.calculate_epsilon(
        ...     noise_multiplier=1.1,
        ...     sampling_rate=0.01,
        ...     steps=1000,
        ...     delta=1e-5
        ... )
        >>> print(f"Epsilon: {epsilon:.4f}")
    """
    
    def __init__(self):
        """Initialize privacy budget calculator."""
        pass
    
    def calculate_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int,
        delta: float,
        method: str = 'rdp'
    ) -> float:
        """Calculate epsilon for given DP-SGD parameters.
        
        Args:
            noise_multiplier: Gaussian noise multiplier (σ)
            sampling_rate: Sampling rate per step (q)
            steps: Number of training steps (T)
            delta: Privacy parameter delta (δ)
            method: Accounting method ('rdp', 'basic', 'advanced')
        
        Returns:
            Calculated epsilon (ε)
        
        Notes:
            This is a simplified calculation. For production use with
            high privacy requirements, use libraries like tensorflow-privacy
            or opacus for precise privacy accounting.
        """
        if method == 'rdp':
            return self._calculate_rdp_epsilon(
                noise_multiplier, sampling_rate, steps, delta
            )
        elif method == 'basic':
            return self._calculate_basic_epsilon(
                noise_multiplier, sampling_rate, steps
            )
        elif method == 'advanced':
            return self._calculate_advanced_epsilon(
                noise_multiplier, sampling_rate, steps, delta
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _calculate_rdp_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int,
        delta: float
    ) -> float:
        """Calculate epsilon using RDP composition (simplified).
        
        Based on Renyi Differential Privacy analysis.
        This is a conservative estimate.
        """
        q = sampling_rate
        sigma = noise_multiplier
        
        # RDP at order α
        # For Gaussian mechanism: ρ(α) ≈ α / (2σ²)
        # With sampling: ρ(α) ≈ q²α / (2σ²)
        
        # Use α = 32 (common choice)
        alpha = 32
        
        # RDP per step
        rdp_per_step = (q * q * alpha) / (2 * sigma * sigma)
        
        # RDP after T steps (composition)
        rdp_total = rdp_per_step * steps
        
        # Convert RDP to (ε, δ)-DP
        # ε(δ) = rdp - log(δ) / (α - 1)
        epsilon = rdp_total - math.log(delta) / (alpha - 1)
        
        return max(0, epsilon)
    
    def _calculate_basic_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int
    ) -> float:
        """Calculate epsilon using basic composition.
        
        Basic composition: ε_total = T * ε_step
        Very conservative, often overestimates.
        """
        sigma = noise_multiplier
        
        # Per-step epsilon (approximate)
        epsilon_per_step = 1.0 / sigma
        
        # Total epsilon
        epsilon = epsilon_per_step * steps
        
        return epsilon
    
    def _calculate_advanced_epsilon(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        steps: int,
        delta: float
    ) -> float:
        """Calculate epsilon using advanced composition.
        
        Advanced composition: ε ≈ O(√(T log(1/δ)) * ε_step)
        Better than basic, but RDP is tighter.
        """
        sigma = noise_multiplier
        
        # Per-step epsilon
        epsilon_per_step = 1.0 / sigma
        
        # Advanced composition
        epsilon = epsilon_per_step * math.sqrt(2 * steps * math.log(1 / delta))
        
        return epsilon
    
    def recommend_parameters(
        self,
        target_epsilon: float,
        dataset_size: int,
        num_epochs: int,
        batch_size: int,
        delta: Optional[float] = None,
        method: str = 'rdp'
    ) -> PrivacyParameters:
        """Recommend DP parameters to achieve target epsilon.
        
        Args:
            target_epsilon: Target privacy budget
            dataset_size: Size of training dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            delta: Privacy delta (defaults to 1/dataset_size²)
            method: Accounting method
        
        Returns:
            Recommended privacy parameters
        
        Example:
            >>> calc = PrivacyBudgetCalculator()
            >>> params = calc.recommend_parameters(
            ...     target_epsilon=1.0,
            ...     dataset_size=10000,
            ...     num_epochs=10,
            ...     batch_size=32
            ... )
            >>> print(f"Use noise_multiplier={params.noise_multiplier}")
        """
        if delta is None:
            delta = 1.0 / (dataset_size ** 2)
        
        # Calculate steps
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * num_epochs
        sampling_rate = batch_size / dataset_size
        
        # Binary search for optimal noise multiplier
        noise_min, noise_max = 0.1, 10.0
        
        for _ in range(20):  # 20 iterations of binary search
            noise_mid = (noise_min + noise_max) / 2
            
            epsilon = self.calculate_epsilon(
                noise_multiplier=noise_mid,
                sampling_rate=sampling_rate,
                steps=total_steps,
                delta=delta,
                method=method
            )
            
            if epsilon > target_epsilon:
                noise_min = noise_mid  # Need more noise
            else:
                noise_max = noise_mid  # Can use less noise
        
        # Use conservative estimate
        recommended_noise = noise_max
        
        # Recommend gradient clipping based on noise
        l2_norm_clip = 1.0  # Standard choice
        
        params = PrivacyParameters(
            epsilon=target_epsilon,
            delta=delta,
            noise_multiplier=recommended_noise,
            l2_norm_clip=l2_norm_clip,
            sampling_rate=sampling_rate,
            steps=total_steps
        )
        
        logger.info(
            f"Recommended parameters for ε={target_epsilon:.2f}: "
            f"noise={recommended_noise:.3f}, "
            f"clip={l2_norm_clip}, "
            f"steps={total_steps}"
        )
        
        return params
    
    def analyze_tradeoffs(
        self,
        dataset_size: int,
        num_epochs: int,
        batch_size: int,
        target_epsilons: List[float] = None,
        delta: Optional[float] = None
    ) -> Dict[float, PrivacyParameters]:
        """Analyze privacy-utility tradeoffs for different epsilon values.
        
        Args:
            dataset_size: Dataset size
            num_epochs: Number of epochs
            batch_size: Batch size
            target_epsilons: List of epsilon values to analyze
            delta: Privacy delta
        
        Returns:
            Dictionary mapping epsilon to recommended parameters
        
        Example:
            >>> calc = PrivacyBudgetCalculator()
            >>> tradeoffs = calc.analyze_tradeoffs(
            ...     dataset_size=10000,
            ...     num_epochs=10,
            ...     batch_size=32,
            ...     target_epsilons=[0.1, 1.0, 5.0, 10.0]
            ... )
            >>> for eps, params in tradeoffs.items():
            ...     print(f"ε={eps}: noise={params.noise_multiplier:.3f}")
        """
        if target_epsilons is None:
            target_epsilons = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
        
        if delta is None:
            delta = 1.0 / (dataset_size ** 2)
        
        tradeoffs = {}
        
        for epsilon in target_epsilons:
            params = self.recommend_parameters(
                target_epsilon=epsilon,
                dataset_size=dataset_size,
                num_epochs=num_epochs,
                batch_size=batch_size,
                delta=delta
            )
            tradeoffs[epsilon] = params
        
        return tradeoffs
    
    def compute_utility_loss(
        self,
        noise_multiplier: float,
        baseline_accuracy: float = 0.9
    ) -> float:
        """Estimate utility loss from differential privacy noise.
        
        Args:
            noise_multiplier: Noise multiplier
            baseline_accuracy: Baseline accuracy without DP
        
        Returns:
            Estimated accuracy with DP
        
        Notes:
            This is a rough approximation. Actual utility loss depends
            on many factors including model architecture, dataset, etc.
        """
        # Empirical approximation: higher noise → more utility loss
        # Typical utility loss: 1-5% for noise_multiplier ∈ [1.0, 1.5]
        
        if noise_multiplier <= 1.0:
            utility_loss = 0.01  # 1% loss
        elif noise_multiplier <= 1.5:
            utility_loss = 0.01 + 0.04 * (noise_multiplier - 1.0)
        else:
            utility_loss = 0.05 + 0.05 * (noise_multiplier - 1.5)
        
        # Cap utility loss at 20%
        utility_loss = min(utility_loss, 0.20)
        
        estimated_accuracy = baseline_accuracy * (1 - utility_loss)
        
        return estimated_accuracy
    
    def compare_privacy_regimes(
        self,
        scenarios: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Compare different privacy scenarios.
        
        Args:
            scenarios: List of scenario configurations
        
        Returns:
            List of scenarios with computed privacy budgets
        
        Example:
            >>> scenarios = [
            ...     {'name': 'Strong Privacy', 'noise': 1.5, 'epochs': 5},
            ...     {'name': 'Moderate Privacy', 'noise': 1.1, 'epochs': 10},
            ...     {'name': 'Weak Privacy', 'noise': 0.8, 'epochs': 20}
            ... ]
            >>> results = calc.compare_privacy_regimes(scenarios)
        """
        results = []
        
        for scenario in scenarios:
            name = scenario.get('name', 'Unnamed')
            noise = scenario.get('noise_multiplier', 1.1)
            epochs = scenario.get('num_epochs', 10)
            dataset_size = scenario.get('dataset_size', 10000)
            batch_size = scenario.get('batch_size', 32)
            delta = scenario.get('delta', 1e-5)
            
            steps = (dataset_size // batch_size) * epochs
            sampling_rate = batch_size / dataset_size
            
            epsilon = self.calculate_epsilon(
                noise_multiplier=noise,
                sampling_rate=sampling_rate,
                steps=steps,
                delta=delta
            )
            
            utility = self.compute_utility_loss(noise)
            
            result = {
                'name': name,
                'epsilon': epsilon,
                'delta': delta,
                'noise_multiplier': noise,
                'num_epochs': epochs,
                'estimated_accuracy': utility,
                'privacy_level': self._classify_privacy_level(epsilon)
            }
            
            results.append(result)
            
            logger.info(
                f"{name}: ε={epsilon:.2f}, δ={delta:.2e}, "
                f"noise={noise:.2f}, privacy={result['privacy_level']}"
            )
        
        return results
    
    def _classify_privacy_level(self, epsilon: float) -> str:
        """Classify privacy level based on epsilon."""
        if epsilon <= 0.1:
            return 'Very Strong'
        elif epsilon <= 1.0:
            return 'Strong'
        elif epsilon <= 3.0:
            return 'Moderate'
        elif epsilon <= 10.0:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def generate_privacy_report(
        self,
        params: PrivacyParameters,
        regulation: str = 'GDPR'
    ) -> str:
        """Generate a privacy compliance report.
        
        Args:
            params: Privacy parameters
            regulation: Regulation to check against ('GDPR', 'HIPAA', 'CCPA')
        
        Returns:
            Formatted privacy report
        """
        report = []
        report.append("=" * 60)
        report.append("DIFFERENTIAL PRIVACY REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("Privacy Parameters:")
        report.append(f"  Epsilon (ε): {params.epsilon:.4f}")
        report.append(f"  Delta (δ): {params.delta:.2e}")
        report.append(f"  Noise Multiplier: {params.noise_multiplier:.3f}")
        report.append(f"  L2 Norm Clip: {params.l2_norm_clip:.3f}")
        report.append(f"  Sampling Rate: {params.sampling_rate:.4f}")
        report.append(f"  Training Steps: {params.steps}")
        report.append("")
        
        privacy_level = self._classify_privacy_level(params.epsilon)
        report.append(f"Privacy Level: {privacy_level}")
        report.append("")
        
        # Regulation-specific guidance
        report.append(f"Compliance Assessment ({regulation}):")
        
        if regulation == 'GDPR':
            if params.epsilon <= 1.0:
                report.append("  ✓ Meets strong privacy requirements")
                report.append("  ✓ Suitable for high-risk processing")
            elif params.epsilon <= 3.0:
                report.append("  ✓ Meets moderate privacy requirements")
                report.append("  ⚠ Consider DPIA for sensitive data")
            else:
                report.append("  ⚠ Weak privacy guarantees")
                report.append("  ⚠ DPIA required, additional safeguards needed")
        
        elif regulation == 'HIPAA':
            if params.epsilon <= 1.0:
                report.append("  ✓ Strong de-identification")
                report.append("  ✓ Suitable for PHI")
            else:
                report.append("  ⚠ May not meet de-identification standards")
                report.append("  ⚠ Expert determination recommended")
        
        elif regulation == 'CCPA':
            if params.epsilon <= 5.0:
                report.append("  ✓ Reasonable privacy safeguards")
                report.append("  ✓ Supports non-discrimination requirements")
            else:
                report.append("  ⚠ Consider additional privacy measures")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Example usage of privacy budget calculator."""
    logger.info("Privacy Budget Calculator Examples")
    logger.info("=" * 60)
    
    calc = PrivacyBudgetCalculator()
    
    # Example 1: Calculate epsilon for given parameters
    logger.info("\nExample 1: Calculate epsilon")
    epsilon = calc.calculate_epsilon(
        noise_multiplier=1.1,
        sampling_rate=0.01,
        steps=1000,
        delta=1e-5
    )
    logger.info(f"Calculated epsilon: {epsilon:.4f}")
    
    # Example 2: Recommend parameters for target epsilon
    logger.info("\nExample 2: Recommend parameters")
    params = calc.recommend_parameters(
        target_epsilon=1.0,
        dataset_size=10000,
        num_epochs=10,
        batch_size=32
    )
    logger.info(f"Recommended noise multiplier: {params.noise_multiplier:.3f}")
    
    # Example 3: Analyze tradeoffs
    logger.info("\nExample 3: Privacy-utility tradeoffs")
    tradeoffs = calc.analyze_tradeoffs(
        dataset_size=10000,
        num_epochs=10,
        batch_size=32,
        target_epsilons=[0.1, 1.0, 5.0, 10.0]
    )
    
    # Example 4: Generate privacy report
    logger.info("\nExample 4: Privacy report")
    report = calc.generate_privacy_report(params, regulation='GDPR')
    print(report)


if __name__ == '__main__':
    main()
