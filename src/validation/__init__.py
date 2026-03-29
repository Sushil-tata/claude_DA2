"""
Validation Framework for Principal Data Science Decision Agent

Comprehensive model validation including:
- Stability testing (PSI/CSI)
- Adversarial validation
- Drift monitoring
- Calibration validation
- Governance and regulatory reporting
"""

from .stability_testing import (
    StabilityTester,
    StabilityResult,
    calculate_psi_quick
)

from .adversarial_validation import (
    AdversarialValidator,
    AdversarialResult,
    quick_adversarial_check
)

from .drift_monitor import (
    DriftMonitor,
    DriftReport
)

from .calibration_validator import (
    CalibrationValidator,
    CalibrationResult,
    quick_calibration_check
)

from .governance_report import (
    GovernanceReporter,
    ModelCard,
    AuditTrail
)

__all__ = [
    # Stability Testing
    'StabilityTester',
    'StabilityResult',
    'calculate_psi_quick',
    
    # Adversarial Validation
    'AdversarialValidator',
    'AdversarialResult',
    'quick_adversarial_check',
    
    # Drift Monitoring
    'DriftMonitor',
    'DriftReport',
    
    # Calibration Validation
    'CalibrationValidator',
    'CalibrationResult',
    'quick_calibration_check',
    
    # Governance & Reporting
    'GovernanceReporter',
    'ModelCard',
    'AuditTrail',
]

__version__ = '1.0.0'
