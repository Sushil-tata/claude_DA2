"""
Recovery Agent Practical
========================

Production-ready debt collections decision agent using rule-based approach.

Architecture: Segmentation → Recovery Curves → Relative Recovery Risk → 6M Scorecard → Action Overlay

Components:
-----------
1. PersonaBuilder: Rule-based customer segmentation (5 personas)
2. RecoveryScorecard6M: 6-month recovery prediction (two-part or Tweedie)
3. ActionOverlayRouter: Rule-based action routing (NO NPV optimization)
4. RecoveryAgentPipeline: End-to-end orchestrator

Usage:
------
```python
from recovery_agent_practical import RecoveryAgentPipeline

# Initialize pipeline
pipeline = RecoveryAgentPipeline(scorecard_model_type="TWO_PART")

# Train scorecard
metrics = pipeline.train_scorecard(features_df, labels_df)

# Score daily batch
daily_scores = pipeline.score_batch(
    features_df=today_features,
    score_date="2024-01-15",
    write_outputs=True,
    output_path="./outputs"
)
```
"""

from recovery_agent_practical.segmentation.persona_builder import (
    PersonaBuilder,
    PersonaAssignment,
)

from recovery_agent_practical.scoring.recovery_scorecard_6m import (
    RecoveryScorecard6M,
    RecoveryScore,
    TwoPartRecoveryModel,
    TweedieRecoveryModel,
    ModelMetrics,
)

from recovery_agent_practical.routing.action_overlay import (
    ActionOverlayRouter,
    ActionRecommendation,
)

from recovery_agent_practical.pipeline import RecoveryAgentPipeline

from recovery_agent_practical.outputs.schemas import (
    DAILY_SCORING_SCHEMA,
    AUDIT_LOG_SCHEMA,
    create_daily_scoring_row,
    create_audit_event,
)

__version__ = "1.0.0"

__all__ = [
    # Pipeline
    "RecoveryAgentPipeline",

    # Components
    "PersonaBuilder",
    "RecoveryScorecard6M",
    "ActionOverlayRouter",

    # Models
    "TwoPartRecoveryModel",
    "TweedieRecoveryModel",

    # Data classes
    "PersonaAssignment",
    "RecoveryScore",
    "ActionRecommendation",
    "ModelMetrics",

    # Schemas
    "DAILY_SCORING_SCHEMA",
    "AUDIT_LOG_SCHEMA",
    "create_daily_scoring_row",
    "create_audit_event",
]
