"""
Advanced Behavioral Physics - State-of-the-Art Bureau Features
================================================================

Novel physics-inspired features beyond basic velocity/acceleration.
True physics concepts applied to credit bureau dynamics.

Feature Families (100+ features):
1. MOMENTUM & INERTIA (15 features)
2. ENERGY DYNAMICS (18 features)
3. THERMODYNAMICS (12 features)
4. WAVE MECHANICS (10 features)
5. STRESS TENSOR (14 features)
6. CHAOS & ATTRACTORS (12 features)
7. NETWORK TOPOLOGY (15 features)
8. FIELD THEORY (10 features)
9. PHASE TRANSITIONS (8 features)
10. RELATIVITY (6 features)

Author: Behavioral Physics Team
Version: 2.0.0 - Advanced
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from typing import Dict, List, Tuple
import numpy as np

# ============================================================================
# 1. MOMENTUM & INERTIA (15 features)
# ============================================================================

def add_momentum_inertia_features(panel: DataFrame, windows=[3, 6, 12]) -> DataFrame:
    """
    Momentum: mass × velocity (how hard is it to stop deterioration)
    Inertia: resistance to state change (stickiness)

    Physics: p = mv, I = mr²
    Bureau: momentum = balance × dpd_velocity
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Debt momentum: balance × DPD velocity
    panel = panel.withColumn(
        "debt_momentum_m",
        F.col("bureau_owed_sum") * F.coalesce(
            (F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 1).over(w)),
            F.lit(0.0)
        )
    )

    # Rotational inertia: balance × (state distance from S0)²
    panel = panel.withColumn(
        "rotational_inertia_m",
        F.col("bureau_owed_sum") * F.pow(F.col("state_num"), 2)
    )

    # State stickiness: streak_len / total_months_observed
    panel = panel.withColumn(
        "state_inertia_index",
        F.col("streak_len") / (F.row_number().over(w))
    )

    # Momentum conservation: when CardX momentum transfers to others
    panel = panel.withColumn(
        "cardx_momentum_m",
        F.col("bureau_cardx_owed") * F.coalesce(
            (F.col("bureau_cardx_max_dpd") - F.lag("bureau_cardx_max_dpd", 1).over(w)),
            F.lit(0.0)
        )
    )

    panel = panel.withColumn(
        "others_momentum_m",
        F.col("bureau_others_owed") * F.coalesce(
            (F.col("bureau_others_max_dpd") - F.lag("bureau_others_max_dpd", 1).over(w)),
            F.lit(0.0)
        )
    )

    # Momentum transfer ratio (conservation check)
    panel = panel.withColumn(
        "momentum_transfer_ratio_m",
        F.when(
            F.col("cardx_momentum_m") != 0,
            F.col("others_momentum_m") / F.col("cardx_momentum_m")
        ).otherwise(F.lit(None))
    )

    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        # Net momentum change (impulse)
        panel = panel.withColumn(
            f"impulse_{m}m",
            F.col("debt_momentum_m") - F.first("debt_momentum_m").over(ww)
        )

        # Momentum volatility (uncertainty)
        panel = panel.withColumn(
            f"momentum_volatility_{m}m",
            F.stddev_pop("debt_momentum_m").over(ww)
        )

        # Angular momentum (cross-lender rotation)
        # |L| = |r × p| approximated as lender_diversity × momentum
        panel = panel.withColumn(
            f"angular_momentum_{m}m",
            F.avg(F.col("bureau_member_cnt") * F.col("debt_momentum_m")).over(ww)
        )

    return panel


# ============================================================================
# 2. ENERGY DYNAMICS (18 features)
# ============================================================================

def add_energy_features(panel: DataFrame, windows=[3, 6, 12]) -> DataFrame:
    """
    Potential Energy: position in state space (height above S0)
    Kinetic Energy: speed of change (velocity²)
    Total Energy: conserved quantity (potential + kinetic)

    Physics: PE = mgh, KE = ½mv², E = PE + KE
    Bureau: PE = balance × state_height, KE = ½ × balance × velocity²
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # State height: map S0→0, S1→1, S2→2, S3→3, S4→4
    panel = panel.withColumn("state_height", F.col("state_num"))

    # Potential energy: balance × state_height (gravitational)
    panel = panel.withColumn(
        "potential_energy_m",
        F.col("bureau_owed_sum") * F.col("state_height")
    )

    # DPD velocity
    panel = panel.withColumn(
        "dpd_velocity_m",
        F.coalesce(
            F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 1).over(w),
            F.lit(0.0)
        )
    )

    # Kinetic energy: ½ × balance × velocity²
    panel = panel.withColumn(
        "kinetic_energy_m",
        0.5 * F.col("bureau_owed_sum") * F.pow(F.col("dpd_velocity_m"), 2)
    )

    # Total energy: PE + KE (should be conserved)
    panel = panel.withColumn(
        "total_energy_m",
        F.col("potential_energy_m") + F.col("kinetic_energy_m")
    )

    # Energy dissipation: loss of total energy (friction)
    panel = panel.withColumn(
        "energy_dissipation_m",
        F.lag("total_energy_m", 1).over(w) - F.col("total_energy_m")
    )

    # Escape velocity: speed needed to reach S4 from current state
    # v_escape = √(2 × g × h) where h = (4 - current_state)
    panel = panel.withColumn(
        "escape_velocity_m",
        F.sqrt(2.0 * 30.0 * (4.0 - F.col("state_num")))  # g=30 DPD/month
    )

    # Energy barrier: activation energy to change state
    # Higher barrier = harder to escape current state
    panel = panel.withColumn(
        "activation_energy_m",
        F.when(
            F.col("state_changed_m") == 1,
            F.abs(F.col("state_num") - F.lag("state_num", 1).over(w)) * F.col("bureau_owed_sum")
        ).otherwise(F.lit(0.0))
    )

    # Work done: force × distance (repayment effort)
    # Work = delever_amount × state_change
    panel = panel.withColumn(
        "work_done_m",
        F.coalesce(
            (F.lag("bureau_owed_sum", 1).over(w) - F.col("bureau_owed_sum")) *
            F.abs(F.col("state_num") - F.lag("state_num", 1).over(w)),
            F.lit(0.0)
        )
    )

    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        # Average potential energy (mean state severity)
        panel = panel.withColumn(
            f"avg_potential_energy_{m}m",
            F.avg("potential_energy_m").over(ww)
        )

        # Average kinetic energy (mean volatility)
        panel = panel.withColumn(
            f"avg_kinetic_energy_{m}m",
            F.avg("kinetic_energy_m").over(ww)
        )

        # Energy efficiency: work_done / total_energy (repayment effectiveness)
        panel = panel.withColumn(
            f"energy_efficiency_{m}m",
            F.when(
                F.sum("total_energy_m").over(ww) > 0,
                F.sum("work_done_m").over(ww) / F.sum("total_energy_m").over(ww)
            ).otherwise(F.lit(None))
        )

        # Power: energy / time (rate of energy change)
        panel = panel.withColumn(
            f"power_{m}m",
            (F.col("total_energy_m") - F.first("total_energy_m").over(ww)) / F.lit(float(m))
        )

        # Free energy: available capacity for deterioration
        # Gibbs free energy: G = H - TS (enthalpy - temp×entropy)
        panel = panel.withColumn(
            f"free_energy_{m}m",
            F.col("total_energy_m") - F.col(f"state_entropy_{m}m") * F.col("bureau_util")
        )

    return panel


# ============================================================================
# 3. THERMODYNAMICS (12 features)
# ============================================================================

def add_thermodynamics_features(panel: DataFrame, windows=[3, 6, 12]) -> DataFrame:
    """
    Temperature: volatility (how "hot" is the customer)
    Entropy: disorder in payment behavior
    Heat: energy transfer between lenders
    Phase transitions: solid → liquid → gas (S0 → S2 → S4)

    Physics: S = k ln(Ω), dS/dt ≥ 0, Q = mcΔT
    Bureau: temp = volatility, entropy = state disorder
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Temperature: DPD standard deviation (volatility = thermal motion)
    # Higher temp = more chaotic behavior
    for m in [3, 6, 12]:
        ww = w.rowsBetween(-m + 1, 0)
        panel = panel.withColumn(
            f"temperature_{m}m",
            F.coalesce(F.stddev_pop("bureau_max_dpd").over(ww), F.lit(0.0))
        )

    # Heat capacity: energy needed to change state by 1 unit
    # C = ΔQ / ΔT (how much stress to change 1 state level)
    panel = panel.withColumn(
        "heat_capacity_m",
        F.when(
            (F.col("state_num") != F.lag("state_num", 1).over(w)) &
            (F.abs(F.col("state_num") - F.lag("state_num", 1).over(w)) > 0),
            F.abs(F.col("total_energy_m") - F.lag("total_energy_m", 1).over(w)) /
            F.abs(F.col("state_num") - F.lag("state_num", 1).over(w))
        ).otherwise(F.lit(None))
    )

    # Entropy production rate: dS/dt (2nd law: always ≥ 0)
    for m in [3, 6]:
        panel = panel.withColumn(
            f"entropy_production_rate_{m}m",
            (F.col(f"state_entropy_{m}m") - F.lag(f"state_entropy_{m}m", 1).over(w)) / F.lit(1.0)
        )

    # Phase state: solid (S0), liquid (S1-S2), gas (S3-S4)
    panel = panel.withColumn(
        "phase_state_m",
        F.when(F.col("state") == "S0", F.lit("SOLID"))
         .when(F.col("state").isin(["S1", "S2"]), F.lit("LIQUID"))
         .when(F.col("state").isin(["S3", "S4"]), F.lit("GAS"))
         .otherwise(F.lit("UNKNOWN"))
    )

    # Latent heat: energy absorbed during phase transition (without temp change)
    panel = panel.withColumn(
        "latent_heat_m",
        F.when(
            F.col("phase_state_m") != F.lag("phase_state_m", 1).over(w),
            F.col("total_energy_m")
        ).otherwise(F.lit(0.0))
    )

    # Boltzmann probability: P(state) = exp(-E/kT)
    # Probability of being in high-energy (bad) state given temperature
    panel = panel.withColumn(
        "boltzmann_prob_6m",
        F.when(
            F.col("temperature_6m") > 0,
            F.exp(-F.col("potential_energy_m") / (F.lit(30.0) * F.col("temperature_6m")))
        ).otherwise(F.lit(None))
    )

    return panel


# ============================================================================
# 4. WAVE MECHANICS (10 features)
# ============================================================================

def add_wave_mechanics_features(panel: DataFrame, windows=[6, 12]) -> DataFrame:
    """
    Oscillations: periodic fluctuations in DPD
    Damping: decay of oscillation amplitude
    Resonance: external shocks amplifying internal oscillations
    Wave propagation: stress spreading across lenders

    Physics: y = A sin(ωt + φ), damping ∝ e^(-γt)
    Bureau: oscillation = DPD up/down cycles
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Oscillation detection: sign changes in dpd_velocity
    panel = panel.withColumn(
        "dpd_velocity_sign_m",
        F.when(F.col("dpd_velocity_m") > 0, 1)
         .when(F.col("dpd_velocity_m") < 0, -1)
         .otherwise(0)
    )

    panel = panel.withColumn(
        "oscillation_flag_m",
        (F.col("dpd_velocity_sign_m") != F.lag("dpd_velocity_sign_m", 1).over(w)).cast("int")
    )

    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        # Oscillation frequency: sign changes per period
        panel = panel.withColumn(
            f"oscillation_freq_{m}m",
            F.sum("oscillation_flag_m").over(ww) / F.lit(float(m))
        )

        # Amplitude: max deviation from mean
        panel = panel.withColumn(
            f"oscillation_amplitude_{m}m",
            F.max("bureau_max_dpd").over(ww) - F.min("bureau_max_dpd").over(ww)
        )

        # DPD Log Decay Rate: exponential decay/growth rate (RENAMED from damping_coeff)
        # Formula: ln(dpd_start / dpd_end) / months
        # Positive = DPD decaying (recovering), Negative = DPD growing (worsening)
        panel = panel.withColumn(
            f"dpd_log_decay_rate_{m}m",
            F.when(
                (F.first("bureau_max_dpd").over(ww) > 0) &
                (F.last("bureau_max_dpd").over(ww) > 0),
                F.log(F.first("bureau_max_dpd").over(ww) /
                      (F.last("bureau_max_dpd").over(ww) + F.lit(0.001))) / F.lit(float(m))
            ).otherwise(F.lit(None))
        )

        # Resonance: external shock (enquiry) amplifying internal oscillation
        # High if enquiry_spike coincides with high oscillation_amplitude
        panel = panel.withColumn(
            f"resonance_index_{m}m",
            F.corr("enq_cnt_m", "bureau_max_dpd").over(ww)
        )

    return panel


# ============================================================================
# 5. STRESS TENSOR (14 features)
# ============================================================================

def add_stress_tensor_features(panel: DataFrame, windows=[3, 6, 12]) -> DataFrame:
    """
    Multi-dimensional stress: not just DPD, but utilization, enquiries, lender diversity
    Tensor components: normal stress (perpendicular) + shear stress (parallel)
    Stress concentration: where stress accumulates

    Physics: σ = F/A, τ = shear force, von Mises stress
    Bureau: stress = weighted combination of risk factors
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Normal stress components (perpendicular to "surface")
    # σ_dpd: DPD stress (normalized 0-1)
    panel = panel.withColumn(
        "stress_dpd_m",
        F.col("bureau_max_dpd") / 180.0  # Normalize to S3 boundary
    )

    # σ_util: Utilization stress
    panel = panel.withColumn(
        "stress_util_m",
        F.coalesce(F.col("bureau_util"), F.lit(0.0))
    )

    # σ_enq: Enquiry stress (normalized by burst threshold)
    panel = panel.withColumn(
        "stress_enq_m",
        F.col("enq_cnt_m") / 3.0
    )

    # Hydrostatic stress: average of normal stresses (pressure)
    panel = panel.withColumn(
        "hydrostatic_stress_m",
        (F.col("stress_dpd_m") + F.col("stress_util_m") + F.col("stress_enq_m")) / 3.0
    )

    # Shear stress: CardX vs others divergence (parallel stress)
    panel = panel.withColumn(
        "shear_stress_cardx_others_m",
        F.abs(
            (F.col("bureau_cardx_max_dpd") / 180.0) -
            (F.col("bureau_others_max_dpd") / 180.0)
        )
    )

    # Von Mises stress: combined stress magnitude
    # σ_vm = √(σ₁² + σ₂² + σ₃² - σ₁σ₂ - σ₂σ₃ - σ₁σ₃)
    panel = panel.withColumn(
        "von_mises_stress_m",
        F.sqrt(
            F.pow(F.col("stress_dpd_m"), 2) +
            F.pow(F.col("stress_util_m"), 2) +
            F.pow(F.col("stress_enq_m"), 2) -
            F.col("stress_dpd_m") * F.col("stress_util_m") -
            F.col("stress_util_m") * F.col("stress_enq_m") -
            F.col("stress_dpd_m") * F.col("stress_enq_m")
        )
    )

    # Stress concentration factor: stress / avg_stress
    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        panel = panel.withColumn(
            f"stress_concentration_{m}m",
            F.when(
                F.avg("von_mises_stress_m").over(ww) > 0,
                F.col("von_mises_stress_m") / F.avg("von_mises_stress_m").over(ww)
            ).otherwise(F.lit(1.0))
        )

        # Stress gradient: rate of stress change
        panel = panel.withColumn(
            f"stress_gradient_{m}m",
            (F.col("von_mises_stress_m") - F.first("von_mises_stress_m").over(ww)) / F.lit(float(m))
        )

        # Principal stress: eigenvalue (max stress direction)
        # Approximated as max of three normal stresses
        panel = panel.withColumn(
            f"principal_stress_{m}m",
            F.greatest(
                F.max("stress_dpd_m").over(ww),
                F.max("stress_util_m").over(ww),
                F.max("stress_enq_m").over(ww)
            )
        )

    return panel


# ============================================================================
# 6. CHAOS & ATTRACTORS (12 features)
# ============================================================================

def add_chaos_attractor_features(panel: DataFrame, windows=[6, 12]) -> DataFrame:
    """
    Lyapunov exponent: sensitivity to initial conditions (chaos)
    Strange attractors: recurring patterns in state space
    Fractal dimension: self-similarity in behavior
    Bifurcation: where behavior splits

    Physics: λ = lim (1/t) ln(|δZ(t)|/|δZ₀|)
    Bureau: chaos = unpredictable transitions
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        # Lyapunov exponent (approximate): ln(deviation_ratio) / time
        # Positive = chaotic, Negative = stable
        panel = panel.withColumn(
            f"lyapunov_exponent_{m}m",
            F.when(
                (F.first("bureau_max_dpd").over(ww) > 0) &
                (F.stddev_pop("bureau_max_dpd").over(ww) > 0),
                F.log(
                    (F.col("bureau_max_dpd") - F.avg("bureau_max_dpd").over(ww)) /
                    F.first("bureau_max_dpd").over(ww)
                ) / F.lit(float(m))
            ).otherwise(F.lit(None))
        )

        # Recurrence: how often returns to similar state
        # Count times when |DPD - lag_DPD| < threshold
        panel = panel.withColumn(
            f"recurrence_count_{m}m",
            F.sum(
                F.when(
                    F.abs(F.col("bureau_max_dpd") - F.lag("bureau_max_dpd", 1).over(w)) < 10,
                    1
                ).otherwise(0)
            ).over(ww)
        )

        # Strange attractor: non-periodic cycling through states
        # High entropy + low recurrence = strange attractor
        panel = panel.withColumn(
            f"strange_attractor_index_{m}m",
            F.col(f"state_entropy_{m}m") * (1.0 - F.col(f"recurrence_count_{m}m") / F.lit(float(m)))
        )

        # Correlation dimension (fractal): self-similarity measure
        # D₂ ≈ log(count_pairs) / log(radius)
        # Approximated using state distribution variance
        panel = panel.withColumn(
            f"fractal_dimension_{m}m",
            F.log(F.countDistinct("state").over(ww)) /
            F.log(F.variance("state_num").over(ww) + F.lit(1.0))
        )

        # DPD Z-score: standardized deviation from mean (RENAMED from bifurcation_index)
        # Measures how many standard deviations customer is from average
        panel = panel.withColumn(
            f"dpd_zscore_{m}m",
            F.abs(F.col("bureau_max_dpd") - F.avg("bureau_max_dpd").over(ww)) /
            (F.stddev_pop("bureau_max_dpd").over(ww) + F.lit(1.0))
        )

    return panel


# ============================================================================
# 7. NETWORK TOPOLOGY (15 features)
# ============================================================================

def add_network_topology_features(panel: DataFrame, windows=[6, 12]) -> DataFrame:
    """
    Lender network: customer is node, lenders are connections
    Centrality: how "central" is customer in lender network
    Clustering: do lenders group together
    Betweenness: customer as bridge between lender groups

    Physics: Graph theory, network science
    Bureau: lender ecosystem topology
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Node degree: number of lenders (connections)
    panel = panel.withColumn(
        "network_degree_m",
        F.col("bureau_member_cnt")
    )

    # Degree centrality: normalized degree
    panel = panel.withColumn(
        "degree_centrality_m",
        F.col("network_degree_m") / 20.0  # Normalize by typical max lenders
    )

    # Network density: actual connections / possible connections
    # For star topology: density = k / k(k-1) = 1/(k-1)
    panel = panel.withColumn(
        "network_density_m",
        F.when(
            F.col("network_degree_m") > 1,
            1.0 / (F.col("network_degree_m") - 1)
        ).otherwise(F.lit(0.0))
    )

    # Lender diversity index (Shannon): -Σ(p_i × log(p_i))
    # Already have this as bureau_member_cnt, but add weighted version
    # Using exposure share instead of equal weights
    panel = panel.withColumn(
        "lender_diversity_hhi_m",
        1.0 / F.col("bureau_member_cnt")  # Inverse as proxy for HHI
    )

    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        # Network growth rate: new lenders added
        panel = panel.withColumn(
            f"network_growth_rate_{m}m",
            (F.col("network_degree_m") - F.first("network_degree_m").over(ww)) /
            (F.first("network_degree_m").over(ww) + F.lit(1.0))
        )

        # Network stability: lender churn rate
        panel = panel.withColumn(
            f"network_churn_{m}m",
            F.stddev_pop("network_degree_m").over(ww) /
            (F.avg("network_degree_m").over(ww) + F.lit(1.0))
        )

        # Clustering coefficient (approximate): stress grouping
        # High if CardX and others both stressed (clustered stress)
        panel = panel.withColumn(
            f"clustering_coeff_{m}m",
            F.avg(
                F.when(
                    (F.col("cardx_stressed_m") == 1) & (F.col("others_stressed_m") == 1),
                    1.0
                ).otherwise(0.0)
            ).over(ww)
        )

        # Betweenness (proxy): simultaneous stress on CardX and others
        # Customer is "bridge" if stress flows through them
        panel = panel.withColumn(
            f"betweenness_proxy_{m}m",
            F.sum(
                F.when(
                    (F.lag("cardx_stressed_m", 1).over(w) == 0) &
                    (F.col("cardx_stressed_m") == 1) &
                    (F.col("others_stressed_m") == 1),
                    1
                ).otherwise(0)
            ).over(ww)
        )

        # Network resilience: ability to maintain connections under stress
        panel = panel.withColumn(
            f"network_resilience_{m}m",
            F.avg(
                F.when(
                    F.col("von_mises_stress_m") > 0.5,
                    F.col("network_degree_m")
                ).otherwise(F.lit(None))
            ).over(ww) / (F.avg("network_degree_m").over(ww) + F.lit(1.0))
        )

    return panel


# ============================================================================
# 8. FIELD THEORY (10 features)
# ============================================================================

def add_field_theory_features(panel: DataFrame, windows=[6, 12]) -> DataFrame:
    """
    Stress field: landscape of stress across lenders
    Potential field: "gravity" pulling toward bad states
    Flux: flow of stress between CardX and others
    Gradient: direction of steepest stress increase

    Physics: ∇φ, ∮E·dA = Q/ε₀, curl, divergence
    Bureau: stress as field, customer navigates landscape
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # NOTE: field_strength_m removed - duplicate of von_mises_stress_m from Stress Tensor
    # Both computed √(σ_dpd² + σ_util² + σ_enq²) - keeping von_mises_stress_m

    # Potential field: "height" in stress landscape
    # φ = state_num × balance (gravitational potential)
    panel = panel.withColumn(
        "potential_field_m",
        F.col("state_num") * F.col("bureau_owed_sum") / 1000000.0
    )

    # Field gradient: ∇φ (direction of steepest increase)
    panel = panel.withColumn(
        "field_gradient_m",
        F.coalesce(
            F.col("potential_field_m") - F.lag("potential_field_m", 1).over(w),
            F.lit(0.0)
        )
    )

    # Flux: flow of stress from CardX to others
    # Φ = ∮ E·dA (stress transfer rate)
    panel = panel.withColumn(
        "stress_flux_cardx_to_others_m",
        F.when(
            (F.lag("cardx_stressed_m", 1).over(w) == 1) &
            (F.col("others_stressed_m") == 1) &
            (F.lag("others_stressed_m", 1).over(w) == 0),
            F.col("bureau_cardx_max_dpd") / 30.0  # Normalized flux
        ).otherwise(F.lit(0.0))
    )

    # Divergence: ∇·E (source/sink of stress)
    # Positive = stress increasing (source), Negative = decreasing (sink)
    # Use von_mises_stress_m instead of removed field_strength_m
    panel = panel.withColumn(
        "field_divergence_m",
        (F.col("von_mises_stress_m") - F.lag("von_mises_stress_m", 1).over(w)) +
        (F.col("network_degree_m") - F.lag("network_degree_m", 1).over(w)) / 10.0
    )

    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        # Field line density: concentration of stress
        # Use von_mises_stress_m instead of removed field_strength_m
        panel = panel.withColumn(
            f"field_line_density_{m}m",
            F.sum("von_mises_stress_m").over(ww) / F.lit(float(m))
        )

        # Curl (rotation): ∇×E (cyclical stress patterns)
        # Approximated as correlation between DPD and utilization changes
        panel = panel.withColumn(
            f"field_curl_{m}m",
            F.coalesce(
                F.corr("dpd_velocity_m", F.col("bureau_util") - F.lag("bureau_util", 1).over(w)).over(ww),
                F.lit(0.0)
            )
        )

    return panel


# ============================================================================
# 9. PHASE TRANSITIONS (8 features)
# ============================================================================

def add_phase_transition_features(panel: DataFrame, windows=[6, 12]) -> DataFrame:
    """
    Critical point: where behavior fundamentally changes
    Order parameter: distinguishes phases
    Universality: scaling laws near critical point
    Hysteresis: path-dependent transitions

    Physics: Liquid-gas critical point, Ising model
    Bureau: transition from normal to distressed behavior
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Order parameter: distinguishes solid/liquid/gas phases
    # M = (count_normal - count_stressed) / total
    for m in windows:
        ww = w.rowsBetween(-m + 1, 0)

        panel = panel.withColumn(
            f"order_parameter_{m}m",
            (F.sum("is_normal_m").over(ww) - F.sum("is_stressed_m").over(ww)) / F.lit(float(m))
        )

        # Critical slowing down: slow response near transition
        # Relaxation time increases as you approach critical point
        panel = panel.withColumn(
            f"critical_slowing_down_{m}m",
            F.when(
                F.abs(F.col(f"order_parameter_{m}m")) < 0.2,  # Near critical point (M ≈ 0)
                F.col(f"damping_coeff_{m}m")
            ).otherwise(F.lit(None))
        )

        # Hysteresis: path-dependent transitions
        # Different behavior going S0→S4 vs S4→S0
        panel = panel.withColumn(
            f"hysteresis_index_{m}m",
            F.sum(
                F.when(
                    (F.col("phase_clean_to_stress_m") == 1) &
                    (F.sum("phase_stress_to_default_m").over(ww) > 0),
                    1
                ).otherwise(0)
            ).over(ww) / (F.sum("phase_clean_to_stress_m").over(ww) + F.lit(1.0))
        )

    return panel


# ============================================================================
# 10. RELATIVITY (6 features)
# ============================================================================

def add_relativity_features(panel: DataFrame) -> DataFrame:
    """
    Time dilation: stress makes time "slow down"
    Reference frames: CardX perspective vs bureau perspective
    Spacetime interval: invariant measure across reference frames
    Lorentz factor: velocity approaching "speed of light" (max DPD)

    Physics: γ = 1/√(1 - v²/c²), proper time τ
    Bureau: time perception changes under stress
    """
    w = Window.partitionBy("cust_id").orderBy("as_of_month")

    # Time dilation: stress makes months feel longer
    # γ = 1 + (stress_level)² (modified Lorentz factor)
    panel = panel.withColumn(
        "time_dilation_factor_m",
        1.0 + F.pow(F.col("von_mises_stress_m"), 2)
    )

    # Proper time: "felt" time in stress reference frame
    # τ = t / γ (time slows down under stress)
    panel = panel.withColumn(
        "proper_time_lag_m",
        1.0 / F.col("time_dilation_factor_m")
    )

    # Lorentz factor for velocity: γ_v = 1/√(1 - v²/c²)
    # c = 180 DPD (S3 boundary, "speed of light")
    # v = dpd_velocity_m
    panel = panel.withColumn(
        "lorentz_factor_velocity_m",
        F.when(
            F.abs(F.col("dpd_velocity_m")) < 180,
            1.0 / F.sqrt(1.0 - F.pow(F.col("dpd_velocity_m") / 180.0, 2))
        ).otherwise(F.lit(None))  # Undefined at v=c
    )

    # Spacetime interval: invariant across reference frames
    # s² = c²t² - x² (in special relativity)
    # Here: s² = (time)² - (stress_distance)²
    panel = panel.withColumn(
        "spacetime_interval_m",
        F.sqrt(
            F.pow(F.row_number().over(w), 2) -
            F.pow(F.col("von_mises_stress_m") * 100, 2)
        )
    )

    # Reference frame divergence: CardX vs bureau perspective difference
    # Large = significant disagreement on stress state
    panel = panel.withColumn(
        "reference_frame_divergence_m",
        F.abs(
            (F.col("bureau_cardx_max_dpd") / 180.0) -
            (F.col("bureau_others_max_dpd") / 180.0)
        )
    )

    return panel


# ============================================================================
# MASTER FUNCTION: Apply All Advanced Features
# ============================================================================

def add_advanced_behavioral_physics_features(panel: DataFrame) -> DataFrame:
    """
    Apply all 100+ advanced physics-inspired features to panel.

    This is the state-of-the-art feature layer on top of production pipeline.

    Args:
        panel: Output from production_pipeline.py

    Returns:
        DataFrame with 100+ new advanced physics features
    """
    print("\n🔬 Adding Advanced Behavioral Physics Features...")

    # 1. Momentum & Inertia (15 features)
    print("   1/10 Momentum & Inertia...")
    panel = add_momentum_inertia_features(panel)

    # 2. Energy Dynamics (18 features)
    print("   2/10 Energy Dynamics...")
    panel = add_energy_features(panel)

    # 3. Thermodynamics (12 features)
    print("   3/10 Thermodynamics...")
    panel = add_thermodynamics_features(panel)

    # 4. Wave Mechanics (10 features)
    print("   4/10 Wave Mechanics...")
    panel = add_wave_mechanics_features(panel)

    # 5. Stress Tensor (14 features)
    print("   5/10 Stress Tensor...")
    panel = add_stress_tensor_features(panel)

    # 6. Chaos & Attractors (12 features)
    print("   6/10 Chaos & Attractors...")
    panel = add_chaos_attractor_features(panel)

    # 7. Network Topology (15 features)
    print("   7/10 Network Topology...")
    panel = add_network_topology_features(panel)

    # 8. Field Theory (10 features)
    print("   8/10 Field Theory...")
    panel = add_field_theory_features(panel)

    # 9. Phase Transitions (8 features)
    print("   9/10 Phase Transitions...")
    panel = add_phase_transition_features(panel)

    # 10. Relativity (6 features)
    print("   10/10 Relativity...")
    panel = add_relativity_features(panel)

    print("✅ Advanced features complete! (~120 new features added)\n")

    return panel
