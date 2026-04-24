"""
Credit Score Diagnostics — Weighted & Unweighted
=================================================
Top-class performance evaluation for credit scoring and behavioural models.

Handles two critical credit-modelling scenarios:
  1. OVERSAMPLING: bads oversampled during training (e.g. 5% → 50% bad rate).
     sample_weight = inverse sampling probability per observation.
  2. REJECT INFERENCE: model trained on approved population only.
     sample_weight = augmented inference weights for rejected segment.

Metrics computed with and without sample_weight:
  - KS statistic + KS score threshold
  - AUC-ROC + Gini coefficient
  - Bad rate by score decile (monotonicity check)
  - PSI (train vs OOT population stability)
  - Calibration (predicted vs actual bad rate by band)

Plots (matplotlib, no external BI tool required):
  - KS separation plot (Good CDF / Bad CDF / separation shading)
  - ROC curve (weighted + unweighted overlay)
  - Score distribution (Good vs Bad density overlay)
  - Bad rate by score decile
  - Cumulative Gains / Lift chart
  - CAP curve (Cumulative Accuracy Profile)
  - PSI bar chart
  - Calibration plot

Usage:
    diag = CreditScoreDiagnostics(scores, labels, sample_weight=w)
    report = diag.full_report()
    diag.plot_all(save_dir="diagnostics/")
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, roc_curve

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

GOOD_COLOUR  = "#2196F3"   # blue
BAD_COLOUR   = "#F44336"   # red
KS_COLOUR    = "#4CAF50"   # green
WEIGHT_ALPHA = 0.65        # opacity for weighted overlays


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KSResult:
    ks_statistic: float
    ks_score_threshold: float   # score value at max separation
    ks_good_cdf_at_threshold: float
    ks_bad_cdf_at_threshold: float
    weighted: bool

    @property
    def gini(self) -> float:
        # Approximation: Gini ≈ 2*KS - 1 is NOT standard; stored separately
        return self.ks_statistic


@dataclass
class AUCResult:
    auc: float
    gini: float            # 2 * AUC - 1
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    weighted: bool


@dataclass
class DecileReport:
    decile: int
    score_min: float
    score_max: float
    n_total: int
    n_bad: int
    bad_rate: float
    cum_bad_capture: float
    lift: float
    weighted: bool


@dataclass
class PSIResult:
    psi: float
    band_psi: pd.DataFrame  # per-band PSI contributions
    interpretation: str     # GREEN / AMBER / RED


@dataclass
class DiagnosticsReport:
    ks_unweighted:       KSResult
    ks_weighted:         Optional[KSResult]
    auc_unweighted:      AUCResult
    auc_weighted:        Optional[AUCResult]
    decile_report:       pd.DataFrame
    decile_report_wtd:   Optional[pd.DataFrame]
    psi:                 Optional[PSIResult]
    calibration:         pd.DataFrame
    n_obs:               int
    n_bad:               int
    bad_rate_raw:        float
    bad_rate_weighted:   Optional[float]

    def summary(self) -> str:
        lines = [
            "═" * 60,
            "CREDIT SCORE DIAGNOSTICS SUMMARY",
            "═" * 60,
            f"  Observations      : {self.n_obs:,}",
            f"  Bads (raw)        : {self.n_bad:,}  ({self.bad_rate_raw:.2%})",
        ]
        if self.bad_rate_weighted is not None:
            lines.append(f"  Bads (weighted)   : {self.bad_rate_weighted:.2%}")
        lines += [
            "─" * 60,
            f"  KS (unweighted)   : {self.ks_unweighted.ks_statistic:.4f}  "
            f"@ score {self.ks_unweighted.ks_score_threshold:.4f}",
        ]
        if self.ks_weighted:
            lines.append(
                f"  KS (weighted)     : {self.ks_weighted.ks_statistic:.4f}  "
                f"@ score {self.ks_weighted.ks_score_threshold:.4f}"
            )
        lines += [
            f"  AUC (unweighted)  : {self.auc_unweighted.auc:.4f}  "
            f"Gini {self.auc_unweighted.gini:.4f}",
        ]
        if self.auc_weighted:
            lines.append(
                f"  AUC (weighted)    : {self.auc_weighted.auc:.4f}  "
                f"Gini {self.auc_weighted.gini:.4f}"
            )
        if self.psi:
            lines.append(f"  PSI               : {self.psi.psi:.4f}  [{self.psi.interpretation}]")
        lines.append("═" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTED METRICS (standalone, reusable)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedMetrics:
    """
    Standalone weighted metric computations.
    All methods accept optional sample_weight; if None, computes unweighted.
    """

    @staticmethod
    def ks(
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> KSResult:
        """
        KS statistic via weighted empirical CDFs.

        Weighted KS: replace equal-weight point mass per obs with
        weight_i / sum(weights) as mass → weighted ECDF.
        Handles oversampling and reject-inference weight scenarios.

        Args:
            scores:        Model scores (higher = riskier, or reverse — handled internally)
            labels:        Binary target (1 = bad/event, 0 = good/non-event)
            sample_weight: Per-observation weights. None = uniform.

        Returns:
            KSResult
        """
        scores  = np.asarray(scores,  dtype=float)
        labels  = np.asarray(labels,  dtype=int)
        weighted = sample_weight is not None

        if sample_weight is None:
            sample_weight = np.ones(len(scores))
        sample_weight = np.asarray(sample_weight, dtype=float)

        # Sort ascending by score
        idx     = np.argsort(scores)
        s       = scores[idx]
        l       = labels[idx]
        w       = sample_weight[idx]

        w_good = w * (1 - l)
        w_bad  = w * l

        sum_good = w_good.sum()
        sum_bad  = w_bad.sum()

        if sum_good == 0 or sum_bad == 0:
            raise ValueError("KS requires both good and bad observations.")

        cum_good = np.cumsum(w_good) / sum_good
        cum_bad  = np.cumsum(w_bad)  / sum_bad

        separation = np.abs(cum_good - cum_bad)
        max_idx    = np.argmax(separation)

        return KSResult(
            ks_statistic             = float(separation[max_idx]),
            ks_score_threshold       = float(s[max_idx]),
            ks_good_cdf_at_threshold = float(cum_good[max_idx]),
            ks_bad_cdf_at_threshold  = float(cum_bad[max_idx]),
            weighted                 = weighted,
        )

    @staticmethod
    def auc(
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> AUCResult:
        """
        AUC-ROC with optional sample weighting.

        sklearn roc_auc_score natively supports sample_weight — this wraps
        it with the full ROC curve and Gini coefficient.
        """
        scores  = np.asarray(scores,  dtype=float)
        labels  = np.asarray(labels,  dtype=int)
        weighted = sample_weight is not None

        auc_val = roc_auc_score(labels, scores, sample_weight=sample_weight)

        # For curve: use sample_weight in roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=sample_weight)

        return AUCResult(
            auc        = float(auc_val),
            gini       = float(2 * auc_val - 1),
            fpr        = fpr,
            tpr        = tpr,
            thresholds = thresholds,
            weighted   = weighted,
        )

    @staticmethod
    def bad_rate_by_decile(
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        n_deciles: int = 10,
    ) -> pd.DataFrame:
        """
        Bad rate by score decile — monotonicity diagnostic.

        Decile 1 = lowest score (best), Decile N = highest score (worst).
        For a well-calibrated model, bad rate should increase monotonically.
        """
        scores  = np.asarray(scores,  dtype=float)
        labels  = np.asarray(labels,  dtype=int)
        weighted = sample_weight is not None

        if sample_weight is None:
            sample_weight = np.ones(len(scores))
        sample_weight = np.asarray(sample_weight, dtype=float)

        # Assign decile based on score quantile
        decile_edges = np.percentile(scores, np.linspace(0, 100, n_deciles + 1))
        decile_edges[-1] += 1e-9   # include max value in last bin
        decile_idx = np.digitize(scores, decile_edges[1:], right=False)
        decile_idx = np.clip(decile_idx, 0, n_deciles - 1)

        rows = []
        total_wtd_bad = (sample_weight * labels).sum()
        cum_bad = 0.0

        for d in range(n_deciles):
            mask         = decile_idx == d
            w_d          = sample_weight[mask]
            l_d          = labels[mask]
            n_obs        = int(mask.sum())
            wtd_total    = w_d.sum()
            wtd_bad      = (w_d * l_d).sum()
            bad_rate     = wtd_bad / wtd_total if wtd_total > 0 else 0.0
            cum_bad     += wtd_bad
            bad_capture  = cum_bad / total_wtd_bad if total_wtd_bad > 0 else 0.0
            pop_share    = wtd_total / sample_weight.sum()
            lift         = bad_capture / ((d + 1) / n_deciles) if d < n_deciles else 1.0

            rows.append({
                "decile":          d + 1,
                "score_min":       float(scores[mask].min()) if n_obs > 0 else np.nan,
                "score_max":       float(scores[mask].max()) if n_obs > 0 else np.nan,
                "n_obs":           n_obs,
                "wtd_total":       round(wtd_total, 2),
                "wtd_bad":         round(wtd_bad, 2),
                "bad_rate":        round(bad_rate, 6),
                "cum_bad_capture": round(bad_capture, 6),
                "lift":            round(lift, 4),
                "pop_share":       round(pop_share, 6),
                "weighted":        weighted,
            })

        df = pd.DataFrame(rows)
        # Monotonicity flag: bad_rate should increase across deciles
        df["monotone_ok"] = df["bad_rate"].is_monotonic_increasing
        return df

    @staticmethod
    def psi(
        scores_train: np.ndarray,
        scores_oot: np.ndarray,
        n_bins: int = 10,
        epsilon: float = 1e-6,
    ) -> PSIResult:
        """
        Population Stability Index between training and OOT score distributions.

        PSI = sum[ (actual% - expected%) * ln(actual% / expected%) ]

        Interpretation:
          PSI < 0.10  → GREEN  : stable, no action needed
          PSI 0.10–0.25 → AMBER: minor shift, monitor
          PSI > 0.25  → RED   : significant shift, investigate / redevelop
        """
        bins = np.percentile(scores_train, np.linspace(0, 100, n_bins + 1))
        bins[0]  -= epsilon
        bins[-1] += epsilon

        train_counts, _ = np.histogram(scores_train, bins=bins)
        oot_counts,   _ = np.histogram(scores_oot,   bins=bins)

        train_pct = train_counts / train_counts.sum()
        oot_pct   = oot_counts   / oot_counts.sum()

        # Avoid log(0)
        train_pct = np.clip(train_pct, epsilon, None)
        oot_pct   = np.clip(oot_pct,   epsilon, None)

        band_psi  = (oot_pct - train_pct) * np.log(oot_pct / train_pct)
        total_psi = float(band_psi.sum())

        band_df = pd.DataFrame({
            "bin":         range(1, n_bins + 1),
            "score_low":   bins[:-1],
            "score_high":  bins[1:],
            "train_pct":   train_pct,
            "oot_pct":     oot_pct,
            "band_psi":    band_psi,
        })

        interp = ("GREEN"  if total_psi < 0.10 else
                  "AMBER"  if total_psi < 0.25 else "RED")

        return PSIResult(psi=total_psi, band_psi=band_df, interpretation=interp)

    @staticmethod
    def calibration(
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Calibration: predicted score (binned) vs actual weighted bad rate.
        Useful for checking if score is well-calibrated as a probability.
        """
        scores  = np.asarray(scores, dtype=float)
        labels  = np.asarray(labels, dtype=int)

        if sample_weight is None:
            sample_weight = np.ones(len(scores))
        sample_weight = np.asarray(sample_weight, dtype=float)

        bins = np.percentile(scores, np.linspace(0, 100, n_bins + 1))
        bins[0]  -= 1e-9
        bins[-1] += 1e-9
        bin_idx = np.digitize(scores, bins[1:], right=False)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        rows = []
        for b in range(n_bins):
            mask      = bin_idx == b
            w_b       = sample_weight[mask]
            l_b       = labels[mask]
            wtd_total = w_b.sum()
            avg_score = float((scores[mask] * w_b).sum() / wtd_total) if wtd_total > 0 else np.nan
            actual_br = float((w_b * l_b).sum() / wtd_total) if wtd_total > 0 else np.nan
            rows.append({
                "bin":         b + 1,
                "avg_score":   avg_score,
                "actual_bad_rate": actual_br,
                "n_obs":       int(mask.sum()),
                "wtd_total":   round(wtd_total, 2),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DIAGNOSTICS CLASS
# ─────────────────────────────────────────────────────────────────────────────

class CreditScoreDiagnostics:
    """
    Full diagnostic suite for credit scoring behavioural models.

    Computes all metrics with and without sample_weight.
    Generates publication-quality plots for model review / regulatory submission.

    Args:
        scores:          Model output scores (float, higher = more risk)
        labels:          Binary target (1 = bad/event, 0 = good/non-event)
        sample_weight:   Per-observation weights for bias correction.
                         None = uniform (no oversampling / reject inference).
        scores_oot:      Optional OOT scores for PSI computation.
        model_name:      Label for plots and report headers.

    Example:
        diag = CreditScoreDiagnostics(
            scores=model.predict_proba(X_test)[:, 1],
            labels=y_test,
            sample_weight=w_test,   # inverse sampling weights
            scores_oot=model.predict_proba(X_oot)[:, 1],
            model_name="BehaviouralScore_v3"
        )
        report = diag.full_report()
        print(report.summary())
        diag.plot_all(save_dir="diagnostics/")
    """

    def __init__(
        self,
        scores: Union[np.ndarray, pd.Series],
        labels: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
        scores_oot: Optional[Union[np.ndarray, pd.Series]] = None,
        model_name: str = "CreditScore",
        n_deciles: int = 10,
    ):
        self.scores       = np.asarray(scores,  dtype=float)
        self.labels       = np.asarray(labels,  dtype=int)
        self.weights      = (np.asarray(sample_weight, dtype=float)
                             if sample_weight is not None else None)
        self.scores_oot   = (np.asarray(scores_oot, dtype=float)
                             if scores_oot is not None else None)
        self.model_name   = model_name
        self.n_deciles    = n_deciles

        self._validate()

    def _validate(self) -> None:
        n = len(self.scores)
        assert len(self.labels) == n, "scores and labels must have same length"
        if self.weights is not None:
            assert len(self.weights) == n, "sample_weight must have same length as scores"
            assert (self.weights > 0).all(), "sample_weight must be strictly positive"
        unique = np.unique(self.labels)
        assert set(unique).issubset({0, 1}), "labels must be binary (0/1)"

    # ── METRICS ──────────────────────────────────────────────────────────────

    def compute_ks(self) -> Tuple[KSResult, Optional[KSResult]]:
        """Returns (unweighted_ks, weighted_ks). weighted_ks is None if no weights."""
        ks_unwt = WeightedMetrics.ks(self.scores, self.labels)
        ks_wt   = (WeightedMetrics.ks(self.scores, self.labels, self.weights)
                   if self.weights is not None else None)
        return ks_unwt, ks_wt

    def compute_auc(self) -> Tuple[AUCResult, Optional[AUCResult]]:
        """Returns (unweighted_auc, weighted_auc). weighted_auc is None if no weights."""
        auc_unwt = WeightedMetrics.auc(self.scores, self.labels)
        auc_wt   = (WeightedMetrics.auc(self.scores, self.labels, self.weights)
                    if self.weights is not None else None)
        return auc_unwt, auc_wt

    def compute_deciles(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Returns (unweighted_deciles, weighted_deciles)."""
        dec_unwt = WeightedMetrics.bad_rate_by_decile(
            self.scores, self.labels, n_deciles=self.n_deciles)
        dec_wt   = (WeightedMetrics.bad_rate_by_decile(
                        self.scores, self.labels, self.weights, self.n_deciles)
                    if self.weights is not None else None)
        return dec_unwt, dec_wt

    def compute_psi(self) -> Optional[PSIResult]:
        """PSI between training scores and OOT scores. None if no OOT scores."""
        if self.scores_oot is None:
            return None
        return WeightedMetrics.psi(self.scores, self.scores_oot)

    def compute_calibration(self) -> pd.DataFrame:
        """Calibration using weights if available."""
        return WeightedMetrics.calibration(self.scores, self.labels, self.weights)

    def full_report(self) -> DiagnosticsReport:
        """Run all diagnostics and return consolidated report."""
        ks_unwt, ks_wt     = self.compute_ks()
        auc_unwt, auc_wt   = self.compute_auc()
        dec_unwt, dec_wt   = self.compute_deciles()
        psi                = self.compute_psi()
        calibration        = self.compute_calibration()

        w_bad_rate = None
        if self.weights is not None:
            w_bad_rate = float((self.weights * self.labels).sum() / self.weights.sum())

        return DiagnosticsReport(
            ks_unweighted     = ks_unwt,
            ks_weighted       = ks_wt,
            auc_unweighted    = auc_unwt,
            auc_weighted      = auc_wt,
            decile_report     = dec_unwt,
            decile_report_wtd = dec_wt,
            psi               = psi,
            calibration       = calibration,
            n_obs             = len(self.scores),
            n_bad             = int(self.labels.sum()),
            bad_rate_raw      = float(self.labels.mean()),
            bad_rate_weighted = w_bad_rate,
        )

    # ── PLOTS ─────────────────────────────────────────────────────────────────

    def plot_ks(self, ax=None, weighted: bool = False) -> None:
        """KS separation plot: Good CDF / Bad CDF / shaded separation."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        scores  = self.scores
        labels  = self.labels
        weights = self.weights if weighted else None

        if weights is None:
            weights = np.ones(len(scores))

        idx     = np.argsort(scores)
        s       = scores[idx]
        l       = labels[idx]
        w       = weights[idx]

        w_good = w * (1 - l);  w_bad = w * l
        cum_good = np.cumsum(w_good) / w_good.sum()
        cum_bad  = np.cumsum(w_bad)  / w_bad.sum()
        sep      = np.abs(cum_good - cum_bad)
        max_idx  = np.argmax(sep)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        ax.plot(s, cum_good, color=GOOD_COLOUR, lw=2, label="Good CDF")
        ax.plot(s, cum_bad,  color=BAD_COLOUR,  lw=2, label="Bad CDF")
        ax.fill_between(s, cum_good, cum_bad,
                        where=(cum_bad >= cum_good),
                        alpha=0.15, color=KS_COLOUR, label="Separation")
        ax.axvline(s[max_idx], color=KS_COLOUR, lw=1.5, ls="--",
                   label=f"KS={sep[max_idx]:.4f} @ {s[max_idx]:.4f}")

        tag = "Weighted" if weighted else "Unweighted"
        ax.set_title(f"{self.model_name} — KS Plot ({tag})", fontsize=12)
        ax.set_xlabel("Score"); ax.set_ylabel("Cumulative %")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)

    def plot_roc(self, ax=None) -> None:
        """ROC curve — unweighted and weighted overlay if weights present."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        auc_unwt, auc_wt = self.compute_auc()

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        ax.plot(auc_unwt.fpr, auc_unwt.tpr, color=GOOD_COLOUR, lw=2,
                label=f"Unweighted AUC={auc_unwt.auc:.4f}  Gini={auc_unwt.gini:.4f}")
        if auc_wt is not None:
            ax.plot(auc_wt.fpr, auc_wt.tpr, color=BAD_COLOUR, lw=2,
                    ls="--", alpha=WEIGHT_ALPHA,
                    label=f"Weighted   AUC={auc_wt.auc:.4f}  Gini={auc_wt.gini:.4f}")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_title(f"{self.model_name} — ROC Curve", fontsize=12)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    def plot_score_distribution(self, ax=None) -> None:
        """Score distribution: Good vs Bad density overlay."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        good_scores = self.scores[self.labels == 0]
        bad_scores  = self.scores[self.labels == 1]

        bins = np.linspace(self.scores.min(), self.scores.max(), 50)
        ax.hist(good_scores, bins=bins, alpha=0.5, color=GOOD_COLOUR,
                density=True, label=f"Good  (n={len(good_scores):,})")
        ax.hist(bad_scores,  bins=bins, alpha=0.5, color=BAD_COLOUR,
                density=True, label=f"Bad   (n={len(bad_scores):,})")

        ax.set_title(f"{self.model_name} — Score Distribution", fontsize=12)
        ax.set_xlabel("Score"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(alpha=0.3)

    def plot_bad_rate_by_decile(self, ax=None, weighted: bool = False) -> None:
        """Bad rate by score decile — monotonicity diagnostic."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        dec_unwt, dec_wt = self.compute_deciles()
        df = dec_wt if (weighted and dec_wt is not None) else dec_unwt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 5))

        colours = [BAD_COLOUR if not row["monotone_ok"] else GOOD_COLOUR
                   for _, row in df.iterrows()]
        ax.bar(df["decile"], df["bad_rate"] * 100, color=colours, alpha=0.8)
        ax.set_xticks(df["decile"])
        ax.set_xticklabels([f"D{d}" for d in df["decile"]])

        tag = "Weighted" if weighted else "Unweighted"
        ax.set_title(f"{self.model_name} — Bad Rate by Decile ({tag})", fontsize=12)
        ax.set_xlabel("Decile (1=lowest score)"); ax.set_ylabel("Bad Rate %")
        ax.grid(axis="y", alpha=0.3)

        # Annotate monotonicity breaks
        for _, row in df.iterrows():
            if not row["monotone_ok"]:
                ax.annotate("⚠", (row["decile"], row["bad_rate"] * 100),
                            ha="center", va="bottom", fontsize=10, color="orange")

    def plot_gains_lift(self, ax=None) -> None:
        """Cumulative Gains and Lift chart."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        dec_unwt, dec_wt = self.compute_deciles()

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            axes = [ax, ax]

        for i, (df, tag) in enumerate([(dec_unwt, "Unweighted"),
                                        (dec_wt,   "Weighted")]):
            if df is None:
                continue
            a = axes[i]
            a.plot(df["decile"], df["cum_bad_capture"] * 100,
                   color=GOOD_COLOUR, lw=2, marker="o", ms=4,
                   label="Model")
            a.plot([1, self.n_deciles], [100 / self.n_deciles, 100],
                   "k--", lw=1, label="Random")
            a.set_title(f"{self.model_name} — Cumulative Gains ({tag})", fontsize=11)
            a.set_xlabel("Decile"); a.set_ylabel("Cumulative Bad Capture %")
            a.set_xticks(df["decile"]); a.legend(); a.grid(alpha=0.3)

        if ax is None:
            plt.tight_layout()

    def plot_cap(self, ax=None) -> None:
        """CAP (Cumulative Accuracy Profile) curve."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        idx      = np.argsort(-self.scores)   # descending: high risk first
        cum_pop  = np.arange(1, len(self.scores) + 1) / len(self.scores)
        cum_bad  = np.cumsum(self.labels[idx]) / self.labels.sum()

        # Perfect model
        bad_rate = self.labels.mean()
        cap_x    = [0, bad_rate, 1]
        cap_y    = [0, 1.0, 1.0]

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        ax.plot(cum_pop, cum_bad, color=GOOD_COLOUR, lw=2, label="Model")
        ax.plot(cap_x, cap_y, color=KS_COLOUR, lw=1.5, ls="--", label="Perfect")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

        # Accuracy Ratio = area between model and random / area between perfect and random
        ar = (np.trapz(cum_bad, cum_pop) - 0.5) / (1 - 0.5 * bad_rate - 0.5)
        ax.set_title(f"{self.model_name} — CAP Curve  (AR={ar:.4f})", fontsize=12)
        ax.set_xlabel("Population %"); ax.set_ylabel("% Bad Captured")
        ax.legend(); ax.grid(alpha=0.3)

    def plot_psi(self, ax=None) -> None:
        """PSI bar chart — train vs OOT distribution shift."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        psi_result = self.compute_psi()
        if psi_result is None:
            warnings.warn("No OOT scores provided; PSI plot skipped.")
            return

        df = psi_result.band_psi
        colour = {"GREEN": KS_COLOUR, "AMBER": "orange", "RED": BAD_COLOUR}[
            psi_result.interpretation]

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        ax.bar(df["bin"], df["band_psi"], color=colour, alpha=0.8)
        ax.axhline(psi_result.psi / len(df), color="k", ls="--", lw=1)
        ax.set_title(
            f"{self.model_name} — PSI={psi_result.psi:.4f} [{psi_result.interpretation}]",
            fontsize=12)
        ax.set_xlabel("Score Band"); ax.set_ylabel("Band PSI Contribution")
        ax.grid(axis="y", alpha=0.3)

    def plot_calibration(self, ax=None) -> None:
        """Calibration: predicted score vs actual bad rate."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        cal = self.compute_calibration().dropna()

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(cal["avg_score"], cal["actual_bad_rate"],
                   color=BAD_COLOUR, s=80, zorder=3, label="Bin actual bad rate")
        ax.plot(cal["avg_score"], cal["avg_score"],
                "k--", lw=1, label="Perfect calibration")

        from numpy.polynomial.polynomial import polyfit
        if len(cal) >= 2:
            c = polyfit(cal["avg_score"], cal["actual_bad_rate"], 1)
            x_range = np.linspace(cal["avg_score"].min(), cal["avg_score"].max(), 50)
            ax.plot(x_range, c[0] + c[1] * x_range, color=GOOD_COLOUR, lw=1.5,
                    label="Fitted trend")

        ax.set_title(f"{self.model_name} — Calibration Plot", fontsize=12)
        ax.set_xlabel("Mean Predicted Score"); ax.set_ylabel("Actual Bad Rate")
        ax.legend(); ax.grid(alpha=0.3)

    def plot_all(self, save_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Generate all 8 diagnostic plots on a single A3-style grid.

        Args:
            save_dir: Directory to save PNG. If None, shows interactively.
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plots")

        has_weights = self.weights is not None
        has_oot     = self.scores_oot is not None

        fig = plt.figure(figsize=(22, 26))
        fig.suptitle(f"{self.model_name} — Credit Score Diagnostics",
                     fontsize=16, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

        # Row 0
        ax_ks_unwt  = fig.add_subplot(gs[0, 0])
        ax_ks_wt    = fig.add_subplot(gs[0, 1])
        ax_roc      = fig.add_subplot(gs[0, 2])
        # Row 1
        ax_dist     = fig.add_subplot(gs[1, 0])
        ax_dec_unwt = fig.add_subplot(gs[1, 1])
        ax_dec_wt   = fig.add_subplot(gs[1, 2])
        # Row 2
        ax_gains    = fig.add_subplot(gs[2, 0])
        ax_cap      = fig.add_subplot(gs[2, 1])
        ax_cal      = fig.add_subplot(gs[2, 2])
        # Row 3
        ax_psi      = fig.add_subplot(gs[3, :])

        self.plot_ks(ax=ax_ks_unwt, weighted=False)
        if has_weights:
            self.plot_ks(ax=ax_ks_wt, weighted=True)
        else:
            ax_ks_wt.text(0.5, 0.5, "No sample_weight\nprovided",
                          ha="center", va="center", transform=ax_ks_wt.transAxes,
                          fontsize=11, color="grey")
            ax_ks_wt.set_title("KS Plot (Weighted) — N/A")

        self.plot_roc(ax=ax_roc)
        self.plot_score_distribution(ax=ax_dist)
        self.plot_bad_rate_by_decile(ax=ax_dec_unwt, weighted=False)
        self.plot_bad_rate_by_decile(ax=ax_dec_wt, weighted=has_weights)
        self.plot_gains_lift(ax=ax_gains)
        self.plot_cap(ax=ax_cap)
        self.plot_calibration(ax=ax_cal)

        if has_oot:
            self.plot_psi(ax=ax_psi)
        else:
            ax_psi.text(0.5, 0.5, "No OOT scores provided — PSI not computed.\n"
                        "Pass scores_oot= to CreditScoreDiagnostics.",
                        ha="center", va="center", transform=ax_psi.transAxes,
                        fontsize=12, color="grey")
            ax_psi.set_title("PSI — N/A")

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{self.model_name}_diagnostics.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Diagnostics saved → {path}")
        else:
            plt.show()

        plt.close(fig)
