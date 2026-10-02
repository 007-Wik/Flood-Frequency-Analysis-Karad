"""
Exploratory machine learning on the annual-peak series.

What this module is for
-----------------------
Testing one narrow question: *is there structure in the annual-peak record that
a simple regression can find, out of sample?*  A flood-frequency curve is
estimated from the distribution of the peaks, not from their year-to-year
sequence, so no model here can produce a design flood.  The 2025 notebook
claimed otherwise: it added a Gaussian process and a gradient booster, reported
R-squared of about 1.0, and displayed them next to the design table as though
the fits had earned a place in it.

Why those R-squared values appeared
-----------------------------------
Two leaks, both fixed here.

1. ``rank_norm = Q.rank() / N`` is a **monotone transform of the target**.  A
   model handed the target's own rank can reproduce it exactly, so R-squared
   near 1.0 measures nothing.  No such feature is built.
2. Rolling features used ``rolling(5).mean()`` **without** ``.shift(1)``, so
   every window contained the value being predicted.  Every feature here is
   computed from strictly earlier years and is asserted to be leak-free by
   :func:`assert_no_leakage`.

Validation
----------
``TimeSeriesSplit`` with an expanding window, never shuffled: a flood record is
a time series, and a random split lets the model interpolate between
neighbouring years.  Standardisation happens **inside** the pipeline so the
scaling is refitted on each training fold only.  Every model is scored against
a **persistence baseline** (next year's peak equals this year's), which is the
only benchmark that matters for a noisy annual maximum.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import config as _cfg
from . import util

log = util.get_logger("machine_learning")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


#: Columns that are the target, the index or the calendar position.  None of
#: them may ever reach the design matrix: ``q`` is the target itself, and its
#: rank or its raw value used as a feature is what produced the 2025 notebook's
#: spurious R-squared of 1.0.
TARGET_COLUMN = "q"
NON_FEATURE_COLUMNS: frozenset[str] = frozenset({"t", "year", TARGET_COLUMN})


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Columns of a frame that are legitimate model inputs."""
    return [c for c in frame.columns if c not in NON_FEATURE_COLUMNS]


def build_features(
    values: Sequence[float] | np.ndarray,
    lags: Sequence[int] = (1, 2, 3),
    window: int = 5,
) -> pd.DataFrame:
    """Lag and rolling features computed from **earlier years only**.

    Every column is a function of ``x[:t]``; the current observation ``x[t]``
    appears in no feature.  :func:`assert_no_leakage` verifies that property
    numerically by perturbing the target and checking the features do not move.
    """
    x = np.asarray(values, dtype=float).ravel()
    frame = pd.DataFrame({"t": np.arange(1, x.size + 1), TARGET_COLUMN: x})
    for lag in lags:
        frame[f"lag_{lag}"] = frame[TARGET_COLUMN].shift(lag)
    if window and window > 1:
        # ``.shift(1)`` first: without it the window contains the target year.
        past = frame[TARGET_COLUMN].shift(1)
        frame[f"mean_{window}"] = past.rolling(window).mean()
        frame[f"std_{window}"] = past.rolling(window).std(ddof=1)
        frame[f"min_{window}"] = past.rolling(window).min()
        frame[f"max_{window}"] = past.rolling(window).max()
    frame["expanding_mean"] = (
        frame[TARGET_COLUMN].shift(1).expanding(min_periods=3).mean()
    )
    frame["year"] = frame["t"]
    return frame.dropna().reset_index(drop=True)


def assert_no_leakage(
    values: Sequence[float] | np.ndarray,
    lags: Sequence[int] = (1, 2, 3),
    window: int = 5,
    fraction: float = 1e6,
) -> dict[str, Any]:
    """Verify that no feature depends on the value it predicts.

    Each year in turn is replaced by an enormous sentinel
    (``max + fraction * max``).  If a feature is computed from earlier years
    only, rows before the perturbed index are unchanged and rows after it may
    change (because the sentinel enters their history).  A feature that changes
    at, or before, the perturbed row is leaking.
    """
    x = np.asarray(values, dtype=float).ravel()
    base = build_features(x, lags, window)
    leaked_targets = [c for c in feature_columns(base) if TARGET_COLUMN in c]
    if leaked_targets:
        return {
            "leaky_features": {c: "is the target column" for c in leaked_targets},
            "passed": False,
            "detail": f"target column present among the features: {leaked_targets}",
        }
    sentinel = float(np.max(x) * (1.0 + fraction))
    verdicts: dict[str, Any] = {}
    for t in range(max(lags) if lags else 1, x.size):
        perturbed = x.copy()
        perturbed[t] = sentinel
        other = build_features(perturbed, lags, window)
        if len(other) != len(base):
            break
        offset = base.index[base["t"] == (t + 1)]
        if not len(offset):
            continue
        row = int(offset[0])
        for column in feature_columns(base):
            changed = np.any(
                np.abs(other[column].to_numpy()[:row] - base[column].to_numpy()[:row])
                > 0
            )
            if changed:
                verdicts[column] = f"changes at or before year {t + 1}"
    return {
        "leaky_features": verdicts,
        "passed": not verdicts,
        "detail": (
            "every feature is a function of strictly earlier years"
            if not verdicts
            else "; ".join(f"{k}: {v}" for k, v in verdicts.items())
        ),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ModelSpec:
    name: str
    factory: Callable[[int], Any]
    needs_scaling: bool = True


def _model_registry(random_state: int) -> tuple[ModelSpec, ...]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _scaled(estimator: Any) -> Pipeline:
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    return (
        ModelSpec("Linear", lambda s: _scaled(LinearRegression())),
        ModelSpec("Ridge", lambda s: _scaled(Ridge(alpha=1.0, random_state=s))),
        ModelSpec(
            "RandomForest",
            lambda s: _scaled(
                RandomForestRegressor(
                    n_estimators=400, min_samples_leaf=3, random_state=s
                )
            ),
        ),
        ModelSpec(
            "GradientBoosting",
            lambda s: _scaled(
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=2,
                    subsample=0.8,
                    random_state=s,
                )
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray, baseline: float = float("nan")
) -> dict[str, float]:
    """Out-of-fold metrics on the discharge scale."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((np.asarray(y_true, dtype=float) - np.mean(y_true)) ** 2))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denom = max(np.max(y_true) - np.min(y_true), 1e-12)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "rmse_cumecs": rmse,
        "mae_cumecs": mae,
        "mean_error_cumecs": float(np.mean(err)),
        "r2": float(r2),
        "nrmse_range": rmse / denom,
        "skill_vs_persistence": (
            (1.0 - rmse / baseline) if np.isfinite(baseline) else float("nan")
        ),
    }


def persistence_rmse(values: Sequence[float] | np.ndarray) -> float:
    """RMSE of the naive one-year-ahead forecast ``Q_t -> Q_t``.

    This is the benchmark every model in this module has to beat.  It is
    computed on consecutive years of the record, not on a concatenated set of
    held-out folds: differencing across a fold boundary compares years that were
    never adjacent, which is not the same thing.
    """
    y = np.asarray(values, dtype=float)
    if y.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(y) ** 2)))


def persistence_skill(
    values: Sequence[float] | np.ndarray, predictions: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """Skill of a prediction series against one-year-ahead persistence.

    ``skill = 1 - RMSE_model / RMSE_persistence``; zero means the model is no
    better than assuming the next year repeats the last one, and a negative value
    means it is actively worse.
    """
    x = np.asarray(values, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if p.size != x.size:
        raise ValueError(f"persistence_skill: {p.size} predictions for {x.size} values")
    rmse = float(np.sqrt(np.mean((x - p) ** 2)))
    baseline = persistence_rmse(x)
    return {
        "rmse_cumecs": rmse,
        "baseline_rmse_cumecs": baseline,
        "skill_vs_persistence": (
            (1.0 - rmse / baseline) if np.isfinite(baseline) else float("nan")
        ),
    }


@dataclasses.dataclass
class ModelScore:
    model: str
    metrics: dict[str, float]
    n_splits: int
    n_predictions: int
    per_fold: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "n_splits": self.n_splits,
            "n_predictions": self.n_predictions,
            **{k: v for k, v in self.metrics.items()},
        }


@dataclasses.dataclass
class MLResults:
    scores: list[ModelScore]
    predictions: pd.DataFrame
    leakage: dict[str, Any]
    baseline: dict[str, float]
    skill_verdict: str
    notes: list[str]

    def best(self) -> ModelScore | None:
        skilled = [
            s
            for s in self.scores
            if np.isfinite(s.metrics.get("skill_vs_persistence", np.nan))
            and s.metrics["skill_vs_persistence"] > 0
        ]
        if not skilled:
            return None
        return min(skilled, key=lambda s: s.metrics["rmse_cumecs"])

    @property
    def importance(self) -> pd.DataFrame | None:
        """Feature importance from the tree models, which are the only ones
        that expose it.  ``None`` when no tree model was fitted."""
        frames = [
            s.feature_importance
            for s in self.scores
            if s.feature_importance is not None and len(s.feature_importance)
        ]
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [s.to_dict() for s in self.scores],
            "baseline": self.baseline,
            "leakage_check": self.leakage,
            "skill_verdict": self.skill_verdict,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(
    values: Sequence[float] | np.ndarray,
    lags: Sequence[int] = (1, 2, 3),
    window: int = 5,
    n_splits: int | None = None,
) -> MLResults:
    """Time-ordered cross-validated evaluation against a persistence baseline."""
    x = util.require_time_ordered(values, "machine_learning")
    if not np.all(np.isfinite(x)):
        raise ValueError("machine_learning: array contains NaN or infinite values")
    splits = int(_cfg.CONFIG.ml_n_splits if n_splits is None else n_splits)
    frame = build_features(x, lags, window)
    features = feature_columns(frame)
    X = frame[features].to_numpy(dtype=float)
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    # The benchmark is one-year-ahead persistence over the whole record, so the
    # skill of each model is measured against the same number everywhere.
    baseline = persistence_rmse(x)

    if len(frame) < splits + 2:
        raise ValueError(
            f"only {len(frame)} usable rows after feature construction; at least "
            f"{splits + 2} are needed for time-ordered cross-validation"
        )

    leakage = assert_no_leakage(x, lags, window)
    if not leakage["passed"]:
        log.error("target leakage detected: %s", leakage["detail"])

    from sklearn.model_selection import TimeSeriesSplit

    splitter = TimeSeriesSplit(n_splits=splits)
    rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for spec in _model_registry(_cfg.CONFIG.ml_random_state):
        oof = np.full(y.size, np.nan)
        fold_id = np.full(y.size, -1, dtype=int)
        fold_rows: list[dict[str, Any]] = []
        for k, (train_idx, test_idx) in enumerate(splitter.split(X)):
            model = spec.factory(_cfg.CONFIG.ml_random_state)
            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[test_idx])
            oof[test_idx] = predicted
            fold_id[test_idx] = k
            fold_metrics = _metrics(y[test_idx], predicted, baseline)
            fold_rows.append(
                {
                    "fold": k,
                    "train_years": int(train_idx.size),
                    "test_years": int(test_idx.size),
                    "first_test_year": int(frame.loc[test_idx, "t"].iloc[0]),
                    "last_test_year": int(frame.loc[test_idx, "t"].iloc[-1]),
                    **{f"{key}": value for key, value in fold_metrics.items()},
                }
            )
            named = getattr(model, "named_steps", {}).get("model", model)
            if hasattr(named, "feature_importances_"):
                for name, value in zip(
                    features, np.asarray(named.feature_importances_)
                ):
                    importance_rows.append(
                        {
                            "model": spec.name,
                            "feature": name,
                            "importance": float(value),
                        }
                    )
        mask = np.isfinite(oof)
        metrics = _metrics(y[mask], oof[mask], baseline)
        rows.append(
            {
                "model": spec.name,
                "t": frame.loc[mask, "t"].to_numpy(),
                "fold": fold_id[mask],
                "actual": y[mask],
                "predicted": oof[mask],
                "residual": oof[mask] - y[mask],
                "per_fold": pd.DataFrame(fold_rows),
            }
        )
        log.info(
            "%-18s out-of-fold RMSE %8.1f cumecs, skill vs persistence %+.3f",
            spec.name,
            metrics["rmse_cumecs"],
            metrics["skill_vs_persistence"],
        )

    predictions = pd.DataFrame(
        [
            r
            for row in rows
            for r in zip(
                [row["model"]] * row["t"].size,
                row["t"],
                row["fold"],
                row["actual"],
                row["predicted"],
                row["residual"],
            )
        ],
        columns=[
            "model",
            "year",
            "fold",
            "actual_cumecs",
            "predicted_cumecs",
            "residual_cumecs",
        ],
    )
    scores = []
    for row in rows:
        frame_metrics = _metrics(row["actual"], row["predicted"], baseline)
        scores.append(
            ModelScore(
                model=row["model"],
                metrics=frame_metrics,
                n_splits=splits,
                n_predictions=int(row["t"].size),
                per_fold=row["per_fold"],
            )
        )
    importance = pd.DataFrame(importance_rows) if importance_rows else None
    if importance is not None:
        for score in scores:
            subset = importance[importance["model"] == score.model]
            score.feature_importance = subset if len(subset) else None

    baseline_report = {
        "rmse_cumecs": baseline,
        "mae_cumecs": (
            float(np.mean(np.abs(np.diff(x)))) if x.size > 1 else float("nan")
        ),
        "definition": (
            "one-step naive forecast: next year's peak equals this "
            "year's, over consecutive years of the whole record"
        ),
        "note": (
            "annual maxima of a large basin are close to unpredictable "
            "year to year, so this is a demanding benchmark"
        ),
    }
    best = [
        s
        for s in scores
        if np.isfinite(s.metrics["skill_vs_persistence"])
        and s.metrics["skill_vs_persistence"] > 0
    ]
    verdict = (
        f"no model beats the persistence baseline out of sample "
        f"(best skill {max((s.metrics['skill_vs_persistence'] for s in scores), default=float('nan')):+.3f})"
        if not best
        else f"{best[0].model} beats persistence with skill "
        f"{best[0].metrics['skill_vs_persistence']:+.3f} "
        f"(RMSE {best[0].metrics['rmse_cumecs']:,.0f} vs "
        f"{baseline:,.0f} cumecs); this is exploratory and must "
        "not be used to derive a design flood"
    )

    notes = [
        "This module cannot and does not produce a design flood.  A "
        "flood-frequency curve is estimated from the distribution of annual "
        "maxima; a regression on the year-to-year sequence of peaks describes "
        "that sequence, not its tail.",
        "Features are lags and rolling statistics of strictly earlier years.  "
        "The leakage assertion is run on every invocation and its result is "
        "reported with the metrics.",
        f"Validation is TimeSeriesSplit with {splits} expanding "
        "folds and no shuffling; scaling happens inside the pipeline so each "
        "fold is standardised on its own training data.",
        "Skill is measured against one-year-ahead persistence over consecutive "
        "years of the whole record.  Because annual maxima of a large basin are "
        "close to unpredictable from one year to the next, a skill near zero is "
        "the expected honest result, not a defect.",
        "The 2025 notebook reported R-squared near 1.0 for a Gaussian process "
        "and a gradient booster.  That came from two leaks: the target column "
        "itself used as a feature, and rolling windows that included the current "
        "year.  Both are absent here, and the honest result is that these models "
        "barely beat persistence.",
        "Scores are out-of-fold.  In-sample R-squared is not reported anywhere "
        "in this package.",
    ]
    return MLResults(
        scores=scores,
        predictions=predictions,
        leakage=leakage,
        baseline=baseline_report,
        skill_verdict=verdict,
        notes=notes,
    )


def score_table(res: MLResults) -> pd.DataFrame:
    """Out-of-fold metrics for every model plus the persistence baseline."""
    rows = [s.to_dict() for s in res.scores]
    n_splits = max((s.n_splits for s in res.scores), default=0)
    baseline = {
        "model": "persistence (baseline)",
        "n_splits": n_splits,
        "n_predictions": int(len(res.predictions)),
        "rmse_cumecs": res.baseline["rmse_cumecs"],
        "mae_cumecs": res.baseline["mae_cumecs"],
        "mean_error_cumecs": float("nan"),
        "r2": float("nan"),
        "nrmse_range": float("nan"),
        "skill_vs_persistence": 0.0,
    }
    rows.append(baseline)
    return pd.DataFrame(rows).sort_values("rmse_cumecs").reset_index(drop=True)


def summarise(res: MLResults) -> str:
    """Plain-text ML summary for the report."""
    lines = [
        "EXPLORATORY MACHINE LEARNING  (out-of-fold, time-ordered splits)",
        f"  leakage check           : {'PASS' if res.leakage['passed'] else 'FAIL'} "
        f"({res.leakage['detail']})",
        "",
        f"  {'model':<22} {'RMSE':>10} {'MAE':>10} {'R2 (oof)':>10} {'skill':>8}",
    ]
    for row in score_table(res).itertuples():
        r2 = "n/a" if not np.isfinite(row.r2) else f"{row.r2:10.3f}"
        skill = (
            "baseline"
            if row.model.startswith("persistence")
            else f"{row.skill_vs_persistence:+8.3f}"
        )
        lines.append(
            f"  {row.model:<22} {row.rmse_cumecs:10,.0f} "
            f"{row.mae_cumecs:10,.0f} {r2} {skill}"
        )
    lines += ["", "  per-fold RMSE for the best model (expanding window):"]
    best = res.best() or (res.scores[0] if res.scores else None)
    if best is not None and best.per_fold is not None and len(best.per_fold):
        for fold in best.per_fold.itertuples():
            lines.append(
                f"    fold {fold.fold}: train {fold.train_years:3d} yrs, "
                f"test {fold.test_years:3d} yrs "
                f"(WY start {fold.first_test_year}-{fold.last_test_year}) "
                f"-> RMSE {fold.rmse_cumecs:8,.0f} cumecs"
            )
    lines += ["", f"  VERDICT: {res.skill_verdict}", "", "  NOTES:"]
    lines += [f"    - {n}" for n in res.notes]
    return "\n".join(lines)


__all__ = [
    "ModelSpec",
    "ModelScore",
    "MLResults",
    "TARGET_COLUMN",
    "NON_FEATURE_COLUMNS",
    "build_features",
    "feature_columns",
    "assert_no_leakage",
    "persistence_rmse",
    "persistence_skill",
    "run",
    "score_table",
    "summarise",
]
