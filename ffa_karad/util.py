"""
Small shared utilities: deterministic RNG streams, output writers, logging.

Every stochastic routine in this package must obtain its generator from
:func:`stream_rng`, never from ``np.random.seed``.  That guarantees that
re-running a single stage reproduces its own numbers exactly, even after other
stages have consumed randomness.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import pathlib
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from . import config as _cfg

LOGGER_NAME = "ffa_karad"
_LOGGER_CONFIGURED = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger writing to stdout and ``outputs/logs/``."""
    global _LOGGER_CONFIGURED
    logger = logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
    if not _LOGGER_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        try:
            _cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(_cfg.LOG_DIR / "pipeline.log", encoding="utf-8")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
                )
            )
            logger.addHandler(fh)
        except OSError:  # read-only checkout: stdout only
            pass
        logger.setLevel(logging.INFO)
        logger.propagate = False
        _LOGGER_CONFIGURED = True
    return logger


def log_stage(name: str) -> logging.Logger:
    """Logger tagged with the pipeline stage currently executing."""
    return get_logger(name)


# ---------------------------------------------------------------------------
# Deterministic RNG streams
# ---------------------------------------------------------------------------


def stream_seed(name: str) -> int:
    """Stable 63-bit seed derived from ``MASTER_SEED`` and a stage label."""
    digest = hashlib.sha256(f"{_cfg.MASTER_SEED}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def stream_rng(name: str) -> np.random.Generator:
    """``numpy`` Generator seeded reproducibly for the named pipeline stage."""
    return np.random.default_rng(stream_seed(name))


# ---------------------------------------------------------------------------
# Filesystem / output
# ---------------------------------------------------------------------------


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_of(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars and containers to JSON types."""
    if isinstance(obj, Mapping):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "__dataclass_fields__"):
        return jsonable(dataclasses.asdict(obj))
    return obj


def provenance(stage: str, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Standard provenance block embedded next to every artefact."""
    block: dict[str, Any] = {
        "stage": stage,
        "generated_utc": _cfg.run_timestamp(),
        "package_version": __import__("ffa_karad").__version__,
        "station": {
            "id": _cfg.STATION.station_id,
            "name": _cfg.STATION.station_name,
            "river": _cfg.STATION.river,
            "catchment_area_km2": _cfg.STATION.catchment_area_km2,
            "zero_gauge_level_m": _cfg.STATION.zero_gauge_level_m,
        },
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import scipy

        block["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        pass
    if extra:
        block.update(extra)
    return block


def write_json(
    payload: Any, name: str, directory: pathlib.Path | None = None
) -> pathlib.Path:
    """Write ``payload`` as pretty JSON, creating the directory if needed."""
    directory = ensure_dir(directory or _cfg.OUTPUT_DIR)
    path = directory / f"{name}.json"
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=False), encoding="utf-8"
    )
    get_logger("io").info("wrote %s", path)
    return path


def write_table(
    df: pd.DataFrame,
    name: str,
    directory: pathlib.Path | None = None,
    stage: str = "table",
    extra: Mapping[str, Any] | None = None,
    float_format: str | None = None,
) -> pathlib.Path:
    """Write a result table to CSV with a JSON sidecar carrying provenance."""
    directory = ensure_dir(directory or _cfg.TABLE_DIR)
    path = directory / f"{name}.csv"
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        float_format=float_format or _cfg.CONFIG.float_format,
    )
    meta = provenance(stage, extra)
    meta["rows"] = int(len(df))
    meta["columns"] = [str(c) for c in df.columns]
    (directory / f"{name}.meta.json").write_text(
        json.dumps(jsonable(meta), indent=2), encoding="utf-8"
    )
    get_logger("io").info("wrote %s (%d rows)", path, len(df))
    return path


def write_workbook(
    sheets: Mapping[str, pd.DataFrame], name: str
) -> pathlib.Path | None:
    """Write a multi-sheet Excel workbook, or return ``None`` if openpyxl absent."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        get_logger("io").warning("openpyxl not installed; skipping workbook %s", name)
        return None
    directory = ensure_dir(_cfg.TABLE_DIR)
    path = directory / f"{name}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet[:31], index=False)
        pd.DataFrame(
            {"key": [f"{k} = {v}" for k, v in _flat_provenance().items()]}
        ).to_excel(writer, sheet_name="_provenance", index=False)
    get_logger("io").info("wrote %s", path)
    return path


def _flat_provenance() -> dict[str, str]:
    return {
        "station": f"{_cfg.STATION.station_id} {_cfg.STATION.station_name}",
        "river": _cfg.STATION.river,
        "record": f"{_cfg.STATION.record_start_wy}-{_cfg.STATION.record_end_wy}",
        "generated_utc": _cfg.run_timestamp(),
        "package_version": __import__("ffa_karad").__version__,
        "seed": str(_cfg.MASTER_SEED),
    }


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def finite(value: Any) -> float:
    """Coerce to ``float``, raising a clear error on non-finite input."""
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(
            f"non-finite value where a finite float was required: {value!r}"
        )
    return out


def as_float_array(values: Iterable[Any]) -> np.ndarray:
    """1-D float array with a finite check, sorted ascending."""
    arr = np.asarray(list(values), dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("empty array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("array contains NaN or infinite values")
    return np.sort(arr)


def require_time_ordered(values: Iterable[Any], where: str) -> np.ndarray:
    """Guard against order-dependent tests being fed a sorted series.

    ``DataBundle.q`` sorts ascending because most hydrologic estimators want
    order statistics.  Feeding that array to a trend, runs, autocorrelation or
    stationarity test yields a non-decreasing series, which is maximally
    autocorrelated: Mann-Kendall returns ``tau = 1.0``, ``S = n(n-1)/2``, every
    Ljung-Box lag rejects, and Pettitt's change point lands on the last index.
    These results look plausible enough to reach a report, so the guard is
    explicit and shared by every stage that consumes time order.

    The test is ``diff >= 0``, not ``diff > 0``: annual peaks contain ties, and a
    sorted series with ties is just as monotone as one without, so a strict
    comparison lets exactly the arrays this guard exists to catch slip through.
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size >= 4 and np.all(np.diff(x) >= 0):
        raise ValueError(
            f"{where} received a non-decreasing series of length {x.size}. "
            "That is the ascending order statistic, not the time order. Pass the "
            "water-year-ordered series (DataBundle.q_ordered / wl_ordered)."
        )
    return x


def describe_timer() -> "Timer":
    """Context manager that logs the wall-clock duration of a stage."""
    return Timer()


class Timer:
    """Context manager logging stage elapsed time."""

    def __init__(self) -> None:
        self.start: dt.datetime | None = None
        self.elapsed: float = 0.0
        self._logger: logging.Logger | None = None
        self._label = ""

    def __enter__(self) -> "Timer":
        self.start = dt.datetime.now()
        self._label = (
            f"timing {self._logger.name.split('.')[-1] if self._logger else ''}"
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.start is not None:
            self.elapsed = (dt.datetime.now() - self.start).total_seconds()
        logger = self._logger or get_logger("timer")
        logger.info("stage finished in %.1f s", self.elapsed)


__all__ = [
    "get_logger",
    "log_stage",
    "stream_seed",
    "stream_rng",
    "ensure_dir",
    "sha256_of",
    "jsonable",
    "provenance",
    "write_json",
    "write_table",
    "write_workbook",
    "finite",
    "as_float_array",
    "require_time_ordered",
    "describe_timer",
    "Timer",
]
