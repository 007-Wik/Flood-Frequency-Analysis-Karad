"""
Data acquisition and tidying for the Karad annual peak series.

Responsibilities
----------------
* Read the supplied station sheet, or fall back to the verbatim embedded record
  when the CSV is unavailable (reproducibility without a data file).
* Produce one tidy row per **water year**, with the water-year *label taken
  verbatim from the source*.  The 2025 notebook re-derived the label from a
  right-truncated year string, which turned ``1999-2000`` into ``1999-00`` and
  silently corrupted the time axis and every autocorrelation test built on it.
* Attach non-exceedance probabilities under three plotting positions.
* Expose a :class:`DataBundle` that every downstream stage consumes.  No
  downstream stage re-reads a file.

The "Mode" (most frequent 3-year mean) field in the 2025 report is
deliberately *not* reproduced.  A 3-year running mean of 57 values is
unimodal for *any* log-symmetric or right-skewed record, so the statistic
carries no information; the 2025 value of 1175.33 cumecs was an artefact of
pandas ``idxmax`` on ties.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from . import config as _cfg
from . import util

log = util.get_logger("data_processing")

#: Monsoon / non-monsoon grouping of the peak month, used by the seasonality QC.
_SOUTHWEST_MONTHS = (6, 7, 8, 9)
_POSTMONSOON_MONTHS = (10, 11, 12)

#: Verbatim copy of the 57 annual instantaneous peaks as supplied.  Kept in the
#: repository so the analysis is reproducible from a clean checkout.  Provenance
#: and the derived-discharge caveat live in :data:`ffa_karad.config.STATION`.
EMBEDDED_RECORD: tuple[tuple[str, float, str, str, int], ...] = (
    ("1965-1966", 562.657, "21-07-1965", "00:00:00", 4186),
    ("1966-1967", 561.042, "29-07-1966", "21:00:00", 3302),
    ("1967-1968", 564.112, "28-07-1967", "02:00:00", 5067),
    ("1968-1969", 556.682, "07-08-1968", "14:00:00", 1414),
    ("1969-1970", 561.682, "01-08-1969", "09:00:00", 3641),
    ("1970-1971", 558.242, "21-08-1970", "12:00:00", 2006),
    ("1971-1972", 558.097, "31-08-1971", "09:00:00", 1947),
    ("1972-1973", 558.762, "07-07-1972", "22:00:00", 2224),
    ("1973-1974", 560.442, "08-07-1973", "07:00:00", 2999),
    ("1974-1975", 559.377, "05-07-1974", "13:00:00", 2496),
    ("1975-1976", 561.442, "10-07-1975", "17:00:00", 3512),
    ("1976-1977", 567.162, "07-06-1976", "03:00:00", 7177),
    ("1977-1978", 559.912, "07-07-1977", "15:00:00", 2743),
    ("1978-1979", 558.402, "02-09-1978", "16:00:00", 2072),
    ("1979-1980", 559.462, "08-08-1979", "18:00:00", 2534),
    ("1980-1981", 560.562, "05-08-1980", "21:00:00", 3059),
    ("1981-1982", 558.632, "10-07-1981", "02:00:00", 2169),
    ("1982-1983", 557.392, "29-07-1982", "16:00:00", 1672),
    ("1983-1984", 558.982, "17-08-1983", "06:00:00", 2320),
    ("1984-1985", 559.082, "18-07-1984", "20:00:00", 2364),
    ("1985-1986", 557.317, "02-08-1985", "07:00:00", 1644),
    ("1986-1987", 559.062, "30-06-1986", "02:00:00", 2355),
    ("1987-1988", 557.472, "08-07-1987", "14:00:00", 1702),
    ("1988-1989", 563.212, "03-08-1988", "05:00:00", 4513),
    ("1989-1990", 559.722, "25-07-1989", "13:00:00", 2654),
    ("1990-1991", 562.312, "18-08-1990", "12:00:00", 3989),
    ("1991-1992", 561.147, "28-07-1991", "07:00:00", 3357),
    ("1992-1993", 559.772, "14-08-1992", "10:00:00", 2677),
    ("1993-1994", 559.402, "19-06-1993", "15:00:00", 2507),
    ("1994-1995", 562.252, "15-07-1994", "13:00:00", 3955),
    ("1995-1996", 556.892, "20-07-1995", "15:00:00", 1488),
    ("1996-1997", 561.512, "04-10-1996", "13:00:00", 3549),
    ("1997-1998", 563.922, "24-08-1997", "08:00:00", 4948),
    ("1998-1999", 558.182, "09-07-1998", "06:00:00", 1982),
    ("1999-2000", 558.742, "28-07-1999", "07:00:00", 2216),
    ("2000-2001", 555.662, "12-07-2000", "22:00:00", 1077),
    ("2001-2002", 556.342, "09-07-2001", "18:00:00", 1297),
    ("2002-2003", 555.582, "07-09-2002", "08:00:00", 1052),
    ("2003-2004", 554.912, "28-07-2003", "08:00:00", 855),
    ("2004-2005", 562.395, "12-08-2004", "23:00:00", 4036),
    ("2005-2006", 565.045, "02-08-2005", "23:00:00", 5675),
    ("2006-2007", 566.305, "30-07-2006", "13:00:00", 6548),
    ("2007-2008", 561.465, "02-07-2008", "08:00:00", 3524),
    ("2008-2009", 560.115, "11-08-2009", "09:00:00", 2840),
    ("2009-2010", 555.775, "15-07-2009", "11:00:00", 1112),
    ("2010-2011", 556.195, "27-07-2010", "01:00:00", 1248),
    ("2011-2012", 563.485, "04-09-2011", "11:00:00", 4678),
    ("2012-2013", 556.235, "26-07-2012", "02:00:00", 1262),
    ("2013-2014", 559.235, "25-07-2013", "15:00:00", 2432),
    ("2014-2015", 556.865, "24-07-2014", "10:00:00", 1479),
    ("2015-2016", 556.345, "04-10-2015", "07:00:00", 1298),
    ("2016-2017", 557.085, "26-09-2016", "18:00:00", 1558),
    ("2017-2018", 557.545, "07-10-2017", "03:00:00", 1730),
    ("2018-2019", 557.245, "13-04-2018", "13:00:00", 1617),
    ("2019-2020", 566.145, "06-08-2019", "19:00:00", 6434),
    ("2020-2021", 559.505, "17-08-2020", "19:00:00", 2553),
    ("2021-2022", 565.485, "24-07-2021", "08:00:00", 6080),
)


# ---------------------------------------------------------------------------
# Plotting positions
# ---------------------------------------------------------------------------

#: Formulae for the non-exceedance probability of the m-th ascending order
#: statistic in a sample of N.  All four are textbook; Weibull is the Indian
#: default for annual-maxima FFA.
PLOTTING_POSITIONS: dict[str, Any] = {
    "weibull": lambda m, n: m / (n + 1.0),
    "hazen": lambda m, n: (m - 0.5) / n,
    "gringorten": lambda m, n: (m - 0.44) / n,
    "cunnane": lambda m, n: (m - 0.4) / (n + 0.2),
}

PlottingPosition = Literal["weibull", "hazen", "gringorten", "cunnane"]


def plotting_position(
    m: np.ndarray | float, n: int, method: PlottingPosition = "weibull"
) -> np.ndarray | float:
    """Non-exceedance probability for order statistic ``m`` of ``n``.

    Parameters
    ----------
    m:
        Ascending ranks, 1-based.
    n:
        Sample size.
    method:
        One of :data:`PLOTTING_POSITIONS`.
    """
    if method not in PLOTTING_POSITIONS:
        raise KeyError(
            f"unknown plotting position {method!r}; "
            f"expected one of {sorted(PLOTTING_POSITIONS)}"
        )
    if n < 2:
        raise ValueError("sample size must be at least 2")
    return PLOTTING_POSITIONS[method](np.asarray(m, dtype=float), float(n))


# ---------------------------------------------------------------------------
# Data bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DataBundle:
    """Everything downstream stages need, in one immutable-ish container."""

    frame: pd.DataFrame
    source: str
    source_sha256: str | None
    n_records: int

    @property
    def q(self) -> np.ndarray:
        """Annual peak discharge, **ascending**.

        Use this for order statistics, L-moments, information criteria and
        goodness-of-fit -- anything that depends only on the multiset of values.

        Do **not** use it for trend, runs, autocorrelation, stationarity or any
        other test that depends on the order of observation.  Feeding a sorted
        series to those tests yields a perfectly increasing sequence, which is
        maximally autocorrelated and will report significant trend in almost
        any record.  Use :attr:`q_ordered` there.
        """
        return util.as_float_array(self.frame["q_cumecs"])

    @property
    def q_ordered(self) -> np.ndarray:
        """Annual peak discharge in water-year order.

        This is the correct input for every time-series test in
        :mod:`ffa_karad.statistical_tests` and :mod:`ffa_karad.autocorrelation`.
        """
        return np.asarray(self.frame["q_cumecs"].to_numpy(), dtype=float)

    @property
    def wl(self) -> np.ndarray:
        """Peak water level, ascending."""
        return util.as_float_array(self.frame["wl_m"])

    @property
    def wl_ordered(self) -> np.ndarray:
        """Peak water level in water-year order."""
        return np.asarray(self.frame["wl_m"].to_numpy(), dtype=float)

    @property
    def log_q(self) -> np.ndarray:
        """Natural log of annual peak discharge, ascending."""
        return np.log(self.q)

    @property
    def years(self) -> pd.Index:
        """Water-year start years, ascending."""
        return pd.Index(self.frame["wy_start_year"].to_numpy())

    def summary(self) -> dict[str, Any]:
        """Descriptive statistics of the peak series."""
        q = self.q
        n = q.size
        mean = float(q.mean())
        sd = float(q.std(ddof=1))
        cv = sd / mean
        return {
            "n": int(n),
            "record_length_years": int(n),
            "min_cumecs": float(q.min()),
            "max_cumecs": float(q.max()),
            "mean_cumecs": mean,
            "median_cumecs": float(np.median(q)),
            "std_cumecs": sd,
            "cv": float(cv),
            "cv_percent": float(100 * cv),
            "log_mean": float(np.log(q).mean()),
            "log_std": float(np.log(q).std(ddof=1)),
            "range_ratio": float(q.max() / q.min()),
            "wy_start": int(self.years.min()),
            "wy_end": int(self.years.max()),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _normalise_columns(columns: list[str]) -> dict[str, str]:
    """Map the messy source headers onto canonical field names.

    The source sheet uses ``Peak WL (m)``, ``Recorded Date``, ``Time`` and
    ``Discharge* (cumecs)``.  The asterisk is significant -- see
    :data:`ffa_karad.config.STATION.discharge_is_derived`.
    """
    mapping: dict[str, str] = {}
    for raw in columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
        key = key.replace("discharge_cumecs", "q_cumecs")
        if key.startswith("sl_no") or key in {"sr_no", "sno"}:
            mapping[str(raw)] = "sl_no"
        elif key.startswith("year"):
            mapping[str(raw)] = "wy_label"
        elif "wl" in key or "level" in key:
            mapping[str(raw)] = "wl_m"
        elif "date" in key:
            mapping[str(raw)] = "recorded_date"
        elif key.startswith("time"):
            mapping[str(raw)] = "recorded_time"
        elif "discharge" in key or "cumec" in key or "q" == key:
            mapping[str(raw)] = "q_cumecs"
        else:
            mapping[str(raw)] = key
    return mapping


def load_raw_csv(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Read the raw station sheet, or return the embedded verbatim record.

    The SHA-256 of the CSV is computed and compared against
    :data:`ffa_karad.config.RAW_DATA_SHA256`.  A mismatch is reported as a
    warning, never a silent pass, so that an updated data file cannot be
    mistaken for the record analysed in the report.
    """
    path = pathlib.Path(path or (_cfg.DATA_DIR / _cfg.RAW_DATA_FILENAME))
    if not path.exists():
        log.warning("raw CSV not found at %s; using the embedded verbatim record", path)
        return pd.DataFrame(
            EMBEDDED_RECORD,
            columns=["wy_label", "wl_m", "recorded_date", "recorded_time", "q_cumecs"],
        )

    digest = util.sha256_of(path)
    if digest != _cfg.RAW_DATA_SHA256:
        log.warning(
            "raw CSV SHA-256 %s does not match the pinned %s -- the data file has "
            "changed. Re-pin config.RAW_DATA_SHA256 only after reviewing the change.",
            digest[:16],
            _cfg.RAW_DATA_SHA256[:16],
        )

    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    frame = frame.rename(columns=_normalise_columns(list(frame.columns)))
    required = {"wy_label", "wl_m", "recorded_date", "q_cumecs"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"raw CSV is missing required columns {sorted(missing)}; "
            f"found {sorted(frame.columns)}"
        )
    for column in ("wl_m", "q_cumecs"):
        frame[column] = pd.to_numeric(
            frame[column]
            .astype(str)
            .str.strip()
            .str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce",
        )
    if "recorded_time" not in frame.columns:
        frame["recorded_time"] = np.nan
    return frame


def _parse_wy_start(label: str) -> int:
    """Start year of a water-year label such as ``1999-2000``.

    The label is kept verbatim in the frame; only the numeric start year is
    derived, and only from the **first** four characters, which removes the
    right-truncation bug that produced ``"1999-00"`` in the 2025 notebook.
    """
    head = re.match(r"\s*(\d{4})", str(label))
    if not head:
        raise ValueError(f"cannot parse water-year label {label!r}")
    return int(head.group(1))


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy the raw sheet into the canonical analysis frame."""
    frame = raw.copy()
    frame["wy_label"] = frame["wy_label"].astype(str).str.strip()
    frame["wy_start_year"] = frame["wy_label"].map(_parse_wy_start)

    date_text = frame["recorded_date"].astype(str).str.strip()
    frame["peak_datetime"] = pd.to_datetime(
        date_text
        + " "
        + frame.get("recorded_time", pd.Series("", index=frame.index))
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .replace("", "00:00:00"),
        dayfirst=True,
        errors="coerce",
    )
    frame["peak_date"] = pd.to_datetime(date_text, dayfirst=True, errors="coerce")
    frame["peak_month"] = frame["peak_date"].dt.month
    frame["peak_doy"] = frame["peak_date"].dt.dayofyear
    frame["season"] = np.select(
        [
            frame["peak_month"].isin(_SOUTHWEST_MONTHS),
            frame["peak_month"].isin(_POSTMONSOON_MONTHS),
            frame["peak_month"].isin((3, 4, 5)),
        ],
        ["southwest_monsoon", "post_monsoon", "pre_monsoon"],
        default="winter",
    )
    frame["is_monsoon"] = frame["season"].eq("southwest_monsoon")
    frame["depth_over_zero_gauge_m"] = frame["wl_m"] - _cfg.STATION.zero_gauge_level_m

    frame = frame.sort_values("wy_start_year").reset_index(drop=True)
    if frame["wy_start_year"].duplicated().any():
        dupes = frame.loc[frame["wy_start_year"].duplicated(), "wy_label"].tolist()
        raise ValueError(f"duplicate water years in the source record: {dupes}")

    return frame


def add_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ascending rank and non-exceedance probability under each position."""
    out = frame.copy()
    n = len(out)
    out["rank_asc"] = np.arange(1, n + 1)
    # Rank of the *largest* observation must be n so that exceedance ordering
    # is consistent with the plotting positions.
    for method in PLOTTING_POSITIONS:
        out[f"p_nonexceed_{method}"] = plotting_position(out["rank_asc"], n, method)
    out["p_exceed_weibull"] = 1.0 - out["p_nonexceed_weibull"]
    out["return_period_observed_yr"] = 1.0 / out["p_exceed_weibull"].replace(0, np.nan)
    out["log_q"] = np.log(out["q_cumecs"])
    return out


def load(
    path: pathlib.Path | None = None,
    force_embedded: bool = False,
) -> DataBundle:
    """Full ingestion: read, tidy, rank, attach probabilities.

    Parameters
    ----------
    path:
        Override for the raw CSV location.
    force_embedded:
        Use the embedded verbatim record even when the CSV is present.  Used by
        the test suite to guarantee the pipeline runs without a data file.
    """
    if force_embedded:
        raw = pd.DataFrame(
            EMBEDDED_RECORD,
            columns=["wy_label", "wl_m", "recorded_date", "recorded_time", "q_cumecs"],
        )
        source, digest = "embedded", None
    else:
        source = str(path or (_cfg.DATA_DIR / _cfg.RAW_DATA_FILENAME))
        raw = load_raw_csv(path)
        digest = (
            util.sha256_of(pathlib.Path(source))
            if pathlib.Path(source).exists()
            else None
        )

    frame = add_probabilities(build_frame(raw))
    bundle = DataBundle(
        frame=frame, source=source, source_sha256=digest, n_records=len(frame)
    )
    log.info(
        "loaded %d annual peaks for %s-%s from %s (mean %.0f cumecs, CV %.3f)",
        bundle.n_records,
        bundle.summary()["wy_start"],
        bundle.summary()["wy_end"],
        source,
        bundle.summary()["mean_cumecs"],
        bundle.summary()["cv"],
    )
    return bundle


def synthetic_record(
    n: int = 57, seed: int | None = None, shape: float = 0.0, scale: float = 1.0
) -> DataBundle:
    """Synthetic GEV annual-maxima series, used by the tests and demos.

    ``shape=0`` reproduces a Gumbel, so the documented Gumbel reduced-variate
    values can be checked against the fitted quantiles.
    """
    rng = np.random.default_rng(util.stream_seed("synthetic") if seed is None else seed)
    q = scale * (
        rng.gumbel(size=n)
        if shape == 0.0
        else rng.gumbel(size=n) + shape * (rng.gumbel(size=n) - np.log(2))
    )
    dates = pd.date_range("1990-01-01", periods=n, freq="YE-DEC")
    raw = pd.DataFrame(
        {
            "wy_label": [f"{d.year}-{str(d.year + 1)[-2:]}" for d in dates],
            "wl_m": _cfg.STATION.zero_gauge_level_m + 4.0 + 0.35 * q,
            "recorded_date": [d.strftime("%d-%m-%Y") for d in dates],
            "recorded_time": "00:00:00",
            "q_cumecs": np.round(q, 0),
        }
    )
    frame = add_probabilities(build_frame(raw))
    return DataBundle(frame=frame, source="synthetic", source_sha256=None, n_records=n)


__all__ = [
    "EMBEDDED_RECORD",
    "PLOTTING_POSITIONS",
    "DataBundle",
    "plotting_position",
    "load_raw_csv",
    "build_frame",
    "add_probabilities",
    "load",
    "synthetic_record",
]
