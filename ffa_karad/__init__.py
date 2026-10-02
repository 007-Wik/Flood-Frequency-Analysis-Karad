"""
Karad (AK000X6) flood-frequency analysis package.

Design principles that differ deliberately from the 2025 notebook
``FFA_Karad_Advanced_CWC_V2.ipynb``:

1.  **No hard-coded results.**  Every number that reaches a report is computed
    from the observed sample and stored in a result object.  There are no
    numeric literals in the reporting layer.
2.  **One source of truth for constants.**  Every tunable constant -- station
    metadata, return periods, significance levels, iteration counts, standards
    coefficients -- lives in :mod:`ffa_karad.config`, each with a citation or
    an explicit "VERIFY" note.
3.  **Refuse to emit untrustworthy numbers.**  Fitted models, MCMC chains and
    peaks-over-threshold fits each carry acceptance gates.  When a gate fails,
    the pipeline reports the failure instead of silently publishing a value.
4.  **Every standard is applied by its *definition*.**  Bulletin 17B skewness
    and kurtosis, Bulletin 17C confidence limits, IS 11223:1985 method
    selection, GPD tail-sign convention and Jacobian-corrected information
    criteria are implemented from first principles and unit tested.
"""

from __future__ import annotations

__version__ = "3.0.0"
__all__ = ["__version__"]
