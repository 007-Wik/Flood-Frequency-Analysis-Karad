"""
Environment manifest and (opt-in) dependency bootstrap.

The notebook hard-pinned nothing and recorded nothing, so a re-run six months
later could silently use different SciPy behaviour.  This module writes a
machine-readable manifest of every package actually loaded, which is embedded
in each result table's provenance sidecar.

``ensure_requirements`` will *not* install anything unless called with
``install=True``; the default reports what is missing and how to fix it.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import platform
import sys
from typing import Any

from . import config as _cfg
from . import util

REQUIRED = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "sklearn",
    "plotly",
)
OPTIONAL = ("openpyxl", "pytest")


def _version_of(dist: str, module: str) -> str | None:
    """Version of an installed distribution, falling back to ``__version__``."""
    for name in (dist, module):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    try:
        mod = importlib.import_module(module)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return None


def environment_manifest() -> dict[str, Any]:
    """Versions, interpreter and platform details for the active environment."""
    log = util.get_logger("packages")
    versions = {}
    missing = []
    for dist in REQUIRED:
        ver = _version_of(dist, dist)
        if ver is None:
            missing.append(dist)
        versions[dist] = ver
    optional = {dist: _version_of(dist, dist) for dist in OPTIONAL}
    manifest = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "required": versions,
        "optional": optional,
        "missing_required": missing,
        "master_seed": _cfg.MASTER_SEED,
    }
    if missing:
        log.warning("missing required packages: %s", ", ".join(missing))
    else:
        log.info(
            "environment OK (%s)",
            ", ".join(f"{k} {v}" for k, v in versions.items() if v),
        )
    return manifest


def check_requirements(raise_on_missing: bool = True) -> list[str]:
    """Return the list of missing required distributions.

    Raises ``RuntimeError`` when ``raise_on_missing`` and anything is absent.
    """
    missing = [d for d in REQUIRED if _version_of(d, d) is None]
    if missing and raise_on_missing:
        raise RuntimeError(
            "missing required packages: "
            + ", ".join(missing)
            + "\nInstall them with:\n    pip install -r requirements.txt"
        )
    return missing


def ensure_requirements(install: bool = False, dry_run: bool = True) -> list[str]:
    """Report, and optionally install, missing requirements.

    Parameters
    ----------
    install:
        When ``False`` (default) nothing is installed and the missing
        distributions are simply reported.  Automatic installation is opt-in
        because it mutates the environment.
    dry_run:
        When ``install=True`` this prints the exact ``pip`` command instead of
        running it.

    Returns
    -------
    list[str]
        The distributions that were (or would be) installed.
    """
    log = util.get_logger("packages")
    missing = check_requirements(raise_on_missing=False)
    if not missing:
        log.info("all required packages present")
        return []
    cmd = f"{sys.executable} -m pip install -r {_cfg.PROJECT_ROOT / 'requirements.txt'}"
    if not install:
        log.warning("missing: %s\n  run: %s", ", ".join(missing), cmd)
        return missing
    if dry_run:
        log.warning("[dry-run] would run: %s", cmd)
        return missing
    import subprocess

    subprocess.run(cmd, check=True, shell=False)
    check_requirements(raise_on_missing=True)
    return missing


def write_manifest() -> "Any":
    """Write ``outputs/environment_manifest.json`` and return the manifest."""
    manifest = environment_manifest()
    manifest["generated_utc"] = _cfg.run_timestamp()
    return util.write_json(manifest, "environment_manifest")


__all__ = [
    "REQUIRED",
    "OPTIONAL",
    "environment_manifest",
    "check_requirements",
    "ensure_requirements",
    "write_manifest",
]
