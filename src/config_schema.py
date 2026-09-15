"""Config schema validation — single source of truth for config.yaml contract.

All thresholds MUST be in config.yaml; no hardcoding in logic.
This module validates the contract and rejects silent fallbacks.
"""

from __future__ import annotations

from typing import Dict, List

REQUIRED_THRESHOLD_KEYS = [
    "corr_threshold_high",
    "corr_threshold_low",
    "beta_min",
    "beta_max",
    "alpha",
]

REQUIRED_TOP_LEVEL_KEYS = ["thresholds", "windows"]


def validate_config(cfg: Dict) -> Dict:
    """Validate config.yaml dict. Raises ValueError on contract violation.

    Checks:
      - Required top-level keys present (thresholds, windows)
      - All REQUIRED_THRESHOLD_KEYS present (no silent .get defaults)
      - windows is non-empty list of positive ints, strictly ascending
      - primary_window (if present) is in windows
      - Delegates range checks to validate_ranges()

    Returns:
        cfg unchanged if valid.

    Raises:
        ValueError: on any contract violation.
        KeyError: if threshold key missing (wrapped as ValueError for caller).
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict (parsed YAML)")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in cfg:
            raise ValueError(f"config.yaml missing required key: '{key}'")

    thresholds = cfg.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("config.yaml 'thresholds' must be a mapping")

    missing = [k for k in REQUIRED_THRESHOLD_KEYS if k not in thresholds]
    if missing:
        raise ValueError(f"config.yaml thresholds missing required keys: {missing} — no silent defaults allowed")

    # windows validation
    windows = cfg.get("windows")
    if not isinstance(windows, list) or len(windows) == 0:
        raise ValueError("config.yaml 'windows' must be a non-empty list")
    if not all(isinstance(w, int) and w > 0 for w in windows):
        raise ValueError(f"config.yaml 'windows' must be list of positive ints, got {windows}")
    if windows != sorted(windows):
        raise ValueError(f"config.yaml 'windows' must be strictly ascending, got {windows}")
    if len(windows) != len(set(windows)):
        raise ValueError(f"config.yaml 'windows' must not contain duplicates, got {windows}")

    primary = cfg.get("primary_window")
    if primary is not None:
        if primary not in windows:
            raise ValueError(f"config.yaml 'primary_window' {primary} must be one of windows {windows}")

    # Range checks
    validate_ranges(cfg)

    return cfg


def validate_ranges(cfg: Dict) -> None:
    """Validate numeric ranges for thresholds. Raises ValueError if violated.

    Checks:
      - beta_min <= beta_max
      - alpha in (0, 1)
      - corr_threshold_low < corr_threshold_high
      - corr thresholds in [0, 1]
    """
    t = cfg.get("thresholds", {})
    beta_min = t.get("beta_min")
    beta_max = t.get("beta_max")
    alpha = t.get("alpha")
    corr_low = t.get("corr_threshold_low")
    corr_high = t.get("corr_threshold_high")

    if beta_min is not None and beta_max is not None:
        if beta_min > beta_max:
            raise ValueError(f"thresholds.beta_min ({beta_min}) must be <= beta_max ({beta_max})")

    if alpha is not None:
        if not (0 < alpha < 1):
            raise ValueError(f"thresholds.alpha ({alpha}) must be in (0, 1)")

    if corr_low is not None and corr_high is not None:
        if not (corr_low < corr_high):
            raise ValueError(f"thresholds.corr_threshold_low ({corr_low}) must be < corr_threshold_high ({corr_high})")
        for name, val in [("corr_threshold_low", corr_low), ("corr_threshold_high", corr_high)]:
            if not (0 <= val <= 1):
                raise ValueError(f"thresholds.{name} ({val}) must be in [0, 1]")
