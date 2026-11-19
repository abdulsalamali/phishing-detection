"""Compatibility shim for the XGBoost feature extraction.

The implementation now lives in :mod:`feature_extraction`. This module is kept
so that existing imports (e.g. in ``XGBoost_model.ipynb``) continue to work
unchanged.
"""

from feature_extraction import (
    EXPECTED_FEATURES,
    extract_features_vectorized,
    load_trusted_domains,
)

__all__ = ["extract_features_vectorized", "load_trusted_domains", "EXPECTED_FEATURES"]
