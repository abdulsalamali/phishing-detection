"""Compatibility shim for the fine-tuned Random Forest feature extraction.

The implementation now lives in :mod:`feature_extraction`. This module is kept
so that existing imports (e.g. in ``rf_test_finetuned.ipynb`` and
``workflow_test.ipynb``) continue to work unchanged.
"""

from feature_extraction import (
    EXPECTED_FEATURES,
    extract_features_vectorized,
    load_trusted_domains,
)

__all__ = ["extract_features_vectorized", "load_trusted_domains", "EXPECTED_FEATURES"]
