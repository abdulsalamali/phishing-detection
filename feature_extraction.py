"""Shared URL feature extraction for the phishing-detection project.

This module is the single source of truth for the lexical and structural
feature engineering used across the Random Forest and XGBoost notebooks.
The per-model modules (``rf_finetuned_feature_extraction``,
``reduced_grid_feature_extraction`` and ``XGBoost_model_feature_extraction``)
are thin re-exports of this module and are kept only for backwards
compatibility with existing notebook imports.
"""

import math
import os
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tldextract


def extract_features_vectorized(urls_series, trusted_domains_set):
    features = {}
    urls = urls_series.reset_index(drop=True)

    # Basic URL structure
    features["url_length"] = urls.str.len()

    def safe_domain(url):
        try:
            return urlparse(url).netloc.lower()
        except:
            return ""

    domains = urls.apply(safe_domain)
    domains_clean = domains.str.replace("www.", "", regex=False)
    features["domain_length"] = domains.str.len()

    def safe_path(url):
        try:
            return urlparse(url).path
        except:
            return ""

    paths = urls.apply(safe_path)

    def safe_query(url):
        try:
            return urlparse(url).query
        except:
            return ""

    queries = urls.apply(safe_query)

    # Trusted domain features
    def safe_base_domain(url):
        try:
            domain = urlparse(url).netloc.lower()
            ext = tldextract.extract(domain)
            return (
                f"{ext.domain}.{ext.suffix}".lower()
                if ext.domain and ext.suffix
                else domain
            )
        except:
            return ""

    base_domains = urls.apply(safe_base_domain)
    features["is_trusted_domain"] = base_domains.isin(trusted_domains_set).astype(int)

    is_trusted_sub = pd.Series(False, index=urls.index)
    for trusted in list(trusted_domains_set)[:100]:
        is_trusted_sub |= domains_clean.str.endswith("." + trusted)
    features["is_trusted_subdomain"] = is_trusted_sub.astype(int)

    # Character counts
    features["num_dots"] = urls.str.count("\\.")
    features["num_hyphens"] = urls.str.count("-")
    features["num_slashes"] = urls.str.count("/")
    features["num_digits"] = urls.str.count("[0-9]")

    # Domain structure features
    features["domain_dots"] = domains.str.count("\\.")
    features["domain_hyphens"] = domains.str.count("-")
    features["domain_digits"] = domains.str.count("[0-9]")
    features["domain_underscores"] = domains.str.count("_")

    features["num_subdomains"] = features["domain_dots"] - 1
    features["num_subdomains"] = features["num_subdomains"].clip(lower=0)

    # Security indicators
    features["has_https"] = urls.str.startswith("https://").astype(int)
    features["is_http"] = urls.str.startswith("http://").astype(int)

    features["at_symbol"] = urls.str.count("@")
    features["has_at_symbol"] = (features["at_symbol"] > 0).astype(int)

    features["double_slash"] = urls.str.count("//")
    features["has_double_slash"] = (features["double_slash"] > 0).astype(int)

    # Suspicious patterns
    features["has_ip"] = urls.str.contains(
        r"\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b", regex=True, na=False
    ).astype(int)

    features["has_suspicious_words"] = urls.str.contains(
        r"(?:paypal|amazon|apple|microsoft|google|facebook|bank|secure|login|verify|update|confirm|account)",
        case=False,
        regex=True,
        na=False,
    ).astype(int)

    suspicious_tlds = [
        ".tk",
        ".ml",
        ".ga",
        ".cf",
        ".cn",
        ".xyz",
        ".gq",
        ".esy.es",
        ".duckdns.org",
    ]
    has_suspicious_tld = pd.Series(False, index=urls.index)
    for tld in suspicious_tlds:
        has_suspicious_tld |= urls.str.endswith(tld)
    features["suspicious_tld"] = has_suspicious_tld.astype(int)

    url_shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "short.link", "ow.ly"]
    has_shortener = pd.Series(False, index=urls.index)
    for shortener in url_shorteners:
        has_shortener |= urls.str.contains(shortener, case=False, regex=False)
    features["uses_shortener"] = has_shortener.astype(int)

    # Character encoding and special characters
    features["url_encoded_chars"] = urls.str.count("%")
    features["special_char_count"] = urls.str.count("[;()\\[\\]{}<>\\\\\\\\\\']")
    features["num_question_marks"] = urls.str.count("\\?")
    features["num_equal_signs"] = urls.str.count("=")
    features["num_hash_symbols"] = urls.str.count("#")
    features["num_dollar_signs"] = urls.str.count("\\$")
    features["num_ampersand_marks"] = urls.str.count("&")

    # Query and path analysis
    features["has_query"] = urls.str.contains("\\?", na=False).astype(int)
    features["query_length"] = queries.str.len()
    features["num_query_params"] = queries.str.count("=")

    features["url_path_length"] = paths.str.len()
    features["path_depth"] = paths.str.count("/") - 1
    features["path_depth"] = features["path_depth"].clip(lower=0)

    def safe_first_dir_length(path):
        try:
            parts = path.split("/")
            return len(parts[1]) if len(parts) > 1 else 0
        except:
            return 0

    features["first_dir_length"] = paths.apply(safe_first_dir_length)
    features["has_empty_directory"] = paths.str.contains("//", na=False).astype(int)

    features["suspicious_path"] = urls.str.contains(
        r"(?:cgi-bin|admin|login|secure|account|verify|update|confirm)",
        case=False,
        regex=True,
        na=False,
    ).astype(int)

    # Ratios and derived metrics
    url_lengths = features["url_length"]
    letter_count = urls.str.count("[a-zA-Z]")

    features["digit_ratio"] = features["num_digits"] / url_lengths.replace(0, 1)
    features["alpha_ratio"] = letter_count / url_lengths.replace(0, 1)
    features["special_char_ratio"] = features[
        "special_char_count"
    ] / url_lengths.replace(0, 1)
    features["digit_to_letter_ratio"] = features["num_digits"] / (letter_count + 1)

    # Character variety
    features["char_variety"] = urls.apply(
        lambda x: len(set(x.lower())) / len(x) if len(x) > 0 else 0
    )
    features["domain_char_variety"] = domains.apply(
        lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
    )

    # Entropy calculations
    def calculate_entropy_safe(string):
        if not string or len(string) <= 1:
            return 0
        try:
            counts = Counter(string)
            length = len(string)
            return -sum(
                (count / length) * math.log2(count / length)
                for count in counts.values()
            )
        except:
            return 0

    features["entropy_url"] = urls.apply(calculate_entropy_safe)
    features["entropy_domain"] = domains.apply(calculate_entropy_safe)
    features["entropy_path"] = paths.apply(calculate_entropy_safe)

    # Domain quality indicators
    features["has_non_ascii"] = domains.str.contains(
        r"[^\\x00-\\x7F]", na=False
    ).astype(int)
    features["long_numeric_sequence"] = domains.str.contains(
        r"[0-9]{4,}", na=False
    ).astype(int)

    # Convert to DataFrame
    df_features = pd.DataFrame(features)

    # Ensure all features are numeric
    for col in df_features.columns:
        if df_features[col].dtype == "object":
            df_features[col] = pd.to_numeric(df_features[col], errors="coerce").fillna(
                0
            )

    return df_features


def load_trusted_domains(file_path=None):
    """Load the set of trusted domains from a CSV file.

    The path can be passed explicitly or set via the ``TRUSTED_DOMAINS_PATH``
    environment variable. It defaults to ``../datasets/trusted_domains.csv``
    relative to the repository root.
    """
    if file_path is None:
        file_path = os.environ.get(
            "TRUSTED_DOMAINS_PATH", "../datasets/trusted_domains.csv"
        )
    try:
        trusted_df = pd.read_csv(file_path)
        return set(trusted_df.iloc[:, 0].str.lower().str.strip())
    except Exception as e:
        print(f"Could not load trusted domains: {e}")
        return set()


EXPECTED_FEATURES = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_slashes",
    "num_digits",
    "domain_dots",
    "domain_hyphens",
    "domain_digits",
    "domain_underscores",
    "num_subdomains",
    "has_https",
    "is_http",
    "at_symbol",
    "has_at_symbol",
    "double_slash",
    "has_double_slash",
    "has_ip",
    "has_suspicious_words",
    "suspicious_tld",
    "uses_shortener",
    "url_encoded_chars",
    "special_char_count",
    "num_question_marks",
    "num_equal_signs",
    "num_hash_symbols",
    "num_dollar_signs",
    "num_ampersand_marks",
    "has_query",
    "query_length",
    "num_query_params",
    "url_path_length",
    "path_depth",
    "first_dir_length",
    "has_empty_directory",
    "suspicious_path",
    "digit_ratio",
    "alpha_ratio",
    "special_char_ratio",
    "digit_to_letter_ratio",
    "char_variety",
    "domain_char_variety",
    "entropy_url",
    "entropy_domain",
    "entropy_path",
    "has_non_ascii",
    "long_numeric_sequence",
    "is_trusted_domain",
    "is_trusted_subdomain",
]
