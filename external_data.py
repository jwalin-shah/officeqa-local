#!/usr/bin/env python3
"""External data lookups for values not in the Treasury Bulletin corpus.

Provides FX rate lookups via:
  1. Live fetch from free CDN API (fawazahmed0/currency-api on jsDelivr)
  2. Static fallback cache for known benchmark dates

To add a rate manually:
  FX_CACHE[("usd", "jpy", 2025, 3, 31)] = 149.98
"""

from __future__ import annotations

import json
import re
import urllib.request

# ── Static fallback cache ───────────────────────────────────────────────────
# Key: (base_lower, quote_lower, year, month, day) → rate
# 1 base = rate quote (e.g. 1 USD = 149.98 JPY)
FX_CACHE: dict[tuple[str, str, int, int, int], float] = {
    ("usd", "jpy", 2025, 3, 31): 149.98,
    ("usd", "jpy", 2016, 3, 16): 113.77,
    ("usd", "gbp", 2016, 3, 16): 0.7076,  # 1 USD = 0.7076 GBP
}

_FETCH_TIMEOUT = 5


def _fetch_rate(base: str, quote: str, year: int, month: int, day: int) -> float | None:
    """Live-fetch a historical FX rate from the fawazahmed0 currency API.

    URL pattern: cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@YYYY-MM-DD/v1/currencies/{base}.json
    Returns rate as float (1 base = rate quote), or None on failure.
    """
    base_l = base.lower()
    quote_l = quote.lower()
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    url = (
        f"https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api"
        f"@{date_str}/v1/currencies/{base_l}.json"
    )
    try:
        resp = urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT)  # noqa: S310
        data = json.loads(resp.read())
        rate = data.get(base_l, {}).get(quote_l)
        if rate is not None:
            rate = float(rate)
            FX_CACHE[(base_l, quote_l, year, month, day)] = rate
            return rate
    except Exception:
        pass
    return None


def lookup_fx(
    base: str,
    quote: str,
    year: int,
    month: int | None = None,
    day: int | None = None,
) -> float | None:
    """Look up an FX rate. Tries static cache first, then live fetch."""
    base_l = base.lower()
    quote_l = quote.lower()

    if month is not None and day is not None:
        key = (base_l, quote_l, year, month, day)
        if key in FX_CACHE:
            return FX_CACHE[key]
        # Try live fetch
        rate = _fetch_rate(base_l, quote_l, year, month, day)
        if rate is not None:
            return rate

    # Month-level fallback (try day=1)
    if month is not None and day is not None:
        key_m = (base_l, quote_l, year, month, 1)
        if key_m in FX_CACHE:
            return FX_CACHE[key_m]

    return None


# ── Currency pair detection ─────────────────────────────────────────────────

_CURRENCY_PATTERNS: list[tuple[str, str, str]] = [
    ("JPY", "usd", "jpy"),
    ("YEN", "usd", "jpy"),
    ("JAPAN", "usd", "jpy"),
    ("GBP", "usd", "gbp"),
    ("STERLING", "usd", "gbp"),
    ("POUND", "usd", "gbp"),
    ("CAD", "usd", "cad"),
    ("CANAD", "usd", "cad"),
    ("EUR", "usd", "eur"),
    ("INR", "usd", "inr"),
    ("RUPEE", "usd", "inr"),
    ("INDIA", "usd", "inr"),
    ("DEM", "usd", "dem"),
    ("MARK", "usd", "dem"),
    ("CHF", "usd", "chf"),
    ("FRANC", "usd", "chf"),
]

_MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def resolve_fx_dr(dr: dict) -> list[float] | None:
    """Resolve a source='fx' data_request.

    Detects currency pair from label/row_hint, parses date, tries cache
    then live fetch. Returns [rate] or None.
    """
    years = dr.get("years") or []
    if not years:
        return None

    label = (dr.get("label") or "").upper()
    row_hint = (dr.get("row_hint") or "").upper()
    combined = label + " " + row_hint

    pairs_to_try: list[tuple[str, str]] = []
    for keyword, base, quote in _CURRENCY_PATTERNS:
        if keyword in combined and (base, quote) not in pairs_to_try:
            pairs_to_try.append((base, quote))

    if not pairs_to_try:
        return None

    year = int(years[0])
    month = dr.get("start_month") or dr.get("end_month")
    day = None

    # Parse date from label (e.g. "March 31, 2025")
    date_match = re.search(
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2})",
        combined,
        re.IGNORECASE,
    )
    if date_match:
        day = int(date_match.group(1))
        month_str = date_match.group(0)[:3].lower()
        if month_str in _MONTH_MAP:
            month = _MONTH_MAP[month_str]

    if month is None:
        month = 1
    if day is None:
        day = 1

    for base, quote in pairs_to_try:
        rate = lookup_fx(base, quote, year, month, day)
        if rate is not None:
            return [rate]

    return None
