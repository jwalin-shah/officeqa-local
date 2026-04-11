"""Row Finder: Locates rows by time period (fiscal year, calendar year, month)."""

import re
from typing import Any


class RowFinder:
    """Finds rows in tables by matching year/period identifiers."""

    # Month patterns: various formats for months
    MONTH_NAMES = {
        "january": 1,
        "jan": 1,
        "jan.": 1,
        "1": 1,
        "01": 1,
        "february": 2,
        "feb": 2,
        "feb.": 2,
        "2": 2,
        "02": 2,
        "march": 3,
        "mar": 3,
        "mar.": 3,
        "3": 3,
        "03": 3,
        "april": 4,
        "apr": 4,
        "apr.": 4,
        "4": 4,
        "04": 4,
        "may": 5,
        "5": 5,
        "05": 5,
        "june": 6,
        "jun": 6,
        "jun.": 6,
        "6": 6,
        "06": 6,
        "july": 7,
        "jul": 7,
        "jul.": 7,
        "7": 7,
        "07": 7,
        "august": 8,
        "aug": 8,
        "aug.": 8,
        "8": 8,
        "08": 8,
        "september": 9,
        "sep": 9,
        "sep.": 9,
        "sept": 9,
        "sept.": 9,
        "9": 9,
        "09": 9,
        "october": 10,
        "oct": 10,
        "oct.": 10,
        "10": 10,
        "november": 11,
        "nov": 11,
        "nov.": 11,
        "11": 11,
        "december": 12,
        "dec": 12,
        "dec.": 12,
        "12": 12,
    }

    def __init__(self):
        """Initialize the row finder."""
        pass

    def find_row_by_period(
        self,
        rows: list[dict[str, Any]],
        year: int,
        period_type: str = "fiscal",
        month: int | None = None,
    ) -> tuple[dict | None, int | None, float]:
        """
        Find row by year/period.

        Args:
            rows: List of row dicts from table
            year: Target year (e.g., 1995)
            period_type: "fiscal" or "calendar"
            month: Target month (1-12) if looking for a specific month

        Returns:
            (row_dict, row_index, confidence)
            - row_dict: The matching row or None
            - row_index: Index of the matching row or None
            - confidence: Score 0.0-1.0 indicating match quality
        """
        if not rows:
            return (None, None, 0.0)

        best_match = None
        best_idx = None
        best_score = 0.0

        for i, row in enumerate(rows):
            # Find the period column (usually first column)
            period_label = self._get_period_label(row)

            if not period_label:
                continue

            # Score how well this row matches the target
            score = self.score_period_match(period_label, year, period_type, month)

            if score > best_score:
                best_score = score
                best_match = row
                best_idx = i

        return (best_match, best_idx, best_score)

    def score_period_match(
        self, row_label: str, target_year: int, period_type: str, month: int | None = None
    ) -> float:
        """
        Score how well a row label matches the target period.

        Scoring (in priority order):
        - Exact fiscal year match (19951, FY1995): 0.99
        - FY/Fiscal prefix match: 0.95
        - Calendar year match (with calendar context): 0.95
        - Month + year match: 0.90
        - Partial match (year but wrong month): 0.60
        - Year only (ambiguous period): 0.50
        - No match: 0.0

        Args:
            row_label: Period identifier from row (e.g., "1995", "FY1995", "Jan. 1995")
            target_year: Target year (e.g., 1995)
            period_type: "fiscal" or "calendar"
            month: Optional month to match (1-12)

        Returns:
            Confidence score 0.0-1.0
        """
        if not row_label or not isinstance(row_label, str):
            return 0.0

        row_label = row_label.strip()

        # Parse what's in this row
        parsed = self.parse_period_string(row_label)

        if parsed is None:
            return 0.0

        parsed_year = parsed.get("year")
        parsed_month = parsed.get("month")
        parsed_period_type = parsed.get("period_type")

        # Year must match (or be absent for month-only labels)
        if parsed_year != target_year:
            # Special case: month-only labels (e.g., "February") have no year.
            # The year context comes from the table header, not the row itself.
            # Give a small confidence when the month matches the target month,
            # so these rows can be used when no better match exists.
            if parsed_year is None and parsed_month is not None and month is not None:
                if parsed_month == month:
                    return 0.30
            return 0.0

        # If looking for a specific month
        if month is not None:
            if parsed_month is None:
                # Year matches but no month -> partial match
                return 0.60

            if parsed_month == month:
                # Exact month and year match
                return 0.90

            # Month doesn't match
            return 0.0

        # Looking for annual/total data (no specific month)

        # If parsed data has a month but we didn't ask for one
        if parsed_month is not None and month is None:
            # This is monthly data, but we wanted annual
            return 0.0

        # Check period type alignment
        if parsed_period_type and period_type:
            if parsed_period_type == period_type:
                # Period type matches
                return 0.99
            elif parsed_period_type == "unknown":
                # We don't know the period type from the label
                return 0.95
            else:
                # Period type mismatch (e.g., fiscal vs calendar)
                # Only penalize if we're confident about the mismatch
                if parsed_period_type in ["fiscal", "calendar"]:
                    # Heavy penalty: wrong period type should be last resort
                    return 0.15

        # Default: year matches, period type unknown or not specified
        return 0.95

    def parse_period_string(self, text: str) -> dict[str, Any] | None:
        """
        Parse a period identifier string.

        Handles formats like:
        - "1995" (year only, ambiguous)
        - "19951" (fiscal year 1995, if context suggests FY)
        - "FY1995", "Fiscal 1995" (explicit fiscal year)
        - "CY1995", "Calendar 1995" (calendar year)
        - "Jan. 1995", "January 1995", "1995 - Jan." (month + year)
        - "February" (month only, no year)

        Returns:
            {
                'year': int or None,
                'month': int (1-12) or None,
                'period_type': 'fiscal' | 'calendar' | 'unknown',
                'confidence': float,
                'original': str
            }
            or None if parsing fails
        """
        if not text or not isinstance(text, str):
            return None

        text = text.strip()
        original = text

        # Check for month + year formats first (more specific)
        month_year_match = self._parse_month_year(text)
        if month_year_match:
            return month_year_match

        # Check for explicit fiscal/calendar year prefixes
        if text.lower().startswith("fy ") or text.lower().startswith("fy"):
            # FY1995 or FY 1995
            year_str = re.sub(r"^fy\s*", "", text, flags=re.IGNORECASE)
            try:
                year = int(year_str.strip())
                return {
                    "year": year,
                    "month": None,
                    "period_type": "fiscal",
                    "confidence": 0.99,
                    "original": original,
                }
            except ValueError:
                pass

        if text.lower().startswith("fiscal ") or text.lower().startswith("fiscal-"):
            # Fiscal 1995
            year_str = re.sub(r"^fiscal[\s\-]*", "", text, flags=re.IGNORECASE)
            try:
                year = int(year_str.strip())
                return {
                    "year": year,
                    "month": None,
                    "period_type": "fiscal",
                    "confidence": 0.95,
                    "original": original,
                }
            except ValueError:
                pass

        if text.lower().startswith("cy ") or text.lower().startswith("cy"):
            # CY1995 or CY 1995
            year_str = re.sub(r"^cy\s*", "", text, flags=re.IGNORECASE)
            try:
                year = int(year_str.strip())
                return {
                    "year": year,
                    "month": None,
                    "period_type": "calendar",
                    "confidence": 0.99,
                    "original": original,
                }
            except ValueError:
                pass

        if text.lower().startswith("calendar ") or text.lower().startswith("calendar-"):
            # Calendar 1995
            year_str = re.sub(r"^calendar[\s\-]*", "", text, flags=re.IGNORECASE)
            try:
                year = int(year_str.strip())
                return {
                    "year": year,
                    "month": None,
                    "period_type": "calendar",
                    "confidence": 0.95,
                    "original": original,
                }
            except ValueError:
                pass

        # Handle fiscal year encoding: "19951" means fiscal year 1995
        # This MUST be checked before the generic 4-digit year regex,
        # which would otherwise swallow "19951" as year=1995 + remainder="1".
        # NOTE: This encoding is Treasury Bulletin specific. The YYYYF format
        # where F is a trailing digit (1-9 for FY, 0 for CY) is used in
        # Treasury corpus data files. Other data sources may use 5-digit
        # numbers for different purposes. Confidence is conservative since
        # this is an assumption about the encoding scheme.
        # Example: 19951 = FY 1995, 19950 = CY 1995
        fy_match = re.match(r"^(\d{4})([0-9])$", text)
        if fy_match:
            year = int(fy_match.group(1))
            fy_indicator = int(fy_match.group(2))

            # If the last digit is 1-9, it's often FY encoding
            # If it's 0, it's CY
            if fy_indicator == 0:
                period_type = "calendar"
                confidence = 0.85
            elif 1 <= fy_indicator <= 9:
                # Assumed FY encoding (Treasury Bulletin convention)
                period_type = "fiscal"
                confidence = 0.70
            else:
                period_type = "unknown"
                confidence = 0.70

            return {
                "year": year,
                "month": None,
                "period_type": period_type,
                "confidence": confidence,
                "original": original,
            }

        # Try year-only formats
        # Watch for ambiguous cases like "1995" - could be FY or CY
        year_match = re.match(r"^(\d{4})(?:\s*[-.]?\s*(.*))?$", text)
        if year_match:
            year = int(year_match.group(1))
            remainder = year_match.group(2)

            # If there's a remainder, it might contain more info
            if remainder:
                # Could be "1995 - 1996" (range), "1995 (est)", etc.
                # For now, just use the year
                pass

            return {
                "year": year,
                "month": None,
                "period_type": "unknown",
                "confidence": 0.85,
                "original": original,
            }

        # No match
        return None

    def _parse_month_year(self, text: str) -> dict[str, Any] | None:
        """Parse month + year formats (most specific)."""
        text_lower = text.lower()

        # Try patterns like "Jan. 1995", "January 1995", "1995 - Jan", etc.

        # Pattern 1: "Month. Year" or "Month Year" (with or without period after month)
        # Handle "Jan." or "Jan" both
        month_list_with_period = "|".join([m.rstrip(".") for m in self.MONTH_NAMES.keys()])
        month_year_pattern = r"(\b(?:" + month_list_with_period + r")\b)\.?\s+(\d{4})"
        match = re.search(month_year_pattern, text_lower)
        if match:
            month_str = match.group(1)
            year = int(match.group(2))
            # Normalize month string (remove period if present)
            month_str_normalized = month_str.rstrip(".")
            month = self.MONTH_NAMES.get(month_str_normalized, None)
            if month:
                return {
                    "year": year,
                    "month": month,
                    "period_type": "calendar",  # Monthly data is calendar-aligned
                    "confidence": 0.95,
                    "original": text,
                }

        # Pattern 2: "Year - Month" or "Year - Month."
        year_month_pattern = r"(\d{4})\s*[-–]\s*(\b(?:" + month_list_with_period + r")\b)\.?"
        match = re.search(year_month_pattern, text_lower)
        if match:
            year = int(match.group(1))
            month_str = match.group(2)
            month_str_normalized = month_str.rstrip(".")
            month = self.MONTH_NAMES.get(month_str_normalized, None)
            if month:
                return {
                    "year": year,
                    "month": month,
                    "period_type": "calendar",
                    "confidence": 0.95,
                    "original": text,
                }

        # Pattern 3: Month only (no year)
        # Just a month name without a year - we can't match without additional context
        month_only_pattern = r"^(\b(?:" + month_list_with_period + r")\b)\.?$"
        match = re.match(month_only_pattern, text_lower)
        if match:
            month_str = match.group(1)
            month_str_normalized = month_str.rstrip(".")
            month = self.MONTH_NAMES.get(month_str_normalized, None)
            if month:
                return {
                    "year": None,
                    "month": month,
                    "period_type": "calendar",
                    "confidence": 0.60,
                    "original": text,
                }

        return None

    def _get_period_label(self, row: dict[str, Any]) -> str | None:
        """
        Extract the period/date identifier from a row.

        Usually this is the first column value. Tries common column names:
        - "Fiscal year or month"
        - "Date"
        - "Period"
        - "Year"
        - Or just the first column if others fail
        """
        if not row:
            return None

        # Try known period column names
        period_column_names = [
            "fiscal year or month",
            "fiscal year",
            "calendar year",
            "period",
            "date",
            "year",
            "month",
            "fiscal",
            "calendar",
        ]

        # Try exact and substring matches on column names
        for col_name in row:
            col_lower = col_name.lower() if isinstance(col_name, str) else ""
            for pattern in period_column_names:
                if pattern in col_lower:
                    value = row.get(col_name)
                    if value is not None and str(value).strip():
                        return str(value)

        # Fallback: use first column value
        # (Usually period info is in the first column of tables)
        if row:
            first_value = next(iter(row.values()))
            if first_value is not None and str(first_value).strip():
                return str(first_value)

        return None

    def find_all_candidate_rows(
        self,
        rows: list[dict[str, Any]],
        year: int,
        period_type: str = "fiscal",
        month: int | None = None,
    ) -> list[tuple[dict, int, float]]:
        """
        Find ALL rows that could match the target period.

        Returns:
            List of (row, index, confidence) sorted by confidence descending
        """
        candidates = []

        for i, row in enumerate(rows):
            period_label = self._get_period_label(row)
            if not period_label:
                continue

            score = self.score_period_match(period_label, year, period_type, month)

            if score > 0.0:
                candidates.append((row, i, score))

        # Sort by confidence descending
        return sorted(candidates, key=lambda x: x[2], reverse=True)


# Example usage and testing
if __name__ == "__main__":
    finder = RowFinder()

    print("=" * 70)
    print("ROW FINDER TEST SUITE")
    print("=" * 70)

    # Test 1: Basic fiscal year matching
    print("\nTest 1: Basic fiscal year matching")
    rows = [
        {"period": "19941", "value": 100},
        {"period": "19951", "value": 200},
        {"period": "19961", "value": 300},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="fiscal")
    print("Looking for FY 1995:")
    print(f"  Found: {row}")
    print(f"  Index: {idx}, Confidence: {conf:.2f}")
    assert row == {"period": "19951", "value": 200}, "Should find FY 1995 row"
    assert conf > 0.75, "Confidence should be high"

    # Test 2: Month + year matching
    print("\nTest 2: Month + year matching")
    rows = [
        {"date": "1995 - Jan.", "value": 20},
        {"date": "1995 - Feb.", "value": 25},
        {"date": "1995 - Mar.", "value": 30},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="calendar", month=2)
    print("Looking for Feb. 1995:")
    print(f"  Found: {row}")
    print(f"  Index: {idx}, Confidence: {conf:.2f}")
    assert row == {"date": "1995 - Feb.", "value": 25}, "Should find Feb. 1995"
    assert conf > 0.85, "Confidence should be high"

    # Test 3: Mixed fiscal year formats
    print("\nTest 3: Mixed fiscal year formats")
    rows = [
        {"fiscal year or month": "FY1994", "value": 100},
        {"fiscal year or month": "Fiscal 1995", "value": 200},
        {"fiscal year or month": "1996", "value": 300},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="fiscal")
    print("Looking for FY 1995:")
    print(f"  Found: {row}")
    print(f"  Index: {idx}, Confidence: {conf:.2f}")
    assert row == {"fiscal year or month": "Fiscal 1995", "value": 200}, "Should find FY 1995"

    # Test 4: Month only in rows (no year)
    print("\nTest 4: Month only (Feb. with implicit year)")
    rows = [
        {"date": "Jan.", "value": 20},
        {"date": "Feb.", "value": 25},
        {"date": "Mar.", "value": 30},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, month=2)
    print("Looking for month 2 (Feb) with year 1995:")
    print(f"  Found: {row}")
    print(f"  Index: {idx}, Confidence: {conf:.2f}")
    # Note: This row doesn't have year info, so match will be weaker
    print("  Note: No year in row, so confidence is partial")

    # Test 5: Parse period strings
    print("\nTest 5: Parse period strings")
    test_strings = [
        "1995",
        "19951",
        "FY1995",
        "Fiscal 1995",
        "CY1995",
        "Jan. 1995",
        "1995 - January",
        "February",
    ]
    for s in test_strings:
        parsed = finder.parse_period_string(s)
        if parsed:
            print(
                f"  '{s}' → year={parsed['year']}, month={parsed['month']}, "
                f"period={parsed['period_type']}, conf={parsed['confidence']:.2f}"
            )
        else:
            print(f"  '{s}' → FAILED TO PARSE")

    # Test 6: Find all candidates
    print("\nTest 6: Find all candidate rows")
    rows = [
        {"period": "19941", "value": 100},
        {"period": "1995", "value": 200},
        {"period": "FY1995", "value": 250},
        {"period": "19951", "value": 300},
    ]
    candidates = finder.find_all_candidate_rows(rows, year=1995, period_type="fiscal")
    print("All candidates for FY 1995:")
    for row, idx, conf in candidates:
        print(f"  Index {idx}: {row['period']} (conf={conf:.2f})")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
