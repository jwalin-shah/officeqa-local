"""Column Finder: Locates target columns in Treasury Bulletin tables with hierarchical headers."""

import re
from typing import Any


class ColumnFinder:
    """Finds columns by matching identifiers against hierarchical column headers."""

    def find_column(
        self, rows: list[dict[str, Any]], column_identifier: str
    ) -> tuple[str | None, float]:
        """
        Find column by keyword matching against column names.

        Args:
            rows: List of row dicts from table (keys are column names)
            column_identifier: Target identifier (e.g., "total receipts", "Total > Receipts")

        Returns:
            (column_name, confidence) where:
            - column_name: The actual column name from rows (preserving original), or None
            - confidence: 0.0-1.0 score indicating match quality
        """

        if not rows or not column_identifier:
            return (None, 0.0)

        # Get all column names from first row
        column_names = list(rows[0].keys())

        if not column_names:
            return (None, 0.0)

        # Normalize the identifier for comparison
        normalized_identifier = self._normalize_for_matching(column_identifier)

        # Score all columns
        candidates = []
        for col_name in column_names:
            score = self.score_column_match(col_name, normalized_identifier)
            if score > 0.0:
                candidates.append((col_name, score))

        if not candidates:
            return (None, 0.0)

        # Return best match (preserve original column name)
        best_col, best_score = max(candidates, key=lambda x: x[1])
        return (best_col, best_score)

    def score_column_match(self, column_name: str, identifier: str) -> float:
        """
        Score how well a column matches an identifier.

        Strategy:
        1. Exact match (after normalization): 0.99
        2. All keywords present: 0.85
        3. Some keywords present: 0.60
        4. No match: 0.0

        Args:
            column_name: The column header (may contain hierarchical structure like "Total > Receipts")
            identifier: Normalized identifier to match against

        Returns:
            Confidence score 0.0-1.0
        """

        if not column_name or not identifier:
            return 0.0

        # Normalize both for comparison
        normalized_col = self._normalize_for_matching(column_name)

        # Exact match
        if normalized_col == identifier:
            return 0.99

        # Extract keywords from identifier (space-separated after normalization)
        identifier_keywords = identifier.split()

        if not identifier_keywords:
            return 0.0

        # Check how many keywords appear in the normalized column (word-boundary matching)
        col_upper = normalized_col.upper()
        col_words = set(col_upper.split())
        matching_keywords = 0

        for keyword in identifier_keywords:
            keyword_upper = keyword.upper()
            kw_words = keyword_upper.split()
            if all(w in col_words for w in kw_words):
                matching_keywords += 1

        # Score based on keyword coverage
        coverage = matching_keywords / len(identifier_keywords)

        if coverage == 1.0:
            # All keywords present
            return 0.85
        elif coverage >= 0.5:
            # At least half keywords present
            return 0.65
        elif coverage > 0.0:
            # Some keywords present (but less than half) - only if more than 1 keyword
            # Single keyword match is weak unless it's a strong signal
            if len(identifier_keywords) == 1:
                # Single keyword - be more conservative
                return 0.55
            else:
                # Multiple keywords, only some match
                return 0.60
        else:
            # No match
            return 0.0

    def _normalize_for_matching(self, text: str) -> str:
        """
        Normalize text for matching by flattening hierarchical structure.

        Converts:
        - "Total > Receipts" → "total receipts"
        - "On-budget > Receipts (1)" → "on budget receipts"
        - "  Multiple   Spaces  " → "multiple spaces"

        Args:
            text: Raw text to normalize

        Returns:
            Normalized, flattened, lowercase text
        """

        if not text:
            return ""

        # Flatten hierarchical headers: remove ">" and handle structure
        text = self.flatten_hierarchical_header(text)

        # Lowercase
        text = text.lower()

        # Remove numbering artifacts like (1), (2)
        text = re.sub(r"\(\d+\)", "", text)

        # Replace dashes with spaces (for "on-budget" → "on budget")
        text = text.replace("-", " ")

        # Remove extra punctuation
        text = text.replace("|", " ")
        text = text.replace('"', "")
        text = text.replace("'", "")

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def flatten_hierarchical_header(self, header: str) -> str:
        """
        Convert hierarchical headers to flat format.

        Examples:
        - "Category > Subcategory" → "category subcategory"
        - "A > B > C" → "a b c"
        - "Single" → "single"

        Args:
            header: Header text (may contain ">")

        Returns:
            Flattened header with ">" replaced by spaces, lowercased
        """

        if not header:
            return ""

        # Replace ">" with spaces
        header = header.replace(">", " ")

        # Clean up multiple spaces
        header = re.sub(r"\s+", " ", header)

        # Lowercase
        header = header.lower()

        # Strip whitespace
        header = header.strip()

        return header

    def find_all_candidate_columns(
        self, rows: list[dict[str, Any]], column_identifier: str, min_confidence: float = 0.50
    ) -> list[tuple[str, float]]:
        """
        Find ALL columns that could match the identifier.

        Returns list of (column_name, confidence) sorted by confidence descending.

        Args:
            rows: List of row dicts
            column_identifier: Target identifier
            min_confidence: Minimum confidence to include (default 0.50)

        Returns:
            List of (column_name, confidence) sorted by confidence descending
        """

        if not rows or not column_identifier:
            return []

        column_names = list(rows[0].keys())
        normalized_identifier = self._normalize_for_matching(column_identifier)

        candidates = []
        for col_name in column_names:
            score = self.score_column_match(col_name, normalized_identifier)
            if score >= min_confidence:
                candidates.append((col_name, score))

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates


# Example usage
if __name__ == "__main__":
    finder = ColumnFinder()

    # Test 1: Exact match with hierarchical header
    rows1 = [
        {"Fiscal year": "1995", "Total > Receipts": 1350576.0},
        {"Fiscal year": "1996", "Total > Receipts": 1413156.0},
    ]
    col_name, conf = finder.find_column(rows1, "total receipts")
    print(f"Test 1 - Exact match: {col_name} (confidence: {conf})")

    # Test 2: Partial match
    rows2 = [
        {
            "Fiscal year or month": "1995",
            "Total on-budget and off-budget results > Total receipts (1)": 1350576.0,
            "Total on-budget and off-budget results > On-budget receipts (2)": 1000000.0,
        }
    ]
    col_name, conf = finder.find_column(rows2, "total receipts")
    print(f"Test 2 - Partial match: {col_name} (confidence: {conf})")

    # Test 3: Missing column
    rows3 = [{"Fiscal year": "1995", "Total > Outlays": 1514389.0}]
    col_name, conf = finder.find_column(rows3, "total receipts")
    print(f"Test 3 - Missing column: {col_name} (confidence: {conf})")

    # Test 4: Multiple candidates
    rows4 = [
        {
            "Year": "1995",
            "Total Receipts": 1350576.0,
            "On-Budget Receipts": 1000000.0,
            "Off-Budget Receipts": 350576.0,
        }
    ]
    candidates = finder.find_all_candidate_columns(rows4, "receipts", min_confidence=0.60)
    print("\nTest 4 - Multiple candidates for 'receipts':")
    for col, conf in candidates:
        print(f"  {col}: {conf}")

    # Test 5: Hierarchical with numbering
    rows5 = [
        {"Fiscal year": "1995", "Summary > Total (1)": 5000000.0, "Summary > Detail (2)": 3000000.0}
    ]
    col_name, conf = finder.find_column(rows5, "summary total")
    print(f"\nTest 5 - Hierarchical with numbering: {col_name} (confidence: {conf})")

    # Test 6: Normalization examples
    print("\nTest 6 - Normalization examples:")
    test_headers = [
        "Total > Receipts",
        "On-budget > Receipts (1)",
        "  Multiple   Spaces  ",
        "Total on-budget and off-budget results > Total receipts (1)",
    ]
    for header in test_headers:
        normalized = finder._normalize_for_matching(header)
        print(f"  '{header}' → '{normalized}'")
