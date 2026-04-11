"""Value Parser: Extracts cell values from table rows with confidence scoring."""

import re
from typing import Any


class ValueParser:
    """Extracts values from cleaned rows with fuzzy column matching and confidence scoring."""

    def __init__(self):
        """Initialize the value parser."""
        pass

    def extract_value(
        self, row: dict[str, Any], column_identifier: str, source_table: str = "unknown"
    ) -> dict[str, Any]:
        """
        Extract a value from a cleaned row by column identifier.

        Args:
            row: Cleaned row dict from DataNormalizer
                 Example: {"fiscal year": "1995", "total receipts": 1350576.0, "total outlays": 1514389.0}
            column_identifier: Column name/identifier to extract
                 Can be exact: "total receipts"
                 Can be fuzzy: "receipts"
                 Can be hierarchical: "total on-budget and off-budget results > total receipts"
            source_table: Table source identifier for metadata (default: "unknown")

        Returns:
            {
                'value': float or None,
                'confidence': float (0.0-1.0),
                'column_matched': str or None,
                'source': str,
                'quality_indicators': {
                    'exact_match': bool,
                    'value_valid': bool,
                    'column_found': bool,
                    'no_missing_data': bool
                },
                'match_strategy': str  # 'exact', 'fuzzy', 'hierarchical', or 'not_found'
            }
        """
        if not row:
            return self._make_result(
                value=None,
                confidence=0.0,
                column_matched=None,
                source=source_table,
                exact_match=False,
                value_valid=False,
                column_found=False,
                no_missing_data=False,
                strategy="not_found",
            )

        # Normalize the input identifier
        normalized_identifier = self._normalize_identifier(column_identifier)

        # Try hierarchical matching first (if contains ">")
        if ">" in normalized_identifier:
            matched_col, strategy = self._match_hierarchical(row, normalized_identifier)
        else:
            # Try exact match first, then fuzzy
            matched_col, strategy = self._find_matching_column(row, normalized_identifier)

        # If no match found
        if matched_col is None:
            return self._make_result(
                value=None,
                confidence=0.0,
                column_matched=None,
                source=source_table,
                exact_match=False,
                value_valid=False,
                column_found=False,
                no_missing_data=False,
                strategy="not_found",
            )

        # Extract the value
        raw_value = row.get(matched_col)
        value_valid = raw_value is not None and raw_value != ""

        if not value_valid:
            confidence = self.score_confidence(
                exact_match=(strategy == "exact"), value_exists=False, matches_found=1
            )
            return self._make_result(
                value=None,
                confidence=confidence,
                column_matched=matched_col,
                source=source_table,
                exact_match=(strategy == "exact"),
                value_valid=False,
                column_found=True,
                no_missing_data=False,
                strategy=strategy,
            )

        # Value exists and is valid
        confidence = self.score_confidence(
            exact_match=(strategy == "exact"), value_exists=True, matches_found=1
        )

        return self._make_result(
            value=raw_value if isinstance(raw_value, (int, float)) else raw_value,
            confidence=confidence,
            column_matched=matched_col,
            source=source_table,
            exact_match=(strategy == "exact"),
            value_valid=True,
            column_found=True,
            no_missing_data=True,
            strategy=strategy,
        )

    def _normalize_identifier(self, identifier: str) -> str:
        """
        Normalize a column identifier for matching.

        Applies same normalization as DataNormalizer.normalize_row_label:
        - Lowercase
        - Strip whitespace
        - Remove pipes and dashes
        - Collapse multiple spaces
        """
        # Strip whitespace
        normalized = identifier.strip()

        # Remove markdown pipes and artifacts (but preserve > for hierarchical)
        normalized = normalized.replace("|", " ")
        normalized = normalized.replace("--", " ")

        # Collapse multiple spaces
        normalized = re.sub(r"\s+", " ", normalized)

        # Remove trailing/leading dashes
        normalized = re.sub(r"^-+\s*", "", normalized)
        normalized = re.sub(r"\s*-+$", "", normalized)

        # Lowercase for matching
        normalized = normalized.lower()

        return normalized.strip()

    def _find_matching_column(
        self, row: dict[str, Any], column_identifier: str
    ) -> tuple[str | None, str]:
        """
        Find which column in row matches the identifier (exact or fuzzy).

        Returns:
            (matched_column_name, strategy)
            where strategy is 'exact', 'fuzzy', or None if no match
        """
        # Try exact match first
        for col_name in row:
            if col_name == column_identifier:
                return (col_name, "exact")

        # Try fuzzy match: identifier keywords appear in column name
        # Split identifier into keywords
        identifier_keywords = column_identifier.split()

        best_match = None
        best_match_count = 0

        for col_name in row:
            col_keywords = col_name.split()

            # Count how many identifier keywords appear in column name
            match_count = sum(1 for kw in identifier_keywords if kw in col_keywords)

            # Only consider it a match if at least one keyword matches
            if match_count > 0 and match_count > best_match_count:
                best_match = col_name
                best_match_count = match_count

        if best_match is not None:
            return (best_match, "fuzzy")

        return (None, None)

    def _match_hierarchical(
        self, row: dict[str, Any], column_identifier: str
    ) -> tuple[str | None, str]:
        """
        Match hierarchical column identifiers (format: "parent > child > ...").

        For hierarchical identifiers, flattens and matches based on the deepest level.

        Returns:
            (matched_column_name, strategy)
        """
        # Split on ">" to get hierarchy levels
        parts = [p.strip() for p in column_identifier.split(">")]

        if not parts:
            return (None, None)

        # Use the deepest (last) part as the primary match target
        target = parts[-1]

        # Try to match the target against row columns
        matched_col, strategy = self._find_matching_column(row, target)

        if matched_col is not None:
            return (matched_col, "hierarchical")

        return (None, None)

    def score_confidence(
        self, exact_match: bool, value_exists: bool, matches_found: int = 1
    ) -> float:
        """
        Calculate confidence score based on match quality and value availability.

        Scoring logic:
        - Exact column match: base 0.99
        - Fuzzy match: base 0.75
        - Value exists: no penalty
        - Value missing: -0.5
        - Multiple possible matches: -0.2 per extra match

        Args:
            exact_match: True if exact column match
            value_exists: True if value is not None/empty
            matches_found: Number of possible matches (for disambiguation penalty)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base score
        if exact_match:
            confidence = 0.99
        else:
            confidence = 0.75

        # Penalty for missing value
        if not value_exists:
            confidence -= 0.5

        # Penalty for multiple possible matches
        if matches_found > 1:
            confidence -= 0.2 * (matches_found - 1)

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))

    def _make_result(
        self,
        value: Any,
        confidence: float,
        column_matched: str | None,
        source: str,
        exact_match: bool,
        value_valid: bool,
        column_found: bool,
        no_missing_data: bool,
        strategy: str,
    ) -> dict[str, Any]:
        """Create a standardized result dict."""
        return {
            "value": value,
            "confidence": confidence,
            "column_matched": column_matched,
            "source": source,
            "quality_indicators": {
                "exact_match": exact_match,
                "value_valid": value_valid,
                "column_found": column_found,
                "no_missing_data": no_missing_data,
            },
            "match_strategy": strategy,
        }


# Example usage and testing
if __name__ == "__main__":
    from data_normalizer import DataNormalizer

    # Initialize components
    normalizer = DataNormalizer()
    parser = ValueParser()

    print("=" * 70)
    print("VALUE PARSER TEST SUITE")
    print("=" * 70)

    # Test 1: Basic exact match
    print("\nTest 1: Exact match on cleaned row")
    messy_row = {
        "  Fiscal Year  ": "1995",
        "Total Receipts | ": "1,350,576",
        "Total Outlays": "$1,514,389.00",
    }
    cleaned = normalizer.normalize_row_dict(messy_row)
    print(f"Cleaned row: {cleaned}")

    result = parser.extract_value(cleaned, "total receipts", source_table="FFO-1")
    print("Extract 'total receipts':")
    print(f"  Value: {result['value']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Match strategy: {result['match_strategy']}")
    print(f"  Quality: {result['quality_indicators']}")

    # Test 2: Fuzzy match (partial identifier)
    print("\nTest 2: Fuzzy match")
    result = parser.extract_value(cleaned, "receipts", source_table="FFO-1")
    print("Extract 'receipts':")
    print(f"  Value: {result['value']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Match strategy: {result['match_strategy']}")
    print(f"  Matched column: {result['column_matched']}")

    # Test 3: Hierarchical match
    print("\nTest 3: Hierarchical match")
    result = parser.extract_value(cleaned, "budget > total receipts", source_table="FFO-1")
    print("Extract 'budget > total receipts':")
    print(f"  Value: {result['value']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Match strategy: {result['match_strategy']}")

    # Test 4: Non-existent column
    print("\nTest 4: Non-existent column")
    result = parser.extract_value(cleaned, "nonexistent column", source_table="FFO-1")
    print("Extract 'nonexistent column':")
    print(f"  Value: {result['value']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Match strategy: {result['match_strategy']}")
    print(f"  Column found: {result['quality_indicators']['column_found']}")

    # Test 5: Missing value
    print("\nTest 5: Missing value in cell")
    row_with_missing = {"fiscal year": "1995", "total receipts": None, "total outlays": 1514389.0}
    result = parser.extract_value(row_with_missing, "total receipts", source_table="FFO-1")
    print("Extract 'total receipts' (missing value):")
    print(f"  Value: {result['value']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Value valid: {result['quality_indicators']['value_valid']}")

    # Test 6: Multiple rows - batch extraction
    print("\nTest 6: Batch extraction from multiple rows")
    messy_rows = [
        {
            "  Fiscal Year  ": "1995",
            "Total Receipts | ": "1,350,576",
            "Total Outlays": "$1,514,389.00",
        },
        {
            "  Fiscal Year  ": "1996",
            "Total Receipts | ": "1,413,156",
            "Total Outlays": "$1,612,128.00",
        },
    ]
    cleaned_rows, _ = normalizer.clean_rows(messy_rows)
    print(f"Extracted 'total receipts' from {len(cleaned_rows)} rows:")
    for i, row in enumerate(cleaned_rows):
        result = parser.extract_value(row, "total receipts", source_table="FFO-1")
        print(
            f"  Row {i}: value={result['value']}, confidence={result['confidence']:.2f}, "
            f"strategy={result['match_strategy']}"
        )

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
