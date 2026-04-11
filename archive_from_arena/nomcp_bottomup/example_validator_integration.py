"""
Example: Integrating TableValidator into a decomposition/search pipeline.

This demonstrates how to use the TableValidator to gate table acceptance
during the decomposition or evidence selection phase.
"""

from typing import Any

from data_normalizer import DataNormalizer
from table_finder import TableFinder
from table_validator import TableValidator


class ValidatingTableExtractor:
    """
    Extracts and validates tables from Treasury Bulletin files.

    Integrates TableFinder, DataNormalizer, and TableValidator into
    a single pipeline that returns only valid, high-quality tables.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_valid_confidence: float = 0.7,
        strict_mode: bool = False,
    ):
        """
        Initialize extractor.

        Args:
            min_confidence: Accept tables with confidence >= this (default 0.5)
            min_valid_confidence: Require this confidence if strict_mode (default 0.7)
            strict_mode: If True, only accept valid=True tables (default False)
        """
        self.finder = TableFinder()
        self.normalizer = DataNormalizer()
        self.validator = TableValidator()

        self.min_confidence = min_confidence
        self.min_valid_confidence = min_valid_confidence
        self.strict_mode = strict_mode

    def extract_and_validate(
        self, filepath: str, table_id: str, required_columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Extract and validate a table from a Treasury Bulletin file.

        Args:
            filepath: Path to Treasury Bulletin text file
            table_id: Table identifier (e.g., "FFO-1")
            required_columns: Optional list of required columns

        Returns:
            Dict with structure:
            {
                'success': bool,
                'table_id': str,
                'filepath': str,
                'rows': List[Dict] | None,  # None if not extracted
                'validation': Dict,          # Full validation result
                'reason': str,               # Why extraction failed (if failed)
                'quality': 'high' | 'acceptable' | 'poor' | 'invalid'
            }
        """

        result = {
            "success": False,
            "table_id": table_id,
            "filepath": filepath,
            "rows": None,
            "validation": {},
            "reason": "",
            "quality": "invalid",
        }

        # Step 1: Find table
        content, find_meta = self.finder.find_table(filepath, table_id)
        if not content:
            result["reason"] = f"Table not found: {find_meta.get('note', 'Unknown')}"
            return result

        # Step 2: Extract rows
        raw_rows = self.finder.extract_rows(content)
        if not raw_rows:
            result["reason"] = "No rows extracted from table content"
            return result

        # Step 3: Normalize rows
        cleaned_rows, norm_meta = self.normalizer.clean_rows(
            raw_rows, required_columns=required_columns
        )

        if not norm_meta["structure_valid"]:
            result["reason"] = f"Table structure invalid: {norm_meta['errors']}"
            return result

        # Step 4: Validate quality
        validation = self.validator.validate(
            cleaned_rows, table_name=table_id, required_columns=required_columns
        )

        result["validation"] = validation

        # Step 5: Accept/reject based on quality
        confidence = validation["confidence"]
        is_valid = validation["is_valid"]

        if self.strict_mode:
            # Strict mode: only accept valid tables
            if is_valid and confidence >= self.min_valid_confidence:
                result["success"] = True
                result["rows"] = cleaned_rows
                result["quality"] = "high" if confidence > 0.9 else "acceptable"
            else:
                result["reason"] = (
                    f"Table not valid or confidence too low "
                    f"(valid={is_valid}, confidence={confidence:.2f})"
                )
                result["quality"] = "invalid"
        else:
            # Permissive mode: accept tables above confidence threshold
            if confidence >= self.min_confidence:
                result["success"] = True
                result["rows"] = cleaned_rows
                if is_valid:
                    result["quality"] = "high" if confidence > 0.9 else "acceptable"
                else:
                    result["quality"] = "poor"
            else:
                result["reason"] = f"Confidence too low: {confidence:.2f} < {self.min_confidence}"
                result["quality"] = "invalid"

        return result

    def extract_multiple(
        self,
        filepath: str,
        table_ids: list[str],
        required_columns_map: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract and validate multiple tables from same file.

        Args:
            filepath: Path to Treasury Bulletin file
            table_ids: List of table IDs to extract
            required_columns_map: Optional dict mapping table_id -> required columns

        Returns:
            List of extraction results
        """

        results = []
        for table_id in table_ids:
            req_cols = None
            if required_columns_map and table_id in required_columns_map:
                req_cols = required_columns_map[table_id]

            result = self.extract_and_validate(filepath, table_id, req_cols)
            results.append(result)

        return results


# Example usage
def example_basic():
    """Basic example: Extract single table."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Table Extraction")
    print("=" * 70)

    extractor = ValidatingTableExtractor(min_confidence=0.5)

    filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"
    result = extractor.extract_and_validate(filepath, "FFO-1")

    print(f"Success: {result['success']}")
    print(f"Table: {result['table_id']}")
    print(f"Quality: {result['quality']}")
    print(f"Reason: {result['reason']}")

    if result["success"]:
        validation = result["validation"]
        print("\nValidation Results:")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Confidence: {validation['confidence']:.2f}")
        print(f"  Rows: {validation['row_count']}")
        print(f"  Columns: {validation['column_count']}")
        print(f"  Complete rows: {validation['quality_metrics']['complete_rows_pct']:.0f}%")

        if result["rows"]:
            print("\nFirst row sample:")
            first_row = result["rows"][0]
            for key, value in list(first_row.items())[:3]:
                print(f"    {key}: {value}")
    print()


def example_strict_mode():
    """Strict mode example: Only accept valid tables."""
    print("=" * 70)
    print("EXAMPLE 2: Strict Mode (Only Valid Tables)")
    print("=" * 70)

    extractor = ValidatingTableExtractor(
        min_confidence=0.7, min_valid_confidence=0.7, strict_mode=True
    )

    filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"
    result = extractor.extract_and_validate(filepath, "FFO-1")

    print(f"Success: {result['success']}")
    print(f"Quality: {result['quality']}")
    if not result["success"]:
        print(f"Reason: {result['reason']}")
    print()


def example_multiple_tables():
    """Multiple tables example: Extract several tables from same file."""
    print("=" * 70)
    print("EXAMPLE 3: Extract Multiple Tables")
    print("=" * 70)

    extractor = ValidatingTableExtractor(min_confidence=0.5)

    filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"
    table_ids = ["FFO-1", "FFO-2"]

    results = extractor.extract_multiple(filepath, table_ids)

    for result in results:
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"{status} | {result['table_id']} | Quality: {result['quality']}")
        if result["success"]:
            validation = result["validation"]
            print(
                f"         Confidence: {validation['confidence']:.2f} | "
                f"Rows: {validation['row_count']} | "
                f"Complete: {validation['quality_metrics']['complete_rows_pct']:.0f}%"
            )
        else:
            print(f"         Reason: {result['reason']}")
    print()


def example_with_required_columns():
    """Required columns example: Validate specific columns are present."""
    print("=" * 70)
    print("EXAMPLE 4: Required Columns Validation")
    print("=" * 70)

    extractor = ValidatingTableExtractor(min_confidence=0.5)

    filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"

    # Try with columns that should exist
    result = extractor.extract_and_validate(
        filepath, "FFO-1", required_columns=["fiscal year", "total receipts"]
    )

    print("With expected columns:")
    print(f"  Success: {result['success']}")
    print(f"  Quality: {result['quality']}")

    # Try with columns that probably don't exist
    result = extractor.extract_and_validate(
        filepath, "FFO-1", required_columns=["nonexistent_column"]
    )

    print("\nWith nonexistent column:")
    print(f"  Success: {result['success']}")
    print(f"  Quality: {result['quality']}")
    print(f"  Reason: {result['reason']}")
    print()


def example_pipeline_integration():
    """
    Pipeline integration example: How to use in decomposition/search.

    This shows how to integrate the validator into a larger pipeline.
    """
    print("=" * 70)
    print("EXAMPLE 5: Pipeline Integration")
    print("=" * 70)

    print("""
Typical decomposition/search pipeline:

1. Planner suggests tables (e.g., "FFO-1")
2. Searcher locates tables in corpus
3. Validator checks quality
   - If valid (confidence > 0.7): Use for analysis
   - If marginal (0.5-0.7): Use but log warning
   - If poor (< 0.5): Skip, try alternative
4. Analyzer extracts values from valid tables
5. Aggregator sums/totals values

Example code:

    planner_suggestions = ["FFO-1", "FFO-2", "receipt-summary"]

    for table_id in planner_suggestions:
        for filepath in search_corpus(table_id):
            result = extractor.extract_and_validate(filepath, table_id)

            if result['success']:
                if result['quality'] == 'high':
                    # Proceed normally
                    analyze_table(result['rows'])
                elif result['quality'] == 'acceptable':
                    # Log warning but use
                    log_warning(f"Low confidence for {table_id}")
                    analyze_table(result['rows'])
                # else: don't use
            else:
                # Try next corpus file
                continue_search()
    """)
    print()


if __name__ == "__main__":
    try:
        example_basic()
    except FileNotFoundError:
        print("Corpus file not found - example_basic skipped\n")

    try:
        example_strict_mode()
    except FileNotFoundError:
        print("Corpus file not found - example_strict_mode skipped\n")

    try:
        example_multiple_tables()
    except FileNotFoundError:
        print("Corpus file not found - example_multiple_tables skipped\n")

    try:
        example_with_required_columns()
    except FileNotFoundError:
        print("Corpus file not found - example_with_required_columns skipped\n")

    example_pipeline_integration()
