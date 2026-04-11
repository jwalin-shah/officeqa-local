"""Data Normalizer: Cleans and standardizes messy markdown table data."""

import re
from typing import Any


class DataNormalizer:
    """Normalizes and cleans table data extracted from markdown."""

    # Common patterns for currency/numbers
    CURRENCY_PATTERN = r"[\$\€\¥\£]"
    THOUSAND_SEP = r"[,\s]+"
    PERCENTAGE_PATTERN = r"%"
    NEGATIVE_PARENS = r"\(([0-9]+(?:[,\s]*[0-9])*(?:\.[0-9]+)?)\)"  # (123) means negative

    def normalize_row_label(self, label: str) -> str:
        """
        Normalize row labels for consistent matching.

        Cleans:
        - Leading/trailing whitespace
        - Multiple spaces → single space
        - Converts to lowercase
        - Removes extra characters (pipes, dashes)
        """
        # Strip whitespace
        label = label.strip()

        # Remove markdown pipes and artifacts
        label = label.replace("|", " ")
        label = label.replace("--", " ")

        # Collapse multiple spaces
        label = re.sub(r"\s+", " ", label)

        # Remove trailing/leading dashes
        label = re.sub(r"^-+\s*", "", label)
        label = re.sub(r"\s*-+$", "", label)

        # Lowercase for matching
        label = label.lower()

        return label.strip()

    def normalize_numeric_value(self, value: str, value_format: str = "number") -> float | None:
        """
        Parse and normalize numeric values from tables.

        Handles:
        - Currency symbols ($, €, ¥, £)
        - Thousand separators (commas, spaces)
        - Percentages
        - Negative numbers (parentheses style: (123) = -123)
        - Missing values (nan, N/A, empty, etc.)

        Args:
            value: Raw value string from table
            value_format: "number", "currency", "percentage"

        Returns:
            Parsed float, or None if invalid/missing
        """

        # Handle missing/invalid values
        if not value or value is None:
            return None

        value_str = str(value).strip()

        if value_str.lower() in ["nan", "n/a", "na", "", "none", "-", "--", "n/m"]:
            return None

        original_value = value_str

        # Check for negative using parentheses style: (123) = -123
        is_negative = False
        match = re.match(self.NEGATIVE_PARENS, value_str)
        if match:
            is_negative = True
            value_str = match.group(1)

        # Remove currency symbols
        value_str = re.sub(self.CURRENCY_PATTERN, "", value_str)

        # Remove percentage sign (handle separately)
        is_percentage = "%" in value_str
        value_str = value_str.replace("%", "")

        # Remove thousand separators (commas and spaces)
        value_str = re.sub(self.THOUSAND_SEP, "", value_str)

        # Remove any remaining whitespace
        value_str = value_str.strip()

        # Try to parse as float
        try:
            number = float(value_str)

            # Apply negative if parentheses indicated
            if is_negative:
                number = -number

            # If percentage, return as decimal (95% = 0.95) or keep as number?
            # For now, keep as number
            if is_percentage and number > 100:
                number = number / 100

            return number

        except ValueError:
            # If parsing fails, return None
            return None

    def normalize_row_dict(
        self, row: dict[str, str], row_schema: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Normalize an entire row dict from a table.

        Args:
            row: Dict from table {column_name: value}
            row_schema: Optional list of column names that should be numeric

        Returns:
            Cleaned row dict with:
            - Row labels normalized
            - Numeric values parsed
            - Maintains structure
        """

        cleaned = {}

        for col_name, value in row.items():
            # Normalize column names (lowercase, clean)
            clean_col_name = self.normalize_row_label(col_name)

            # Decide if this should be numeric
            # Heuristic: column name contains "value", "receipts", "outlays", "amount", "total", "rate", "percent"
            numeric_keywords = [
                "receipts",
                "outlays",
                "value",
                "amount",
                "total",
                "deficit",
                "surplus",
                "rate",
                "percent",
                "balance",
                "transaction",
                "financing",
            ]
            is_numeric = any(kw in clean_col_name for kw in numeric_keywords)

            # Try numeric parsing if looks numeric
            if is_numeric:
                numeric_val = self.normalize_numeric_value(value)
                if numeric_val is not None:
                    cleaned[clean_col_name] = numeric_val
                else:
                    cleaned[clean_col_name] = None
            else:
                # Keep as string, but clean whitespace
                cleaned[clean_col_name] = self.normalize_row_label(value)

        return cleaned

    def validate_table_structure(
        self, rows: list[dict[str, str]], required_columns: list[str] | None = None
    ) -> tuple[bool, str]:
        """
        Validate that a table has expected structure.

        Checks:
        - At least one row
        - All rows have same column count
        - Required columns present (if specified)

        Returns:
            (is_valid, message)
        """

        if not rows:
            return (False, "Table has no rows")

        if len(rows) == 0:
            return (False, "Empty table")

        # Check column consistency
        first_cols = set(rows[0].keys())
        for i, row in enumerate(rows[1:], 1):
            row_cols = set(row.keys())
            if row_cols != first_cols:
                return (False, f"Row {i} has different columns: {row_cols} vs {first_cols}")

        # Check required columns
        if required_columns:
            normalized_required = {self.normalize_row_label(c) for c in required_columns}
            normalized_actual = {self.normalize_row_label(c) for c in rows[0].keys()}

            missing = normalized_required - normalized_actual
            if missing:
                return (False, f"Missing required columns: {missing}")

        return (True, "Table structure valid")

    def clean_rows(
        self, rows: list[dict[str, str]], required_columns: list[str] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Clean a list of rows from a table.

        Args:
            rows: List of row dicts
            required_columns: Columns that must be present

        Returns:
            (cleaned_rows, metadata)
            - cleaned_rows: normalized rows
            - metadata: {
                'total_rows': int,
                'cleaned_rows': int,
                'invalid_rows': int,
                'structure_valid': bool,
                'errors': [list of errors]
              }
        """

        errors = []

        # Validate structure first
        is_valid, msg = self.validate_table_structure(rows, required_columns)
        if not is_valid:
            errors.append(msg)

        # Clean rows
        cleaned = []
        invalid_count = 0

        for i, row in enumerate(rows):
            try:
                cleaned_row = self.normalize_row_dict(row)
                cleaned.append(cleaned_row)
            except Exception as e:
                invalid_count += 1
                errors.append(f"Row {i} failed to clean: {str(e)}")

        return (
            cleaned,
            {
                "total_rows": len(rows),
                "cleaned_rows": len(cleaned),
                "invalid_rows": invalid_count,
                "structure_valid": is_valid,
                "errors": errors,
            },
        )


# Example usage
if __name__ == "__main__":
    normalizer = DataNormalizer()

    # Test 1: Normalize row labels
    labels = ["  Total Receipts  ", "Total|Receipts", "TOTAL RECEIPTS", "Total -- Receipts"]
    print("Label Normalization:")
    for label in labels:
        print(f"  '{label}' → '{normalizer.normalize_row_label(label)}'")
    print()

    # Test 2: Normalize numeric values
    values = [
        "1,234,567.89",
        "$1,234,567.89",
        "(1,234,567.89)",  # negative
        "95%",
        "nan",
        "N/A",
        "",
    ]
    print("Numeric Normalization:")
    for v in values:
        result = normalizer.normalize_numeric_value(v)
        print(f"  '{v}' → {result}")
    print()

    # Test 3: Clean full rows
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
    print("Full Row Cleaning:")
    cleaned, meta = normalizer.clean_rows(messy_rows)
    for row in cleaned:
        print(f"  {row}")
    print(f"Metadata: {meta}")
