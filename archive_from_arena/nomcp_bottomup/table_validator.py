"""Table Validator: Validates table structure, completeness, and data quality."""

import re
from collections import defaultdict
from typing import Any


class TableValidator:
    """Validates Treasury Bulletin tables for structural integrity, completeness, and quality."""

    # Thresholds for validation
    MIN_ROWS = 5
    MIN_COMPLETE_ROWS_PCT = 50
    MAX_NULL_PCT_OVERALL = 50
    MAX_NULL_PCT_COLUMN = 70
    MIN_NUMERIC_COLUMNS = 1

    def __init__(self):
        """Initialize validator with default thresholds."""
        self.min_rows = self.MIN_ROWS
        self.min_complete_pct = self.MIN_COMPLETE_ROWS_PCT
        self.max_null_pct = self.MAX_NULL_PCT_OVERALL
        self.max_null_pct_column = self.MAX_NULL_PCT_COLUMN

    def validate(
        self,
        rows: list[dict[str, Any]],
        table_name: str,
        required_columns: list[str] | None = None,
        expected_row_count: int | None = None,
    ) -> dict[str, Any]:
        """
        Validate table structure and completeness.

        Args:
            rows: List of row dicts from table extraction
            table_name: Name/ID of the table (e.g., "FFO-1")
            required_columns: Optional list of required column names (normalized)
            expected_row_count: Optional expected minimum row count

        Returns:
            Dict with structure:
            {
                'is_valid': bool,
                'confidence': float (0.0-1.0),
                'table_name': str,
                'row_count': int,
                'column_count': int,
                'structural_issues': [list of strings],
                'completeness_issues': [list of strings],
                'quality_metrics': dict,
                'recommendations': [list of strings]
            }
        """

        result = {
            "is_valid": True,
            "confidence": 1.0,
            "table_name": table_name,
            "row_count": len(rows),
            "column_count": len(rows[0].keys()) if rows else 0,
            "structural_issues": [],
            "completeness_issues": [],
            "quality_metrics": {},
            "recommendations": [],
        }

        # Handle empty table
        if not rows:
            result["is_valid"] = False
            result["confidence"] = 0.0
            result["structural_issues"].append("Table is empty (0 rows)")
            return result

        # Check structural integrity
        struct_ok, struct_issues = self.check_structural_integrity(rows)
        result["structural_issues"].extend(struct_issues)
        result["is_valid"] = result["is_valid"] and struct_ok

        # Check completeness
        complete_issues = self.check_completeness(
            rows, required_columns=required_columns, expected_row_count=expected_row_count
        )
        result["completeness_issues"].extend(complete_issues)
        result["is_valid"] = result["is_valid"] and (len(complete_issues) == 0)

        # Check data quality
        quality_metrics = self.check_data_quality(rows)
        result["quality_metrics"] = quality_metrics

        # Update validity based on quality thresholds
        if quality_metrics.get("null_percentage", 0) > self.max_null_pct:
            result["is_valid"] = False
            result["completeness_issues"].append(
                f"Too many missing values: {quality_metrics['null_percentage']:.1f}% > {self.max_null_pct}%"
            )

        if quality_metrics.get("numeric_columns", 0) < self.MIN_NUMERIC_COLUMNS:
            result["is_valid"] = False
            result["structural_issues"].append("No numeric columns found")

        # Calculate confidence score
        result["confidence"] = self.score_table_quality(
            result["structural_issues"], result["completeness_issues"], quality_metrics
        )

        # Generate recommendations
        result["recommendations"] = self._generate_recommendations(result)

        return result

    def check_structural_integrity(self, rows: list[dict]) -> tuple[bool, list[str]]:
        """
        Check that table has consistent structure.

        Validates:
        - All rows have same number of columns
        - No completely empty rows
        - Column headers are reasonable (not empty, not all special chars)

        Returns:
            (is_valid, issues_list)
        """

        issues = []
        is_valid = True

        if not rows:
            return (False, ["Empty table"])

        # Check column consistency
        first_col_count = len(rows[0])
        for i, row in enumerate(rows[1:], 1):
            if len(row) != first_col_count:
                issues.append(f"Row {i} has {len(row)} columns, expected {first_col_count}")
                is_valid = False

        # Check for completely empty rows
        for i, row in enumerate(rows):
            if all(
                v is None or v == "" or (isinstance(v, str) and v.strip() == "")
                for v in row.values()
            ):
                issues.append(f"Row {i} is completely empty")
                is_valid = False

        # Check column headers
        first_row_headers = list(rows[0].keys())
        for header in first_row_headers:
            # Header should not be empty
            if not header or not header.strip():
                issues.append("Empty column header found")
                is_valid = False
            # Header should not be all special characters
            elif not re.search(r"[a-zA-Z0-9]", header):
                issues.append(f'Column header has no alphanumeric chars: "{header}"')
                is_valid = False

        return (is_valid, issues)

    def check_completeness(
        self,
        rows: list[dict],
        required_columns: list[str] | None = None,
        expected_row_count: int | None = None,
    ) -> list[str]:
        """
        Check table completeness.

        Validates:
        - Minimum number of data rows
        - Required columns are present
        - Not too many missing values

        Returns:
            List of issue strings
        """

        issues = []

        # Check minimum row count
        min_rows = expected_row_count if expected_row_count else self.min_rows
        if len(rows) < min_rows:
            issues.append(f"Too few rows: {len(rows)} < {min_rows}")

        # Check required columns
        if required_columns:
            actual_cols = set(rows[0].keys())
            required_lower = {col.lower() for col in required_columns}
            actual_lower = {col.lower() for col in actual_cols}

            missing = required_lower - actual_lower
            if missing:
                issues.append(f"Missing required columns: {missing}")

        # Check for columns with too many missing values
        col_null_counts = defaultdict(int)
        for row in rows:
            for col, value in row.items():
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    col_null_counts[col] += 1

        for col, null_count in col_null_counts.items():
            null_pct = (null_count / len(rows)) * 100
            if null_pct > self.max_null_pct_column:
                issues.append(
                    f'Column "{col}" is {null_pct:.1f}% empty (>{self.max_null_pct_column}%)'
                )

        return issues

    def check_data_quality(self, rows: list[dict]) -> dict[str, Any]:
        """
        Check data quality metrics.

        Analyzes:
        - Percentage of missing values
        - Percentage of complete rows
        - Number and consistency of numeric columns
        - Value range reasonableness

        Returns:
            Dict with metrics:
            {
                'null_percentage': float,
                'null_by_column': dict,
                'complete_rows': int,
                'complete_rows_pct': float,
                'numeric_columns': int,
                'numeric_values_count': int,
                'numeric_values_complete': int,
                'data_range_check': str ('PASS', 'WARNING', 'FAIL'),
                'all_zeros': bool,
                'value_ranges': dict
            }
        """

        metrics = {
            "null_percentage": 0.0,
            "null_by_column": {},
            "complete_rows": 0,
            "complete_rows_pct": 0.0,
            "numeric_columns": 0,
            "numeric_values_count": 0,
            "numeric_values_complete": 0,
            "data_range_check": "PASS",
            "all_zeros": False,
            "value_ranges": {},
        }

        if not rows:
            return metrics

        total_cells = 0
        null_cells = 0
        complete_row_count = 0
        numeric_col_indices = {}
        col_names = list(rows[0].keys())

        # First pass: identify numeric columns
        for col_idx, col_name in enumerate(col_names):
            numeric_count = 0
            for row in rows:
                value = row.get(col_name)
                if self._is_numeric(value):
                    numeric_count += 1

            # Column is numeric if >70% of values are numeric
            if numeric_count > len(rows) * 0.7:
                numeric_col_indices[col_idx] = col_name

        metrics["numeric_columns"] = len(numeric_col_indices)

        # Second pass: collect metrics
        for row_idx, row in enumerate(rows):
            row_null_count = 0

            for col_idx, col_name in enumerate(col_names):
                value = row.get(col_name)
                total_cells += 1

                # Track null cells
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    null_cells += 1
                    row_null_count += 1
                else:
                    # Track numeric values
                    if col_idx in numeric_col_indices:
                        metrics["numeric_values_count"] += 1
                        if self._is_numeric(value):
                            metrics["numeric_values_complete"] += 1

            # Track complete rows
            if row_null_count == 0:
                complete_row_count += 1

        # Calculate percentages
        if total_cells > 0:
            metrics["null_percentage"] = (null_cells / total_cells) * 100
        metrics["complete_rows"] = complete_row_count
        metrics["complete_rows_pct"] = (complete_row_count / len(rows)) * 100 if rows else 0.0

        # Per-column null percentages
        for col_idx, col_name in enumerate(col_names):
            col_null_count = 0
            for row in rows:
                value = row.get(col_name)
                if value is None or value == "" or (isinstance(value, str) and value.strip() == ""):
                    col_null_count += 1
            null_pct = (col_null_count / len(rows)) * 100 if rows else 0.0
            metrics["null_by_column"][col_name] = null_pct

        # Check value ranges for numeric columns
        all_zeros = True
        all_zeros_cols = []  # Track which columns are all zeros
        value_ranges = {}
        for col_idx, col_name in numeric_col_indices.items():
            numeric_values = []
            for row in rows:
                value = row.get(col_name)
                if self._is_numeric(value):
                    try:
                        numeric_val = float(value) if isinstance(value, str) else value
                        numeric_values.append(numeric_val)
                        if numeric_val != 0:
                            all_zeros = False
                    except (ValueError, TypeError):
                        pass

            if numeric_values:
                min_val = min(numeric_values)
                max_val = max(numeric_values)
                value_ranges[col_name] = {
                    "min": min_val,
                    "max": max_val,
                    "mean": sum(numeric_values) / len(numeric_values),
                }
                # Track columns that are entirely zeros
                if min_val == 0 and max_val == 0:
                    all_zeros_cols.append(col_name)

        metrics["value_ranges"] = value_ranges
        metrics["all_zeros"] = all_zeros

        # Determine data range check
        if (
            all_zeros
            and metrics["numeric_columns"] > 0
            or metrics["numeric_values_complete"] == 0
            and metrics["numeric_columns"] > 0
        ):
            metrics["data_range_check"] = "FAIL"

        return metrics

    def score_table_quality(
        self,
        structural_issues: list[str],
        completeness_issues: list[str],
        quality_metrics: dict[str, Any],
    ) -> float:
        """
        Calculate overall confidence score (0.0-1.0).

        Scoring logic:
        - Start at 1.0
        - Deduct for structural issues (-0.3 each, max -1.0)
        - Deduct for completeness issues (-0.2 each, max -0.5)
        - Deduct for quality issues (based on null%, coverage, etc.)

        Returns:
            Confidence score 0.0-1.0
        """

        confidence = 1.0

        # Structural issues are critical
        for _ in structural_issues:
            confidence -= 0.3
        confidence = max(0.0, confidence)

        # Completeness issues are moderate
        for _ in completeness_issues:
            confidence -= 0.2
        confidence = max(0.0, confidence)

        # Quality penalizations
        null_pct = quality_metrics.get("null_percentage", 0)
        if null_pct > 50:
            confidence -= 0.3
        elif null_pct > 30:
            confidence -= 0.15
        elif null_pct > 10:
            confidence -= 0.05

        # Penalize for incomplete rows
        complete_pct = quality_metrics.get("complete_rows_pct", 100)
        if complete_pct < 30:
            confidence -= 0.2
        elif complete_pct < 60:
            confidence -= 0.1

        # Penalize for data range issues
        if quality_metrics.get("data_range_check") == "FAIL":
            confidence -= 0.3
        elif quality_metrics.get("data_range_check") == "WARNING":
            confidence -= 0.1

        # Bonus for good data
        if null_pct < 5 and complete_pct > 90:
            confidence += 0.1

        return max(0.0, min(1.0, confidence))

    def _is_numeric(self, value: Any) -> bool:
        """Check if a value is numeric or can be converted to numeric."""
        if value is None or value == "":
            return False

        if isinstance(value, (int, float)):
            return True

        if isinstance(value, str):
            # Try to parse as float
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False

        return False

    def _generate_recommendations(self, result: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on validation results."""

        recommendations = []
        metrics = result.get("quality_metrics", {})

        # Positive recommendations
        if result["is_valid"]:
            complete_pct = metrics.get("complete_rows_pct", 0)
            if complete_pct > 90:
                recommendations.append(
                    f"Excellent completeness: {complete_pct:.0f}% of rows have all values"
                )
            elif complete_pct > 70:
                recommendations.append(
                    f"Good completeness: {complete_pct:.0f}% of rows have all values"
                )

        # Issues and recommendations
        if metrics.get("null_percentage", 0) > 30:
            recommendations.append(
                f"High missing data ({metrics['null_percentage']:.1f}%): "
                "Consider filtering out sparse columns"
            )

        if metrics.get("numeric_columns", 0) == 0:
            recommendations.append(
                "No numeric columns detected: Table may be reference/lookup only"
            )

        if metrics.get("all_zeros"):
            recommendations.append("All numeric values are zero: Verify data is correct")

        if result["row_count"] < 10:
            recommendations.append(
                f"Small table ({result['row_count']} rows): May have limited statistical relevance"
            )

        if result["row_count"] >= 20:
            recommendations.append(
                f"Table spans multiple years ({result['row_count']} rows): "
                "Good for time-series analysis"
            )

        return recommendations


# Example usage
if __name__ == "__main__":
    from data_normalizer import DataNormalizer
    from table_finder import TableFinder

    finder = TableFinder()
    normalizer = DataNormalizer()
    validator = TableValidator()

    # Test with real data
    filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"

    # Find and extract FFO-1 table
    content, meta = finder.find_table(filepath, "FFO-1")

    if content:
        rows = finder.extract_rows(content)
        cleaned_rows, clean_meta = normalizer.clean_rows(rows)

        print(f"Extracted {len(rows)} raw rows, {len(cleaned_rows)} cleaned rows")
        print()

        # Validate
        result = validator.validate(cleaned_rows, "FFO-1")

        print(f"Table: {result['table_name']}")
        print(f"Valid: {result['is_valid']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Dimensions: {result['row_count']} rows x {result['column_count']} cols")
        print()

        if result["structural_issues"]:
            print("Structural Issues:")
            for issue in result["structural_issues"]:
                print(f"  - {issue}")
            print()

        if result["completeness_issues"]:
            print("Completeness Issues:")
            for issue in result["completeness_issues"]:
                print(f"  - {issue}")
            print()

        print("Quality Metrics:")
        for key, value in result["quality_metrics"].items():
            if key not in ["null_by_column", "value_ranges"]:
                print(f"  {key}: {value}")
        print()

        if result["recommendations"]:
            print("Recommendations:")
            for rec in result["recommendations"]:
                print(f"  - {rec}")
