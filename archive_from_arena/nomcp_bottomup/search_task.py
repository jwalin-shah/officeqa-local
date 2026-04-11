"""Search Task: Orchestrates the complete extraction pipeline for a single decomposition piece.

This is the main coordinator that:
1. Locates file(s) for the given year/month
2. Finds and validates the table
3. Spawns 3 parallel subsearch strategies (Strict, Fuzzy, Contextual)
4. Votes on the best result using ConsensusVoter
5. Returns the winning value with full transparency and metadata

Architecture:
  Decomposition Piece
    ↓
  File Locator: Find file(s) for year
    ↓
  For EACH file (parallel):
    - Table Finder: Locate table
    - Data Normalizer: Clean rows
    - Table Validator: Check quality
    - If valid, spawn 3 subsearches (parallel):
      - Subsearch A (Strict): Exact column + row matching
      - Subsearch B (Fuzzy): Fuzzy column + row matching
      - Subsearch C (Contextual): Contextual column + row matching
    - Consensus Voter: Pick best from A/B/C
  ↓
  Compare results across files
  ↓
  Return winning value + confidence
"""

from pathlib import Path
from typing import Any

from column_finder import ColumnFinder
from consensus_voter import ConsensusVoter
from data_normalizer import DataNormalizer
from file_locator import FileLocator
from row_matcher import RowMatcher
from table_finder import TableFinder
from table_validator import TableValidator
from value_parser import ValueParser


class SearchTask:
    """Orchestrates the full search pipeline for one decomposition piece."""

    def __init__(self):
        """Initialize all components."""
        self.file_locator = FileLocator()
        self.table_finder = TableFinder()
        self.data_normalizer = DataNormalizer()
        self.table_validator = TableValidator()
        self.row_matcher = RowMatcher()
        self.column_finder = ColumnFinder()
        self.value_parser = ValueParser()
        self.consensus_voter = ConsensusVoter()

    def execute(self, decomposition_piece: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        """
        Execute full search pipeline for one decomposition piece.

        Args:
            decomposition_piece: {
                'piece_id': int,
                'year': int,
                'period_type': 'fiscal' | 'calendar',
                'table_id': str,
                'row_identifier': [list of str] or str,
                'column_identifier': str,
                'prefer_months': [list of months] (optional),
                'row_index': int (optional)
            }
            verbose: Print step-by-step progress

        Returns:
            {
                'piece_id': int,
                'value': float or None,
                'confidence': float,
                'success': bool,
                'source': {
                    'file': str,
                    'table': str,
                    'row_index': int,
                    'column': str
                },
                'steps': [list of step descriptions],
                'file_results': [
                    {
                        'file': str,
                        'value': float,
                        'confidence': float,
                        'success': bool
                    }
                ],
                'voting_details': {
                    'strict_result': {...},
                    'fuzzy_result': {...},
                    'contextual_result': {...},
                    'consensus': {...}
                },
                'errors': [list of error messages]
            }
        """
        result = {
            "piece_id": decomposition_piece.get("piece_id"),
            "value": None,
            "confidence": 0.0,
            "success": False,
            "source": None,
            "steps": [],
            "file_results": [],
            "voting_details": None,
            "errors": [],
        }

        try:
            # Extract piece metadata
            year = decomposition_piece.get("year")
            table_id = decomposition_piece.get("table_id")
            row_identifier = decomposition_piece.get("row_identifier")
            column_identifier = decomposition_piece.get("column_identifier")
            prefer_months = decomposition_piece.get("prefer_months")
            row_index = decomposition_piece.get("row_index")

            if verbose:
                print(f"\n=== SearchTask: Piece {decomposition_piece.get('piece_id')} ===")
                print(f"Year: {year}, Table: {table_id}")
                print(f"Row: {row_identifier}, Column: {column_identifier}")

            # Step 1: Validate inputs
            if not year or not table_id or not row_identifier or not column_identifier:
                result["errors"].append(
                    f"Missing required fields: year={year}, table={table_id}, "
                    f"row={row_identifier}, column={column_identifier}"
                )
                if verbose:
                    print(f"  ❌ {result['errors'][-1]}")
                return result

            # Normalize row_identifier to list
            if isinstance(row_identifier, str):
                row_identifier = [row_identifier]

            result["steps"].append(f"Validating inputs: year={year}, table={table_id}")

            # Step 2: Locate file(s)
            file_path, file_meta = self.file_locator.locate_file(
                year=year,
                month=decomposition_piece.get("month"),
                prefer_months=prefer_months,
                return_all=False,
            )

            # Handle ambiguous case: multiple candidates
            candidates = file_meta.get("candidates", [])
            if file_path is None and candidates:
                # Try all candidates
                file_candidates = candidates
                result["steps"].append(
                    f"Found {len(file_candidates)} file candidates for year {year}"
                )
                if verbose:
                    print(f"  📁 Found {len(file_candidates)} candidates for year {year}")
            elif file_path:
                file_candidates = [file_path]
                result["steps"].append(f"Found file: {Path(file_path).name}")
                if verbose:
                    print(f"  📁 Found file: {Path(file_path).name}")
            else:
                result["errors"].append(f"No files found for year {year}: {file_meta.get('note')}")
                if verbose:
                    print(f"  ❌ {result['errors'][-1]}")
                return result

            # Step 3: Process each file candidate
            file_results = []
            best_file_result = None
            best_file_confidence = 0.0

            for file_idx, file_path in enumerate(file_candidates):
                if verbose:
                    print(
                        f"\n  Processing file {file_idx + 1}/{len(file_candidates)}: {Path(file_path).name}"
                    )

                file_result = self._process_file(
                    file_path=file_path,
                    table_id=table_id,
                    row_identifier=row_identifier,
                    column_identifier=column_identifier,
                    row_index=row_index,
                    verbose=verbose,
                )

                if file_result["success"]:
                    file_results.append(file_result)

                    # Track best result
                    if file_result["confidence"] > best_file_confidence:
                        best_file_confidence = file_result["confidence"]
                        best_file_result = file_result

                result["file_results"].append(
                    {
                        "file": Path(file_path).name,
                        "value": file_result.get("value"),
                        "confidence": file_result.get("confidence", 0.0),
                        "success": file_result.get("success", False),
                    }
                )

            # Step 4: Consolidate results across files
            if best_file_result:
                result["success"] = True
                result["value"] = best_file_result["value"]
                result["confidence"] = best_file_result["confidence"]
                result["source"] = best_file_result.get("source")
                result["steps"].append(
                    f"Final: Found value {result['value']} with confidence {result['confidence']:.2f}"
                )
                result["voting_details"] = best_file_result.get("voting_details")

                if verbose:
                    print(
                        f"\n  ✅ SUCCESS: Value={result['value']}, Confidence={result['confidence']:.2f}"
                    )
            else:
                result["errors"].append("All file candidates failed to extract value")
                if verbose:
                    print("\n  ❌ All files failed")

            return result

        except Exception as e:
            result["errors"].append(f"Exception in execute: {str(e)}")
            if verbose:
                print(f"  ❌ Exception: {str(e)}")
            return result

    def _process_file(
        self,
        file_path: str,
        table_id: str,
        row_identifier: list[str],
        column_identifier: str,
        row_index: int | None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """
        Process a single file candidate.

        Returns:
            {
                'value': float or None,
                'confidence': float,
                'success': bool,
                'source': {...},
                'voting_details': {...},
                'errors': [...]
            }
        """
        result = {
            "value": None,
            "confidence": 0.0,
            "success": False,
            "source": None,
            "voting_details": None,
            "errors": [],
        }

        try:
            # Step 1: Find and extract table
            table_content, table_meta = self.table_finder.find_table(file_path, table_id)
            if not table_content:
                result["errors"].append(f"Table '{table_id}' not found in {Path(file_path).name}")
                if verbose:
                    print("    ❌ Table not found")
                return result

            if verbose:
                print(f"    📊 Found table: {table_meta['table_name']}")

            # Step 2: Extract and normalize rows
            raw_rows = self.table_finder.extract_rows(table_content)
            if not raw_rows:
                result["errors"].append("Could not extract rows from table")
                if verbose:
                    print("    ❌ No rows extracted")
                return result

            cleaned_rows, norm_meta = self.data_normalizer.clean_rows(raw_rows)
            if not cleaned_rows:
                result["errors"].append("All rows failed to clean")
                if verbose:
                    print("    ❌ No rows after cleaning")
                return result

            if verbose:
                print(f"    🧹 Cleaned {len(cleaned_rows)} rows")

            # Step 3: Validate table
            validation = self.table_validator.validate(cleaned_rows, table_meta["table_name"])
            if not validation["is_valid"]:
                result["errors"].append(
                    f"Table validation failed: {validation['structural_issues']}"
                )
                if verbose:
                    print("    ⚠️ Table validation failed")
                return result

            if verbose:
                print(f"    ✓ Table validation passed (confidence: {validation['confidence']:.2f})")

            # Step 4: Spawn 3 parallel subsearches
            strict_result = self._subsearch_strict(cleaned_rows, row_identifier, column_identifier)
            fuzzy_result = self._subsearch_fuzzy(cleaned_rows, row_identifier, column_identifier)
            contextual_result = self._subsearch_contextual(
                cleaned_rows, row_identifier, column_identifier, row_index
            )

            if verbose:
                print("    🔀 Subsearch results:")
                print(
                    f"      - Strict: {strict_result.get('value')} (conf: {strict_result.get('confidence', 0.0):.2f})"
                )
                print(
                    f"      - Fuzzy:  {fuzzy_result.get('value')} (conf: {fuzzy_result.get('confidence', 0.0):.2f})"
                )
                print(
                    f"      - Context: {contextual_result.get('value')} (conf: {contextual_result.get('confidence', 0.0):.2f})"
                )

            # Step 5: Vote on best result
            consensus = self.consensus_voter.vote(
                strict_result=strict_result,
                fuzzy_result=fuzzy_result,
                contextual_result=contextual_result,
            )

            if verbose:
                print(
                    f"    🗳️ Consensus: {consensus['winning_value']} "
                    f"({consensus['agreement_level']}, method: {consensus['winning_method']})"
                )

            # Step 6: Determine success and extract source info
            if consensus["winning_value"] is not None:
                result["success"] = True
                result["value"] = consensus["winning_value"]
                result["confidence"] = consensus["winning_confidence"]
                result["voting_details"] = consensus

                # Extract source information
                winning_method = consensus["winning_method"]
                winning_result = None
                if winning_method == "strict":
                    winning_result = strict_result
                elif winning_method == "fuzzy":
                    winning_result = fuzzy_result
                elif winning_method == "contextual":
                    winning_result = contextual_result

                if winning_result:
                    result["source"] = {
                        "file": Path(file_path).name,
                        "table": table_meta["table_name"],
                        "row_identifier": row_identifier,
                        "column": winning_result.get("column_matched"),
                        "method": winning_method,
                        "agreement_level": consensus["agreement_level"],
                    }
            else:
                result["errors"].append("All subsearch strategies returned None")
                if verbose:
                    print("    ❌ All subsearches failed")

            return result

        except Exception as e:
            result["errors"].append(f"Exception in _process_file: {str(e)}")
            if verbose:
                print(f"    ❌ Exception: {str(e)}")
            return result

    def _subsearch_strict(
        self, rows: list[dict[str, Any]], row_identifier: list[str], column_identifier: str
    ) -> dict[str, Any]:
        """
        Subsearch A: Strict matching strategy.

        - Exact column matching
        - Exact row matching
        - Fast and high confidence when it works
        """
        result = {
            "value": None,
            "confidence": 0.0,
            "method": "strict",
            "column_matched": None,
            "column_confidence": 0.0,
            "row_matched": False,
            "row_confidence": 0.0,
            "errors": [],
        }

        try:
            # Find row (strict)
            matched_row = self.row_matcher.find_row_strict(rows, row_identifier)
            if matched_row is None:
                result["errors"].append("No exact row match")
                return result

            result["row_matched"] = True
            result["row_confidence"] = 0.99

            # Find column (exact match preferred)
            column_name, column_conf = self.column_finder.find_column(rows, column_identifier)
            if column_name is None:
                result["errors"].append("No column match")
                return result

            result["column_matched"] = column_name
            result["column_confidence"] = column_conf

            # Extract value using value_parser
            value_result = self.value_parser.extract_value(
                row=matched_row,
                column_identifier=column_identifier,
                source_table="strict_subsearch",
            )

            if value_result.get("value") is not None:
                result["value"] = value_result["value"]
                # Confidence is combination of row + column + value quality
                result["confidence"] = min(
                    result["row_confidence"],
                    result["column_confidence"],
                    value_result.get("confidence", 0.9),
                )
            else:
                result["errors"].append("Value extraction failed or None")

            return result

        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
            return result

    def _subsearch_fuzzy(
        self, rows: list[dict[str, Any]], row_identifier: list[str], column_identifier: str
    ) -> dict[str, Any]:
        """
        Subsearch B: Fuzzy matching strategy.

        - Fuzzy column matching (keyword-based)
        - Fuzzy row matching (keyword-based)
        - Broader coverage, moderate confidence
        """
        result = {
            "value": None,
            "confidence": 0.0,
            "method": "fuzzy",
            "column_matched": None,
            "column_confidence": 0.0,
            "row_matched": False,
            "row_confidence": 0.0,
            "errors": [],
        }

        try:
            # Find row (fuzzy)
            matched_row, row_conf = self.row_matcher.find_row_fuzzy(rows, row_identifier)
            if matched_row is None:
                result["errors"].append("No fuzzy row match")
                return result

            result["row_matched"] = True
            result["row_confidence"] = row_conf

            # Find column (fuzzy matching)
            column_name, column_conf = self.column_finder.find_column(rows, column_identifier)
            if column_name is None:
                result["errors"].append("No column match")
                return result

            result["column_matched"] = column_name
            result["column_confidence"] = column_conf

            # Extract value
            value_result = self.value_parser.extract_value(
                row=matched_row, column_identifier=column_identifier, source_table="fuzzy_subsearch"
            )

            if value_result.get("value") is not None:
                result["value"] = value_result["value"]
                # Confidence is combination, weighted slightly lower for fuzzy matches
                result["confidence"] = (
                    min(
                        result["row_confidence"],
                        result["column_confidence"],
                        value_result.get("confidence", 0.8),
                    )
                    * 0.95
                )  # Slight penalty for fuzzy matching

            else:
                result["errors"].append("Value extraction failed or None")

            return result

        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
            return result

    def _subsearch_contextual(
        self,
        rows: list[dict[str, Any]],
        row_identifier: list[str],
        column_identifier: str,
        row_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Subsearch C: Contextual matching strategy.

        - Contextual column matching (semantic understanding)
        - Contextual row matching (position, structure awareness)
        - Moderate confidence, good for complex cases
        """
        result = {
            "value": None,
            "confidence": 0.0,
            "method": "contextual",
            "column_matched": None,
            "column_confidence": 0.0,
            "row_matched": False,
            "row_confidence": 0.0,
            "errors": [],
        }

        try:
            # Find row (contextual)
            matched_row, row_conf = self.row_matcher.find_row_contextual(
                rows, row_identifier, row_index=row_index
            )
            if matched_row is None:
                result["errors"].append("No contextual row match")
                return result

            result["row_matched"] = True
            result["row_confidence"] = row_conf

            # Find column (fuzzy matching, same as fuzzy for now)
            column_name, column_conf = self.column_finder.find_column(rows, column_identifier)
            if column_name is None:
                result["errors"].append("No column match")
                return result

            result["column_matched"] = column_name
            result["column_confidence"] = column_conf

            # Extract value
            value_result = self.value_parser.extract_value(
                row=matched_row,
                column_identifier=column_identifier,
                source_table="contextual_subsearch",
            )

            if value_result.get("value") is not None:
                result["value"] = value_result["value"]
                # Confidence for contextual match
                result["confidence"] = (
                    min(
                        result["row_confidence"],
                        result["column_confidence"],
                        value_result.get("confidence", 0.8),
                    )
                    * 0.90
                )  # Slight penalty for contextual matching

            else:
                result["errors"].append("Value extraction failed or None")

            return result

        except Exception as e:
            result["errors"].append(f"Exception: {str(e)}")
            return result


# Example usage and testing
if __name__ == "__main__":
    import json

    task = SearchTask()

    # Test 1: Simple valid piece
    print("=" * 80)
    print("Test 1: Valid decomposition piece")
    print("=" * 80)
    piece1 = {
        "piece_id": 1,
        "year": 1995,
        "table_id": "FFO-1",
        "row_identifier": ["Total"],
        "column_identifier": "total receipts",
        "prefer_months": [12, 6],
    }

    result1 = task.execute(piece1, verbose=True)
    print("\nResult:")
    print(
        json.dumps(
            {
                "piece_id": result1["piece_id"],
                "value": result1["value"],
                "confidence": result1["confidence"],
                "success": result1["success"],
                "source": result1["source"],
                "file_results": result1["file_results"],
                "num_errors": len(result1["errors"]),
            },
            indent=2,
        )
    )

    # Test 2: Missing required field
    print("\n" + "=" * 80)
    print("Test 2: Missing year (should fail gracefully)")
    print("=" * 80)
    piece2 = {
        "piece_id": 2,
        "table_id": "FFO-1",
        "row_identifier": ["Total"],
        "column_identifier": "total receipts",
    }

    result2 = task.execute(piece2, verbose=True)
    print("\nResult:")
    print(
        json.dumps(
            {
                "piece_id": result2["piece_id"],
                "success": result2["success"],
                "errors": result2["errors"],
            },
            indent=2,
        )
    )

    # Test 3: Invalid year (no file)
    print("\n" + "=" * 80)
    print("Test 3: Invalid year (should fail gracefully)")
    print("=" * 80)
    piece3 = {
        "piece_id": 3,
        "year": 2050,
        "table_id": "FFO-1",
        "row_identifier": ["Total"],
        "column_identifier": "total receipts",
    }

    result3 = task.execute(piece3, verbose=True)
    print("\nResult:")
    print(
        json.dumps(
            {
                "piece_id": result3["piece_id"],
                "success": result3["success"],
                "errors": result3["errors"],
            },
            indent=2,
        )
    )
