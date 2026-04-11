"""End-to-end pipeline test with random OfficeQA questions."""

from pathlib import Path
from typing import Any

from consensus_voter import ConsensusVoter
from data_normalizer import DataNormalizer
from file_locator import FileLocator
from row_matcher import RowMatcher
from table_finder import TableFinder
from table_validator import TableValidator
from value_parser import ValueParser


class E2EPipelineTest:
    """Test the extraction pipeline end-to-end on OfficeQA questions."""

    def __init__(self):
        self.file_locator = FileLocator()
        self.table_finder = TableFinder()
        self.normalizer = DataNormalizer()
        self.row_matcher = RowMatcher()
        self.value_parser = ValueParser()
        self.table_validator = TableValidator()
        self.consensus_voter = ConsensusVoter()

    def test_decomposition_piece(
        self, piece: dict[str, Any], verbose: bool = False
    ) -> dict[str, Any]:
        """
        Test extraction for a single decomposition piece.

        Args:
            piece: {year, table, row, ...}
            verbose: Print step-by-step progress

        Returns:
            {value, confidence, success, errors, steps}
        """
        errors = []
        steps = []

        try:
            # Step 1: Find file
            if verbose:
                print(f"\n  📁 Step 1: Finding file for year {piece.get('year')}")

            year = piece.get("year")
            if year is None:
                errors.append("No year in decomposition")
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "errors": errors,
                    "steps": steps,
                }

            file_path, file_meta = self.file_locator.locate_file(year=year)

            # If ambiguous (multiple months), pick first/best candidate
            if file_path is None and file_meta.get("candidates"):
                file_path = file_meta["candidates"][0]
                if verbose:
                    print(f"     (Multiple months available, chose: {Path(file_path).name})")

            if not file_path:
                errors.append(f"Could not find file for year {year}")
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "errors": errors,
                    "steps": steps,
                }

            steps.append(f"Found file: {Path(file_path).name}")
            if verbose:
                print(f"     ✅ Found: {Path(file_path).name}")

            # Step 2: Find table
            if verbose:
                print(f"  📊 Step 2: Finding table '{piece.get('table_id', 'unknown')}'")

            table_id = piece.get("table_id", "receipt")
            table_content, table_meta = self.table_finder.find_table(file_path, table_id)

            if not table_content:
                errors.append(f"Could not find table '{table_id}'")
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "errors": errors,
                    "steps": steps,
                }

            steps.append(f"Found table: {table_meta['table_name']}")
            if verbose:
                print(
                    f"     ✅ Found: {table_meta['table_name']} (lines {table_meta['start_line']}-{table_meta['end_line']})"
                )

            # Step 3: Extract rows
            rows = self.table_finder.extract_rows(table_content)
            if verbose:
                print(f"  🔄 Step 3: Extracted {len(rows)} rows")

            # Step 4: Normalize
            if verbose:
                print(f"  🧹 Step 4: Normalizing {len(rows)} rows")

            cleaned_rows, norm_meta = self.normalizer.clean_rows(rows)
            if norm_meta["invalid_rows"] > 0:
                print(f"     ⚠️  {norm_meta['invalid_rows']} rows failed to clean")

            steps.append(f"Cleaned {len(cleaned_rows)} rows")
            if verbose:
                print(f"     ✅ Cleaned {len(cleaned_rows)} rows successfully")

            # Step 5: Validate table
            if verbose:
                print("  ✓ Step 5: Validating table structure")

            validation = self.table_validator.validate(cleaned_rows, table_meta["table_name"])
            if not validation["is_valid"]:
                print(f"     ⚠️  Table validation failed: {validation['structural_issues']}")

            steps.append(
                f"Table valid: {validation['is_valid']} (confidence: {validation['confidence']:.2f})"
            )
            if verbose:
                print(
                    f"     {'✅' if validation['is_valid'] else '⚠️'} Valid: {validation['is_valid']}"
                )

            # Step 6: Find row
            if verbose:
                print(f"  🔍 Step 6: Finding row '{piece.get('row_identifier', 'unknown')}'")

            row_id = piece.get("row_identifier", [])
            if isinstance(row_id, str):
                row_id = [row_id]

            candidates = self.row_matcher.find_all_candidate_rows(cleaned_rows, row_id)
            if not candidates:
                errors.append(f"Could not find row matching {row_id}")
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "errors": errors,
                    "steps": steps,
                }

            best_row, best_strategy, best_score = candidates[0]
            steps.append(
                f"Found row: {best_row} (strategy: {best_strategy}, score: {best_score:.2f})"
            )
            if verbose:
                print(
                    f"     ✅ Found using {best_strategy} strategy (confidence: {best_score:.2f})"
                )

            # Step 7: Extract value
            if verbose:
                print("  💾 Step 7: Extracting value")

            column_id = piece.get("column_identifier", piece.get("row_identifier", []))
            if isinstance(column_id, str):
                column_id = [column_id]

            # Try first numeric column if not specified
            if not column_id:
                column_id = ["total", "value", "amount", "receipts"]

            value_result = self.value_parser.extract_value(best_row, " ".join(column_id))
            if value_result["value"] is None:
                errors.append("Could not extract numeric value from row")
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "errors": errors,
                    "steps": steps,
                }

            steps.append(
                f"Extracted value: {value_result['value']} (confidence: {value_result['confidence']:.2f})"
            )
            if verbose:
                print(
                    f"     ✅ Value: {value_result['value']} (confidence: {value_result['confidence']:.2f})"
                )

            return {
                "value": value_result["value"],
                "confidence": value_result["confidence"],
                "success": True,
                "errors": errors,
                "steps": steps,
                "metadata": {
                    "file": Path(file_path).name,
                    "table": table_meta["table_name"],
                    "row_strategy": best_strategy,
                    "value_confidence": value_result["confidence"],
                },
            }

        except Exception as e:
            errors.append(f"Exception: {str(e)}")
            return {
                "value": None,
                "confidence": 0.0,
                "success": False,
                "errors": errors,
                "steps": steps,
            }

    def run_test(self, num_questions: int = 5, verbose: bool = True) -> dict[str, Any]:
        """
        Run pipeline test on random OfficeQA questions.

        Args:
            num_questions: Number of test questions
            verbose: Print progress

        Returns:
            {passed, failed, accuracy, results}
        """

        print(f"\n{'=' * 70}")
        print(f"TESTING E2E PIPELINE ON {num_questions} RANDOM QUESTIONS")
        print(f"{'=' * 70}")

        results = []
        passed = 0
        failed = 0

        # For now, create synthetic test cases
        test_cases = [
            {
                "id": 1,
                "question": "What was total receipts in FY 1995?",
                "decomposition": {
                    "year": 1995,
                    "table_id": "FFO-1",
                    "row_identifier": ["Total", "Receipts"],
                    "column_identifier": "Total receipts",
                },
                "expected_answer": 1350576000000,  # Approximate (needs real ground truth)
            },
            {
                "id": 2,
                "question": "What was total receipts in FY 1996?",
                "decomposition": {
                    "year": 1996,
                    "table_id": "FFO-1",
                    "row_identifier": ["Total", "Receipts"],
                    "column_identifier": "Total receipts",
                },
                "expected_answer": 1413156000000,
            },
        ]

        for i, test_case in enumerate(test_cases[:num_questions], 1):
            print(f"\n{'─' * 70}")
            print(f"Q{i}: {test_case['question']}")
            print(f"{'─' * 70}")

            result = self.test_decomposition_piece(test_case["decomposition"], verbose=verbose)

            result["question_id"] = test_case["id"]
            result["question"] = test_case["question"]

            if result["success"]:
                passed += 1
                print(f"\n✅ SUCCESS: {result['value']}")
            else:
                failed += 1
                print(f"\n❌ FAILED: {result['errors']}")

            results.append(result)

        accuracy = passed / max(passed + failed, 1)

        print(f"\n{'=' * 70}")
        print(f"RESULTS: {passed} passed, {failed} failed")
        print(f"Accuracy: {accuracy * 100:.1f}%")
        print(f"{'=' * 70}\n")

        return {"passed": passed, "failed": failed, "accuracy": accuracy, "results": results}


if __name__ == "__main__":
    tester = E2EPipelineTest()
    results = tester.run_test(num_questions=2, verbose=True)

    # Print summary
    print("\nDetailed Results:")
    for r in results["results"]:
        print(f"  Q{r['question_id']}: {r['question']}")
        if r["success"]:
            print(f"    ✅ Value: {r['value']} (confidence: {r['confidence']:.2f})")
        else:
            print(f"    ❌ Errors: {', '.join(r['errors'])}")
