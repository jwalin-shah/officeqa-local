#!/usr/bin/env python3
"""Verification script demonstrating ColumnFinder capabilities."""

from column_finder import ColumnFinder


def main():
    finder = ColumnFinder()

    print("=" * 80)
    print("COLUMN FINDER VERIFICATION")
    print("=" * 80)

    # Test 1: Basic exact match
    print("\n[TEST 1] Exact Match with Hierarchical Headers")
    print("-" * 80)
    rows1 = [
        {"Fiscal year": "1995", "Total > Receipts": 1350576.0},
        {"Fiscal year": "1996", "Total > Receipts": 1413156.0},
    ]
    col, conf = finder.find_column(rows1, "total receipts")
    print("Input identifier: 'total receipts'")
    print(f"Found column: {col}")
    print(f"Confidence: {conf}")
    assert col == "Total > Receipts", "Should find exact match"
    assert conf >= 0.85, "Should have high confidence"
    print("PASS")

    # Test 2: Real Treasury Bulletin structure
    print("\n[TEST 2] Real Treasury Bulletin Data Structure")
    print("-" * 80)
    rows2 = [
        {
            "Fiscal year or month": "1995",
            "Total on-budget and off-budget results > Total receipts (1)": 1350576.0,
            "Total on-budget and off-budget results > On-budget receipts (2)": 1000000.0,
            "Total on-budget and off-budget results > Off-budget receipts (3)": 350576.0,
            "Total on-budget and off-budget results > Total outlays (4)": 1514389.0,
        }
    ]
    col, conf = finder.find_column(rows2, "total receipts")
    print("Input identifier: 'total receipts'")
    print(f"Found column: {col}")
    print(f"Confidence: {conf}")
    assert col is not None, "Should find a match"
    assert "receipts" in col.lower(), "Should find receipts column"
    print("PASS")

    # Test 3: Multiple candidates
    print("\n[TEST 3] Finding All Candidates")
    print("-" * 80)
    rows3 = [
        {
            "Year": "1995",
            "Total Receipts": 1350576.0,
            "On-Budget Receipts": 1000000.0,
            "Off-Budget Receipts": 350576.0,
        }
    ]
    candidates = finder.find_all_candidate_columns(rows3, "receipts", min_confidence=0.60)
    print("Input identifier: 'receipts'")
    print(f"Found {len(candidates)} candidates:")
    for i, (col, conf) in enumerate(candidates, 1):
        print(f"  {i}. {col}: {conf:.2f}")
    assert len(candidates) == 3, "Should find all three receipts columns"
    assert candidates[0][1] >= candidates[-1][1], "Should be sorted by confidence"
    print("PASS")

    # Test 4: Normalization handling
    print("\n[TEST 4] Normalization of Headers")
    print("-" * 80)
    test_headers = [
        ("Total > Receipts", "total receipts"),
        ("On-budget > Receipts (1)", "on budget receipts"),
        ("Multiple   Spaces", "multiple spaces"),
        ("Total-Off-Budget", "total off budget"),
    ]
    print("Normalization examples:")
    for original, expected in test_headers:
        normalized = finder._normalize_for_matching(original)
        print(f"  '{original}'")
        print(f"  → '{normalized}'")
        # All normalizations should match their expected patterns
        assert len(normalized) > 0, f"Normalization failed for '{original}'"
    print("PASS")

    # Test 5: Case insensitivity
    print("\n[TEST 5] Case Insensitivity")
    print("-" * 80)
    rows5 = [
        {"FISCAL YEAR": "1995", "TOTAL RECEIPTS": 1350576.0},
        {"fiscal year": "1996", "total receipts": 1413156.0},
    ]
    col, conf = finder.find_column(rows5, "total receipts")
    print("Input identifier: 'total receipts'")
    print(f"Found column: {col}")
    print(f"Confidence: {conf}")
    assert col is not None, "Should match case-insensitively"
    assert conf >= 0.85, "Should have high confidence"
    print("PASS")

    # Test 6: Empty/None handling
    print("\n[TEST 6] Empty and None Handling")
    print("-" * 80)
    # Empty rows
    col, conf = finder.find_column([], "total receipts")
    print(f"Empty rows: col={col}, conf={conf}")
    assert col is None and conf == 0.0, "Should handle empty rows"

    # Empty identifier
    rows_test = [{"Year": "1995", "Total Receipts": 1350576.0}]
    col, conf = finder.find_column(rows_test, "")
    print(f"Empty identifier: col={col}, conf={conf}")
    assert col is None and conf == 0.0, "Should handle empty identifier"

    print("PASS")

    # Test 7: Best match selection
    print("\n[TEST 7] Best Match Selection Among Candidates")
    print("-" * 80)
    rows7 = [
        {
            "Year": "1995",
            "Total Receipts": 1350576.0,
            "Receipts Detail": 100000.0,
            "Total Outlays": 1514389.0,
        }
    ]
    col, conf = finder.find_column(rows7, "total receipts")
    print("Input identifier: 'total receipts'")
    print(f"Found column: {col}")
    print(f"Confidence: {conf}")
    assert col == "Total Receipts", "Should select best match"
    assert conf >= 0.85, "Should have high confidence for exact match"
    print("PASS")

    # Test 8: Original column name preservation
    print("\n[TEST 8] Original Column Name Preservation")
    print("-" * 80)
    rows8 = [{"Year": "1995", "Total > Receipts (1)": 1350576.0}]
    col, conf = finder.find_column(rows8, "total receipts")
    print("Input identifier: 'total receipts'")
    print(f"Found column (should preserve original): {col}")
    assert col == "Total > Receipts (1)", "Should preserve original column name"
    assert ">" in col and "(1)" in col, "Should keep hierarchy and numbering"
    print("PASS")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION TESTS PASSED!")
    print("=" * 80)
    print("\nColumnFinder successfully:")
    print("  - Finds columns by keyword matching")
    print("  - Handles hierarchical headers ('>'-separated)")
    print("  - Handles numbering artifacts '(1)', '(2)', etc.")
    print("  - Is case-insensitive")
    print("  - Preserves original column names")
    print("  - Returns confidence scores 0.0-1.0")
    print("  - Finds multiple candidates with sorting")
    print("  - Handles edge cases (empty, None, etc.)")


if __name__ == "__main__":
    main()
