"""Row Matcher: Finds rows in tables by identifier patterns."""


class RowMatcher:
    """Matches rows in tables using different strategies."""

    def find_row_strict(self, rows: list[dict[str, str]], row_identifier: list[str]) -> dict | None:
        """
        Strict matching: all identifier parts must match exactly.

        Args:
            rows: List of row dicts from table
            row_identifier: List of strings to match (e.g., ["Total", "Receipts"])

        Returns:
            Matching row or None
        """
        identifier_text = " ".join(row_identifier).upper()

        for row in rows:
            # Check first column (row label) - usually the key
            first_col = next(iter(row.values()))
            if first_col.upper() == identifier_text:
                return row

        return None

    def find_row_fuzzy(
        self, rows: list[dict[str, str]], row_identifier: list[str]
    ) -> tuple[dict | None, float]:
        """
        Fuzzy matching: contains all identifier keywords.

        Args:
            rows: List of row dicts
            row_identifier: List of keywords to find

        Returns:
            (matching_row, confidence_score) - row is None if no match
        """
        keywords = [k.upper() for k in row_identifier]

        best_match = None
        best_score = 0.0

        for row in rows:
            first_col = next(iter(row.values())).upper()

            # Count how many keywords appear in row label
            matches = sum(1 for kw in keywords if kw in first_col)
            score = matches / len(keywords) if keywords else 0.0

            if score > best_score:
                best_score = score
                best_match = row

        return (best_match, best_score)

    def find_row_contextual(
        self, rows: list[dict[str, str]], row_identifier: list[str], row_index: int | None = None
    ) -> tuple[dict | None, float]:
        """
        Contextual matching: uses table structure and position.

        Looks at:
        - Row label similarity
        - Position in table (totals often at end/beginning)
        - Similar rows nearby

        Args:
            rows: List of row dicts
            row_identifier: Keywords to match
            row_index: Expected position (if known)

        Returns:
            (matching_row, confidence_score)
        """
        keywords = [k.upper() for k in row_identifier]

        matches = []
        for i, row in enumerate(rows):
            first_col = next(iter(row.values())).upper()

            # Score based on keyword matches
            keyword_matches = sum(1 for kw in keywords if kw in first_col)
            keyword_score = keyword_matches / len(keywords) if keywords else 0.0

            # Score based on position (totals often at end or beginning)
            if "TOTAL" in first_col or "SUMMARY" in first_col:
                position_score = 0.5  # Bonus for total-like rows
            else:
                position_score = 0.0

            total_score = keyword_score * 0.7 + position_score * 0.3

            if total_score > 0:
                matches.append((row, total_score, i))

        if not matches:
            return (None, 0.0)

        # Return best match
        best_row, best_score, best_idx = max(matches, key=lambda x: x[1])
        return (best_row, best_score)

    def find_all_candidate_rows(
        self, rows: list[dict[str, str]], row_identifier: list[str]
    ) -> list[tuple[dict, str, float]]:
        """
        Find ALL rows that could match the identifier.

        Returns:
            List of (row, strategy, confidence) sorted by confidence
        """
        candidates = []

        # Try strict match
        strict = self.find_row_strict(rows, row_identifier)
        if strict:
            candidates.append((strict, "strict", 0.99))

        # Try fuzzy match
        fuzzy_row, fuzzy_score = self.find_row_fuzzy(rows, row_identifier)
        if fuzzy_row is not None and fuzzy_row not in [c[0] for c in candidates]:
            candidates.append((fuzzy_row, "fuzzy", fuzzy_score))

        # Try contextual match
        contextual_row, contextual_score = self.find_row_contextual(rows, row_identifier)
        if contextual_row is not None and contextual_row not in [c[0] for c in candidates]:
            candidates.append((contextual_row, "contextual", contextual_score))

        # Sort by confidence descending
        return sorted(candidates, key=lambda x: x[2], reverse=True)


# Example usage
if __name__ == "__main__":
    # Sample rows from a table
    rows = [
        {"Fiscal year or month": "1995", "Total receipts": "1350576"},
        {"Fiscal year or month": "1996", "Total receipts": "1413156"},
        {"Fiscal year or month": "Total", "Total receipts": "2763732"},
    ]

    matcher = RowMatcher()

    # Test 1: Strict match
    result = matcher.find_row_strict(rows, ["Total", "Receipts"])
    print(f"Strict match for ['Total', 'Receipts']: {result}")

    # Test 2: Fuzzy match
    fuzzy_row, fuzzy_score = matcher.find_row_fuzzy(rows, ["Total"])
    print(f"Fuzzy match for ['Total']: {fuzzy_row} (score: {fuzzy_score})")

    # Test 3: Find all candidates
    candidates = matcher.find_all_candidate_rows(rows, ["Total"])
    print("\nAll candidates for ['Total']:")
    for row, strategy, score in candidates:
        print(f"  {strategy}: {row} (confidence: {score})")
