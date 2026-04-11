"""Table Finder: Locates and extracts tables from Treasury Bulletin files."""

import re


class TableFinder:
    """Finds and extracts tables from Treasury Bulletin text files."""

    def find_table(self, filepath: str, table_id: str) -> tuple[str | None, dict]:
        """
        Find and extract a table from a Treasury Bulletin file.

        Args:
            filepath: Path to Treasury Bulletin text file
            table_id: Table identifier (e.g., "FFO-1", "receipt_summary", or pattern)

        Returns:
            (table_content, metadata)
            - table_content: Full text of the table, or None if not found
            - metadata: {
                'table_id': str,
                'status': 'found' | 'not_found' | 'ambiguous',
                'start_line': int,
                'end_line': int,
                'table_name': str,
                'note': str
              }
        """

        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return (
                None,
                {
                    "table_id": table_id,
                    "status": "not_found",
                    "start_line": None,
                    "end_line": None,
                    "table_name": None,
                    "note": f"File not found: {filepath}",
                },
            )

        # Search for table header
        start_line = self._find_table_start(lines, table_id)
        if start_line is None:
            return (
                None,
                {
                    "table_id": table_id,
                    "status": "not_found",
                    "start_line": None,
                    "end_line": None,
                    "table_name": None,
                    "note": f'Table "{table_id}" not found in file',
                },
            )

        # Find table end (next TABLE or end of file)
        end_line = self._find_table_end(lines, start_line)
        table_name = self._extract_table_name(lines[start_line])

        # Extract table content
        table_content = "".join(lines[start_line:end_line])

        return (
            table_content,
            {
                "table_id": table_id,
                "status": "found",
                "start_line": start_line,
                "end_line": end_line,
                "table_name": table_name,
                "note": f"Found table spanning lines {start_line}-{end_line}",
            },
        )

    def _find_table_start(self, lines: list[str], table_id: str) -> int | None:
        """
        Find the line where a table starts (actual table, not TOC entry).

        Looks for patterns like:
        - "TABLE FFO-1.--Summary of fiscal operations" (with actual table after)
        """

        table_id_upper = table_id.upper()

        # First pass: find candidate lines with TABLE + ID
        candidates = []
        for i, line in enumerate(lines):
            line_upper = line.upper()

            # Match "TABLE ID" pattern
            if re.search(rf"TABLE\s+{re.escape(table_id_upper)}", line_upper):
                candidates.append(i)

        if not candidates:
            return None

        # Second pass: prefer the candidate that's followed by actual table data
        # (has pipes "|" for markdown table in next 10 lines)
        for start_idx in candidates:
            # Check if this is followed by table content (pipe-delimited columns)
            for check_idx in range(start_idx + 1, min(start_idx + 15, len(lines))):
                if "|" in lines[check_idx]:
                    # This looks like the real table, not TOC
                    return start_idx

        # Fallback: return last candidate (usually the real table)
        return candidates[-1]

    def _find_table_end(self, lines: list[str], start_line: int) -> int:
        """
        Find the line where a table ends.

        A table ends when:
        - Next "TABLE" header is found
        - Empty line followed by non-table content
        - End of file
        """

        for i in range(start_line + 1, len(lines)):
            line = lines[i].strip()

            # Check for next TABLE header
            if re.match(r"TABLE\s+", line.upper()):
                return i

            # Check for analysis section or other major section
            if re.match(r"(ANALYSIS|PROFILE|FEDERAL|NOTE|SOURCE)", line.upper()):
                return i

        return len(lines)

    def _extract_table_name(self, header_line: str) -> str:
        """Extract table name from header line."""
        # Remove leading/trailing whitespace
        name = header_line.strip()
        # Remove "TABLE" prefix
        name = re.sub(r"^TABLE\s*", "", name, flags=re.IGNORECASE)
        return name

    def extract_rows(self, table_content: str) -> list[dict[str, str]]:
        """
        Extract rows from table content.

        Parses markdown table format and returns list of row dicts.

        Returns:
            List of dicts where keys are column headers
        """

        lines = table_content.split("\n")
        rows = []

        # Find header row (contains "|")
        header_line = None
        data_start_idx = 0

        for i, line in enumerate(lines):
            if "|" in line and header_line is None:
                header_line = line
                data_start_idx = i + 1
                # Skip separator line
                if i + 1 < len(lines) and "---" in lines[i + 1]:
                    data_start_idx = i + 2
                break

        if not header_line:
            return []

        # Parse headers
        headers = [h.strip() for h in header_line.split("|")][1:-1]  # Remove first/last empty

        # Parse data rows
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if not line or "|" not in line:
                continue

            cells = [c.strip() for c in line.split("|")][1:-1]  # Remove first/last empty

            # Skip separator lines
            if all(c.replace("-", "").strip() == "" for c in cells):
                continue

            # Create row dict
            if len(cells) == len(headers):
                row = dict(zip(headers, cells))
                rows.append(row)

        return rows


# Example usage
if __name__ == "__main__":
    finder = TableFinder()

    # Test 1: Find FFO-1 table
    filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"
    content1, meta1 = finder.find_table(filepath, "FFO-1")

    print(f"Table: {meta1['table_name']}")
    print(f"Status: {meta1['status']}")
    print(f"Lines: {meta1['start_line']}-{meta1['end_line']}")
    print(f"Note: {meta1['note']}")
    print()

    if content1:
        # Extract rows
        rows = finder.extract_rows(content1)
        print(f"Found {len(rows)} data rows")
        if rows:
            print("First row:", rows[0])
            print("Second row:", rows[1])
