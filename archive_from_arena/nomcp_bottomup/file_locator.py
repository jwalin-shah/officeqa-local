"""File Locator: Maps year/month to Treasury Bulletin file paths."""

import glob
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent.parent / "corpus"


class FileLocator:
    """Locates Treasury Bulletin files by year and month."""

    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self._build_index()

    def _build_index(self):
        """Build index of available files: {year: [months]}"""
        self.files_by_year = {}
        pattern = str(self.corpus_dir / "treasury_bulletin_*.txt")

        for filepath in glob.glob(pattern):
            # Parse: treasury_bulletin_YYYY_MM.txt
            filename = Path(filepath).stem
            parts = filename.split("_")
            if len(parts) >= 4:
                try:
                    year = int(parts[2])
                    month = int(parts[3]) if len(parts) >= 4 else None

                    if year not in self.files_by_year:
                        self.files_by_year[year] = {}
                    if month:
                        self.files_by_year[year][month] = filepath
                except ValueError:
                    continue

        # Sort years
        self.available_years = sorted(self.files_by_year.keys())

    def locate_file(
        self,
        year: int | None = None,
        month: int | None = None,
        prefer_months: list[int] | None = None,
        return_all: bool = False,
    ) -> tuple[str | None, dict]:
        """
        Locate file(s) for given year/month.

        Args:
            year: Fiscal/Calendar year to search (None = all years)
            month: Specific month (None = return all available)
            prefer_months: Month priority order (None = return all months available)
            return_all: If True, return all matching files instead of best

        Returns:
            (file_path, metadata)
            - file_path: Single best file, None if not found or multiple available
            - metadata: {
                'year': year,
                'month': month,
                'status': 'found' | 'ambiguous' | 'not_found',
                'candidates': [list of matching files],
                'note': 'explanation'
              }
        """

        if prefer_months is None:
            # Default: no preference, return all available
            prefer_months = None

        # Case 1: Return ALL files (search across all years)
        if year is None:
            all_files = []
            for y in self.available_years:
                files_by_month = self.files_by_year[y]
                # If prefer_months specified, pick based on preference
                if prefer_months:
                    for m in prefer_months:
                        if m in files_by_month:
                            all_files.append(files_by_month[m])
                            break
                else:
                    # No preference: return all files for each year
                    all_files.extend(files_by_month.values())

            return (
                None,
                {
                    "year": None,
                    "month": None,
                    "status": "ambiguous",
                    "candidates": sorted(all_files),
                    "note": f"No year specified. Returning {len(all_files)} files",
                },
            )

        # Case 2: Year specified, month NOT specified
        if year not in self.files_by_year:
            return (
                None,
                {
                    "year": year,
                    "month": None,
                    "status": "not_found",
                    "candidates": [],
                    "note": f"Year {year} not in corpus",
                },
            )

        files_by_month = self.files_by_year[year]

        if month is None:
            # If prefer_months specified, auto-select best month
            if prefer_months:
                for m in prefer_months:
                    if m in files_by_month:
                        return (
                            files_by_month[m],
                            {
                                "year": year,
                                "month": m,
                                "status": "found",
                                "candidates": [files_by_month[m]],
                                "note": f"Auto-selected month {m} for year {year}",
                            },
                        )

            # No preferred month found or no preference specified
            # Return all available months as candidates
            candidates = list(files_by_month.values())
            return (
                None,
                {
                    "year": year,
                    "month": None,
                    "status": "ambiguous",
                    "candidates": candidates,
                    "note": f"Multiple months available for year {year}: {sorted(files_by_month.keys())}",
                },
            )

        # Case 3: Both year and month specified
        if month in files_by_month:
            return (
                files_by_month[month],
                {
                    "year": year,
                    "month": month,
                    "status": "found",
                    "candidates": [files_by_month[month]],
                    "note": f"Exact match: {year}-{month:02d}",
                },
            )

        # Month not found for this year
        available_months = sorted(files_by_month.keys())
        return (
            None,
            {
                "year": year,
                "month": month,
                "status": "not_found",
                "candidates": list(files_by_month.values()),
                "note": f"Month {month} not available for year {year}. Available: {available_months}",
            },
        )

    def locate_files(self, years: list[int]) -> tuple[list[str], dict]:
        """
        Locate files for multiple years.

        Args:
            years: List of years to search

        Returns:
            (file_paths, metadata)
        """
        files = []
        metadata = {}

        for year in years:
            filepath, meta = self.locate_file(year=year)
            if filepath:
                files.append(filepath)
            metadata[year] = meta

        return (
            files,
            {
                "years": years,
                "found": len(files),
                "not_found": len(years) - len(files),
                "per_year": metadata,
            },
        )

    def get_available_years(self) -> list[int]:
        """Return list of available years in corpus."""
        return self.available_years

    def get_available_months(self, year: int) -> list[int]:
        """Return list of available months for given year."""
        if year in self.files_by_year:
            return sorted(self.files_by_year[year].keys())
        return []


# Example usage
if __name__ == "__main__":
    locator = FileLocator()

    print("Available years:", locator.get_available_years())
    print()

    # Test: Single year
    file1, meta1 = locator.locate_file(year=1995)
    print(f"1995 → {file1}")
    print(f"  Metadata: {meta1}\n")

    # Test: Specific year + month
    file2, meta2 = locator.locate_file(year=1995, month=12)
    print(f"1995-12 → {file2}")
    print(f"  Metadata: {meta2}\n")

    # Test: Unknown year
    file3, meta3 = locator.locate_file(year=None)
    print(f"All years → {len(meta3['candidates'])} files")
    print(f"  Metadata: {meta3}\n")

    # Test: Year not in corpus
    file4, meta4 = locator.locate_file(year=2050)
    print(f"2050 → {file4}")
    print(f"  Metadata: {meta4}\n")

    # Test: Multiple years
    files5, meta5 = locator.locate_files([1995, 1996, 2050])
    print(f"[1995, 1996, 2050] → {len(files5)} files found")
    print(f"  Metadata: {meta5}")
