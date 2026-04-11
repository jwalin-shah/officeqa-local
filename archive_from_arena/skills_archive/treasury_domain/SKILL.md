# Treasury Domain Quick Reference

## Fiscal Year
- Pre-1977: Jul 1 to Jun 30. FY1940 = Jul 1939 - Jun 1940.
- Post-1976: Oct 1 to Sep 30. FY1977 = Oct 1976 - Sep 1977.
- Annual totals for year Y appear in Y+1 bulletins (January-March).
- The Master Ledger has pre-computed FY totals (period_basis="fiscal") using the correct definition for each era.

## Calendar Year vs Fiscal Year
- CY1940 = Jan 1940 - Dec 1940 (sum of 12 calendar months).
- FY1940 = Jul 1939 - Jun 1940 (pre-1977 definition).
- These are DIFFERENT values. CY1940 national defense = 2,602M vs FY1940 = 1,559M.
- Use period_basis="calendar" or "fiscal" in search_ledger, or filter by year in search_canonical.
- The Ledger only creates CY/FY totals when ALL 12 months are present.

## Defense (1940s)
- Split into "War Department" + "Navy Department" with no combined row.
- Sum them for total defense. Post-1950: use "National defense" row directly.

## Units
- Always verify units via get_table_profile(table_pk=...) on the source table from Ledger results.
- "In thousands" / "In millions" / "In billions" — convert before combining.
- value_scaled in extract_values results is already converted — do not multiply again.

## CPI Adjustment
- Formula: real_value = nominal × (target_CPI / source_CPI)
- Call get_cpi_index(year, month) for both periods. Base: 1982-84 = 100.

## Currency Conversion
- Call get_exchange_rate(pair, year, month). Available: USD/JPY, USD/GBP, USD/INR, USD/DEM, USD/CAD.

## Search Tips
- Start with search_canonical (935K facts). Fall back to search_ledger (13,629 metrics, 1915-2016) if canonical store is unavailable.
- Try synonyms: "receipts" vs "revenue", "expenditures" vs "outlays".
- If ledger returns multiple results, use the one from the latest bulletin (most revised).
- For exact table context, use the table_pk from ledger results with get_table_profile.
