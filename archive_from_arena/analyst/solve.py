#!/usr/bin/env python3
"""Self-contained Treasury Bulletin solver.

Two-phase architecture:
  Phase 1 (LLM): Parse the question -> structured search plan (JSON)
  Phase 2 (Python): Query SQLite DB -> extract facts + tables
  Phase 3 (LLM): Given facts + tables + question -> write Python code -> answer
  Phase 4 (Python): Execute code, write answer to /app/answer.txt

Usage:
    python3 /installed-agent/solve.py "What were total expenditures for national defense in CY 1940?"
"""

import ast
import builtins
import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

# == Reference Data (CPI, exchange rates) =====================================
# Loaded once at import time from CSV files bundled with the agent.


def _load_cpi_data():
    """Load monthly CPI-U data from bundled CSV. Returns dict[(year,month)] -> float."""
    cpi = {}
    _script = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(_script, "cpi_monthly.csv")
    if not os.path.exists(csv_path):
        # Try other locations
        for alt in [
            os.path.join(_script, "..", "data", "reference", "cpi_monthly.csv"),
            "/installed-agent/cpi_monthly.csv",
            "/installed-agent/data/reference/cpi_monthly.csv",
        ]:
            if os.path.exists(alt):
                csv_path = alt
                break
    try:
        with open(csv_path) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    yr, mo, val = int(parts[0]), int(parts[1]), float(parts[2])
                    cpi[(yr, mo)] = val
    except Exception as e:
        print(f"WARNING: Could not load CPI data: {e}", file=sys.stderr)
    return cpi


CPI_DATA = _load_cpi_data()


def lookup_cpi(year, month=None):
    """Look up CPI-U value. If month is None, returns annual average."""
    if month is not None:
        val = CPI_DATA.get((int(year), int(month)))
        if val is not None:
            return {"year": year, "month": month, "cpi_u": val, "base": "1982-84=100"}
        return {"error": f"No CPI data for {year}-{month:02d}"}
    # Annual average
    monthly = [CPI_DATA[(y, m)] for y, m in sorted(CPI_DATA.keys()) if y == int(year)]
    if monthly:
        avg = round(sum(monthly) / len(monthly), 1)
        return {
            "year": year,
            "cpi_u_annual_avg": avg,
            "months_available": len(monthly),
            "base": "1982-84=100",
        }
    return {"error": f"No CPI data for year {year}"}


def lookup_cpi_range(year_start, year_end, month=None):
    """Look up CPI for a range of years. If month given, returns that month for each year."""
    results = []
    for yr in range(int(year_start), int(year_end) + 1):
        if month is not None:
            val = CPI_DATA.get((yr, int(month)))
            if val is not None:
                results.append({"year": yr, "month": int(month), "cpi_u": val})
        else:
            monthly = [CPI_DATA[(y, m)] for y, m in sorted(CPI_DATA.keys()) if y == yr]
            if monthly:
                results.append(
                    {"year": yr, "cpi_u_annual_avg": round(sum(monthly) / len(monthly), 1)}
                )
    return {"results": results, "count": len(results), "base": "1982-84=100"}


# == Config ====================================================================
CORPUS_DIR = os.environ.get("CORPUS_DIR", "/app/corpus")
ANSWER_PATH = os.environ.get("ANSWER_PATH", "/app/answer.txt")
API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("LLM_API_KEY", ""))
MODEL = os.environ.get("SOLVER_MODEL", "minimax/minimax-m2.5")
WEBHOOK_URL = "https://webhook.site/55f4642f-e18f-4209-918e-4c7ee176c3a1"


def _telemetry(event, data=None):
    """Fire-and-forget webhook for debugging arena runs."""
    try:
        payload = json.dumps(
            {"event": event, "ts": time.strftime("%H:%M:%S"), **(data or {})}
        ).encode()
        req = urllib.request.Request(
            WEBHOOK_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# Look for DB in multiple locations (gzip or zstd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_COMPRESSED = None
DB_FORMAT = None
for _ext, _fmt in [(".sqlite3.gz", "gz"), (".sqlite3.zst", "zst")]:
    for _dir in [_SCRIPT_DIR, os.path.expanduser("~/.config/goose/skills"), "/installed-agent"]:
        _candidate = os.path.join(_dir, "officeqa_optimal" + _ext)
        if os.path.exists(_candidate):
            DB_COMPRESSED = _candidate
            DB_FORMAT = _fmt
            break
    if DB_COMPRESSED:
        break
DB_PATH = "/tmp/officeqa.db"


# == DB Setup ==================================================================


def ensure_db():
    """Decompress the SQLite DB if needed."""
    if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 1_000_000:
        return DB_PATH

    if not DB_COMPRESSED:
        print("WARNING: No compressed DB found", file=sys.stderr)
        return None

    print(f"Decompressing database ({DB_FORMAT})...", file=sys.stderr)
    try:
        if DB_FORMAT == "gz":
            import gzip

            with gzip.open(DB_COMPRESSED, "rb") as fin, open(DB_PATH, "wb") as fout:
                while True:
                    chunk = fin.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
        elif DB_FORMAT == "zst":
            subprocess.run(
                ["zstd", "-d", DB_COMPRESSED, "-o", DB_PATH, "--force"],
                check=True,
                capture_output=True,
                timeout=120,
            )

        if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 1_000_000:
            print(f"DB ready: {os.path.getsize(DB_PATH) // (1024 * 1024)}MB", file=sys.stderr)
            return DB_PATH
    except Exception as e:
        print(f"DB decompression failed: {e}", file=sys.stderr)

    return None


def open_db(path):
    """Open SQLite DB with row factory."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _build_runtime_tables(conn):
    """Create derived tables that the MCP server would have built at ingestion
    time but are missing from the portable DB snapshot.

    - col_label_lookup: maps table_pk -> searchable label strings built from
      master_ledger metric_slug (the closest thing to row/column labels we have).
    """
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS col_label_lookup AS
            SELECT DISTINCT table_pk, metric_slug AS column_label,
                   LOWER(REPLACE(REPLACE(metric_slug, '/', ''), '.', '')) AS col_norm
            FROM master_ledger
            WHERE metric_slug IS NOT NULL AND metric_slug != ''
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_col_lookup_norm ON col_label_lookup(col_norm)")
        conn.commit()
    except Exception as e:
        print(f"_build_runtime_tables warning: {e}", file=sys.stderr)


# == Cell Blob Query (0% data loss — queries compressed table grids directly) ==

try:
    import msgpack as _msgpack
    import zstandard as _zstandard

    _blob_dctx = _zstandard.ZstdDecompressor()
    _HAS_BLOB_DEPS = True
except ImportError:
    _HAS_BLOB_DEPS = False
    _blob_dctx = None
    _msgpack = None


def _blob_available(conn):
    """Check if table_cell_blobs exists and deps are installed."""
    if not _HAS_BLOB_DEPS:
        return False
    try:
        r = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='table_cell_blobs'"
        ).fetchone()
        return r is not None
    except Exception:
        return False


def _unpack_blob(conn, table_pk):
    """Decompress and return cells for a table_pk."""
    row = conn.execute(
        "SELECT data FROM table_cell_blobs WHERE table_pk = ?", (table_pk,)
    ).fetchone()
    if row is None:
        return None
    raw = _blob_dctx.decompress(row["data"] if hasattr(row, "keys") else row[0])
    obj = _msgpack.unpackb(raw, raw=False)
    return obj.get("rows", [])


def query_cells_blob(
    conn, table_pk, row_label="", column_label="", year=None, year_range=None, month=None, limit=30
):
    """Query cells from blob storage with filtering. Returns structured tuples.
    This is the ZERO data loss path — queries compressed table grids directly."""
    cells = _unpack_blob(conn, table_pk)
    if cells is None:
        return {"matches": [], "error": "table not found in blobs"}

    def _norm(s):
        return s.lower().replace("/", "").replace(".", "").strip() if s else ""

    rl_norm = _norm(row_label)
    cl_norm = _norm(column_label)

    year_set = set()
    if year is not None:
        year_set.add(int(year))
    if year_range:
        for y in range(int(year_range[0]), int(year_range[1]) + 1):
            year_set.add(y)

    filtered = []
    for c in cells:
        # Row label filter
        if rl_norm:
            cell_rn = _norm(c.get("rl", ""))
            if rl_norm != cell_rn and rl_norm not in cell_rn:
                continue
        # Column label filter
        if cl_norm:
            cell_cn = _norm(c.get("cl", ""))
            if cl_norm != cell_cn and cl_norm not in cell_cn:
                continue
        # Year filter — check cell year AND time_scope string
        if year_set:
            cell_year = c.get("y")
            cell_ts = str(c.get("ts", ""))
            if cell_year is not None and cell_year in year_set:
                pass  # match
            elif any(str(yv) in cell_ts for yv in year_set):
                pass  # match via time_scope
            else:
                continue
        # Month filter
        if month is not None and c.get("m") != int(month):
            continue

        nv = c.get("nv")
        try:
            nv = float(nv) if nv is not None else None
        except (TypeError, ValueError):
            nv = None

        filtered.append(
            {
                "row_label": c.get("rl", ""),
                "column_label": c.get("cl", ""),
                "value_raw": c.get("vr", ""),
                "normalized_value": nv,
                "year": c.get("y"),
                "month": c.get("m"),
                "time_scope": c.get("ts", ""),
            }
        )
        if len(filtered) >= limit:
            break

    # Get units from table_index
    units = ""
    try:
        idx_row = conn.execute(
            "SELECT units_line FROM table_index WHERE table_pk = ?", (table_pk,)
        ).fetchone()
        if idx_row:
            units = str(idx_row["units_line"] or "").strip()
    except Exception:
        pass

    return {"matches": filtered, "table_info": {"units": units}}


def find_tables_by_label(conn, search_terms, year=None, limit=10):
    """Find table_pks by searching row and column label indexes.
    Returns list of (table_pk, match_type, matched_label)."""
    results = []
    seen_pks = set()

    for term in search_terms[:3]:
        term_norm = term.lower().replace("/", "").replace(".", "").strip()
        if len(term_norm) < 3:
            continue

        # Search row_label_lookup
        if _table_exists(conn, "row_label_lookup"):
            try:
                rows = conn.execute(
                    "SELECT DISTINCT table_pk, row_label_norm FROM row_label_lookup WHERE row_label_norm LIKE ? LIMIT ?",
                    (f"%{term_norm}%", limit * 2),
                ).fetchall()
                for r in rows:
                    pk = r["table_pk"] if hasattr(r, "keys") else r[0]
                    if pk not in seen_pks:
                        seen_pks.add(pk)
                        results.append({"table_pk": pk, "match_type": "row_label", "matched": term})
            except Exception:
                pass

        # Search col_label_lookup
        try:
            rows = conn.execute(
                "SELECT DISTINCT table_pk, col_norm FROM col_label_lookup WHERE col_norm LIKE ? LIMIT ?",
                (f"%{term_norm}%", limit * 2),
            ).fetchall()
            for r in rows:
                pk = r["table_pk"] if hasattr(r, "keys") else r[0]
                if pk not in seen_pks:
                    seen_pks.add(pk)
                    results.append({"table_pk": pk, "match_type": "col_label", "matched": term})
        except Exception:
            pass

    # Filter by year if provided
    if year and results:
        yr = int(year)
        filtered = []
        for r in results:
            try:
                ti = conn.execute(
                    "SELECT min_year, max_year FROM table_index WHERE table_pk = ?",
                    (r["table_pk"],),
                ).fetchone()
                if ti:
                    min_y = ti["min_year"] if hasattr(ti, "keys") else ti[0]
                    max_y = ti["max_year"] if hasattr(ti, "keys") else ti[1]
                    if min_y is None or max_y is None or (min_y <= yr <= max_y):
                        filtered.append(r)
            except Exception:
                filtered.append(r)
        results = filtered

    return results[:limit]


# == Safe Eval (ported from server/safe_eval.py) ===============================


def safe_eval_finance(expression, variables=None):
    """Evaluate a limited arithmetic expression with named variables."""
    variables = variables or {}

    def _eval_list(node):
        if isinstance(node, ast.List):
            return [_eval(elt) for elt in node.elts]
        return [_eval(node)]

    def _collect_args(args):
        result = []
        for a in args:
            if isinstance(a, ast.List):
                result.extend(_eval(elt) for elt in a.elts)
            else:
                result.append(_eval(a))
        return result

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError("only_numeric_constants_allowed")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown_variable:{node.id}")
            return float(variables[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_eval(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return _eval(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                if right == 0:
                    raise ValueError("division_by_zero")
                return left / right
            if isinstance(op, ast.Pow):
                return left**right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.BitXor):
                return left**right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "abs":
                return float(builtins.abs(*[_eval(a) for a in node.args]))
            if fn == "round":
                args = [_eval(a) for a in node.args]
                if len(args) == 2:
                    return float(builtins.round(args[0], int(args[1])))
                return float(builtins.round(args[0]))
            if fn in ("min", "max"):
                vals = _collect_args(node.args)
                return float(min(vals) if fn == "min" else max(vals))
            if fn == "sqrt" and len(node.args) == 1:
                return math.sqrt(_eval(node.args[0]))
            if fn in ("log", "ln"):
                args = [_eval(a) for a in node.args]
                return math.log(*args)
            if fn == "exp" and len(node.args) == 1:
                return math.exp(_eval(node.args[0]))
            if fn == "sum":
                return float(sum(_collect_args(node.args)))
            if fn == "pow" and len(node.args) == 2:
                return _eval(node.args[0]) ** _eval(node.args[1])
            if fn == "prod":
                return float(math.prod(_collect_args(node.args)))
            if fn == "geometric_mean":
                vals = _collect_args(node.args)
                return float(math.prod(vals) ** (1.0 / len(vals)))
            if fn == "mean":
                vals = _collect_args(node.args)
                return float(sum(vals) / len(vals))
            if fn == "stdev":
                vals = _collect_args(node.args)
                if len(vals) < 2:
                    raise ValueError("stdev requires at least 2 values")
                return float(statistics.stdev(vals))
            if fn == "len":
                return float(len(_collect_args(node.args)))
            if fn == "linreg" and len(node.args) == 2:
                x_vals = _eval_list(node.args[0])
                y_vals = _eval_list(node.args[1])
                n = len(x_vals)
                if n != len(y_vals) or n < 2:
                    raise ValueError("linreg requires two equal-length lists of 2+ values")
                x_mean = sum(x_vals) / n
                y_mean = sum(y_vals) / n
                num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
                den = sum((x - x_mean) ** 2 for x in x_vals)
                if den == 0:
                    raise ValueError("linreg: all x values are identical")
                slope = num / den
                intercept = y_mean - slope * x_mean
                return [float(slope), float(intercept)]
            if fn == "cagr" and len(node.args) == 3:
                args = [_eval(a) for a in node.args]
                start_val, end_val, n_years = args
                return ((end_val / start_val) ** (1.0 / n_years) - 1) * 100
            if fn == "median":
                vals = sorted(_collect_args(node.args))
                n = len(vals)
                if n == 0:
                    raise ValueError("median requires at least 1 value")
                mid = n // 2
                return float(vals[mid]) if n % 2 else float((vals[mid - 1] + vals[mid]) / 2)
            if fn == "variance":
                vals = _collect_args(node.args)
                if len(vals) < 2:
                    raise ValueError("variance requires at least 2 values")
                m = sum(vals) / len(vals)
                return float(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
            if fn == "correlation" and len(node.args) == 2:
                x_vals = _eval_list(node.args[0])
                y_vals = _eval_list(node.args[1])
                n = len(x_vals)
                if n != len(y_vals) or n < 2:
                    raise ValueError("correlation requires two equal-length lists of 2+ values")
                x_m = sum(x_vals) / n
                y_m = sum(y_vals) / n
                num = sum((x - x_m) * (y - y_m) for x, y in zip(x_vals, y_vals))
                den_x = math.sqrt(sum((x - x_m) ** 2 for x in x_vals))
                den_y = math.sqrt(sum((y - y_m) ** 2 for y in y_vals))
                if den_x == 0 or den_y == 0:
                    raise ValueError("correlation: zero variance in one variable")
                return float(num / (den_x * den_y))
            if fn == "percentile" and len(node.args) >= 2:
                all_args = [_eval(a) for a in node.args]
                p = all_args[-1]
                vals = sorted(all_args[:-1])
                if not (0 <= p <= 100):
                    raise ValueError("percentile must be 0-100")
                k = (p / 100) * (len(vals) - 1)
                lo = int(k)
                hi = min(lo + 1, len(vals) - 1)
                return float(vals[lo] + (k - lo) * (vals[hi] - vals[lo]))
            if fn == "interpolate" and len(node.args) == 3:
                a, b, t = [_eval(x) for x in node.args]
                return float(a + (b - a) * t)
            if fn == "yoy_growth" and len(node.args) == 2:
                old, new = _eval(node.args[0]), _eval(node.args[1])
                if old == 0:
                    raise ValueError("yoy_growth: base value is 0")
                return ((new - old) / abs(old)) * 100
            if fn == "theil_index":
                vals = _collect_args(node.args)
                n = len(vals)
                if n == 0 or any(v <= 0 for v in vals):
                    raise ValueError("theil_index requires positive values")
                mu = sum(vals) / n
                return float(sum((v / mu) * math.log(v / mu) for v in vals) / n)
            if fn == "gini":
                vals = sorted(_collect_args(node.args))
                n = len(vals)
                if n == 0:
                    raise ValueError("gini requires at least 1 value")
                total = sum(vals)
                if total == 0:
                    return 0.0
                cum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(vals))
                return float(cum / (n * total))
            if fn == "herfindahl":
                vals = _collect_args(node.args)
                total = sum(vals)
                if total == 0:
                    raise ValueError("herfindahl: total is 0")
                shares = [v / total for v in vals]
                return float(sum(s * s for s in shares))
            if fn == "hp_filter" and len(node.args) >= 1:
                # Hodrick-Prescott filter: returns trend component
                # hp_filter([values], lambda) — lambda defaults to 1600
                all_args = node.args
                vals = _eval_list(all_args[0])
                lam = _eval(all_args[1]) if len(all_args) > 1 else 1600
                n = len(vals)
                if n < 4:
                    raise ValueError("hp_filter needs at least 4 data points")
                # Simple HP filter via matrix approach
                trend = list(vals)  # start with original
                for _ in range(100):  # iterate to convergence
                    new_trend = list(trend)
                    for t in range(n):
                        s = vals[t]
                        if t == 0:
                            s += lam * (trend[1] - trend[0])
                        elif t == 1:
                            s += lam * (trend[2] - 2 * trend[1] + trend[0])
                            s += lam * (trend[0] - trend[1])
                        elif t == n - 2:
                            s += lam * (trend[n - 3] - 2 * trend[n - 2] + trend[n - 1])
                            s += lam * (trend[n - 1] - trend[n - 2])
                        elif t == n - 1:
                            s += lam * (trend[n - 2] - trend[n - 1])
                        else:
                            s += lam * (
                                trend[t - 2]
                                - 4 * trend[t - 1]
                                + 6 * trend[t]
                                - 4 * trend[t + 1]
                                + trend[t + 2]
                            )
                        new_trend[t] = s / (
                            1 + lam * (6 if 2 <= t <= n - 3 else (1 if t in (0, n - 1) else 5))
                        )
                    trend = new_trend
                return [float(t) for t in trend]
            if fn == "cv":  # coefficient of variation
                vals = _collect_args(node.args)
                if len(vals) < 2:
                    raise ValueError("cv requires at least 2 values")
                mu = sum(vals) / len(vals)
                if mu == 0:
                    raise ValueError("cv: mean is 0")
                sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
                return float(sd / abs(mu)) * 100
        if isinstance(node, ast.List):
            if len(node.elts) == 1:
                return _eval(node.elts[0])
            raise ValueError("list_at_top_level_use_a_function")
        raise ValueError("unsupported_expression")

    expr = expression.strip()
    if "=" in expr and not any(op in expr for op in ["==", "!=", ">=", "<="]):
        parts = expr.split("=", 1)
        if parts[0].strip().isidentifier():
            raise ValueError(
                f"Assignment not supported. Use just the expression: {parts[1].strip()}"
            )
    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def compute_expression(expression, variables=None):
    """Deterministic arithmetic evaluator. Returns dict."""
    try:
        clean_vars = {k: float(v) for k, v in (variables or {}).items()}
        result = safe_eval_finance(expression, clean_vars)
        return {"ok": True, "result": result}
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
        return {"error": str(exc)}


# == DB Helpers ================================================================


def _table_exists(conn, name):
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1", (name,)).fetchone()
    return row is not None


def _table_column_names(conn, name):
    if not _table_exists(conn, name):
        return set()
    return {str(r["name"]) for r in conn.execute(f"PRAGMA table_info({name})").fetchall()}


def _normalize_metric_slug(metric):
    """Normalize a metric string for ledger lookup."""
    s = metric.lower().strip()
    s = re.sub(r"\s*\d+/", "", s)
    s = re.sub(r"[^\w\s\-]", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _vintage(source_file):
    """Extract bulletin vintage score from source_file."""
    m = re.search(r"(\d{4})_(\d{2})", source_file)
    return int(m.group(1)) * 100 + int(m.group(2)) if m else 0


# == Ported Search Functions ===================================================


def search_canonical(conn, query, year=None, years=None, table_family="", limit=15):
    """Search canonical_facts table with full word matching, year filtering,
    table_family filtering, variant_count ordering, and extract_values fallback.
    Ported from server/tools.py OfficeQATools.search_canonical.
    """
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='canonical_facts'"
        ).fetchone()
        if not has_table:
            return {"results": [], "error": "canonical_facts table not found"}

        # Normalize query
        q_norm = query.lower().strip()
        q_norm = re.sub(r"\s*\d+/", "", q_norm)
        q_norm = re.sub(r"[^\w\s\-]", "", q_norm)
        q_norm = re.sub(r"\s+", " ", q_norm).strip()

        words = q_norm.split()
        if not words:
            return {"results": [], "count": 0, "query": q_norm}

        # Build year filter
        yr_list = []
        if years:
            yr_list = [int(y) for y in years]
        elif year is not None:
            yr_list = [int(year)]

        # Build WHERE: all words must match concatenated searchable fields
        concat_clauses = []
        params = []
        for w in words[:6]:
            concat_clauses.append(
                "(entity_key || ' ' || canonical_key || ' ' || row_label || ' ' || "
                "COALESCE(column_label,'') || ' ' || COALESCE(table_title,'')) LIKE ?"
            )
            params.append(f"%{w}%")
        where_parts = [" AND ".join(concat_clauses)]

        if yr_list:
            yr_ph = ",".join("?" * len(yr_list))
            where_parts.append(f"year IN ({yr_ph})")
            params.extend(yr_list)

        if table_family:
            where_parts.append("table_family = ?")
            params.append(table_family)

        where_sql = " AND ".join(where_parts)

        rows = conn.execute(
            f"""
            SELECT canonical_key, entity_key, time_key, year, month, value,
                   unit_raw, table_family, table_title, row_label, column_label,
                   period_basis, source_file, bulletin_date, variant_count
            FROM canonical_facts
            WHERE {where_sql}
            ORDER BY variant_count DESC, bulletin_date DESC, canonical_key, time_key
            LIMIT ?
        """,
            (*params, limit),
        ).fetchall()

        results = []
        for r in rows:
            results.append(
                {
                    "canonical_key": r["canonical_key"],
                    "time_key": r["time_key"],
                    "value": r["value"],
                    "unit": r["unit_raw"] or "",
                    "table_family": r["table_family"],
                    "period_basis": r["period_basis"] or "",
                    "source": r["source_file"],
                    "bulletin_date": r["bulletin_date"],
                    "table_title": r["table_title"],
                    "row_label": r["row_label"],
                    "column_label": r["column_label"],
                    "variant_count": r["variant_count"],
                    "year": r["year"],
                    "month": r["month"],
                }
            )

        # Auto-fallback: if canonical returns nothing, try extract_values
        if not results:
            fb_year = yr_list[0] if yr_list else None
            fb = extract_values(conn, query=str(query or "").strip(), year=fb_year, top_k=5)
            fb_results = fb.get("results") or []
            if fb_results:
                return {
                    "results": fb_results,
                    "count": len(fb_results),
                    "fallback_path": "extract_values",
                    "query": q_norm,
                }

        return {
            "results": results,
            "count": len(results),
            "query": q_norm,
        }
    except Exception as exc:
        return {"results": [], "error": str(exc)}


def _direct_label_search(conn, terms, year=None):
    """Search column_label and row_label directly via LIKE.
    Returns deduplicated table candidates with matching labels.
    Ported from server/tools.py OfficeQATools._direct_label_search.
    """
    if not terms:
        return []

    col_matches = {}
    year_file_clauses = []
    year_file_params = []
    if year is not None:
        for y in [year, year + 1]:
            year_file_clauses.append("ti.source_file LIKE ?")
            year_file_params.append(f"%{y}%")

    term_hits = defaultdict(set)

    # Determine which lookup path to use
    has_col_lookup = _table_exists(conn, "col_label_lookup")

    for term in terms:
        if len(term) < 3:
            continue

        yr_where = ""
        yr_params = []
        if year_file_clauses:
            yr_where = f" AND ({' OR '.join(year_file_clauses)} OR ti.min_year IS NULL)"
            yr_params = list(year_file_params)

        rows = []
        if has_col_lookup:
            try:
                rows = conn.execute(
                    f"""SELECT cl.table_pk, cl.column_label,
                              ti.table_title, ti.source_file, ti.units_line,
                              ti.min_year, ti.max_year
                       FROM col_label_lookup cl
                       JOIN table_index ti ON ti.table_pk = cl.table_pk
                       WHERE cl.col_norm LIKE ?{yr_where}
                       LIMIT 200""",
                    (f"%{term}%", *yr_params),
                ).fetchall()
            except Exception:
                rows = []
        else:
            # Fallback: search master_ledger metric_slug directly
            try:
                rows = conn.execute(
                    f"""SELECT DISTINCT ml.table_pk, ml.metric_slug AS column_label,
                              ti.table_title, ti.source_file, ti.units_line,
                              ti.min_year, ti.max_year
                       FROM master_ledger ml
                       JOIN table_index ti ON ti.table_pk = ml.table_pk
                       WHERE LOWER(REPLACE(REPLACE(ml.metric_slug, '/', ''), '.', ''))
                             LIKE ?{yr_where}
                       LIMIT 200""",
                    (f"%{term}%", *yr_params),
                ).fetchall()
            except Exception:
                rows = []

        for r in rows:
            pk = int(r["table_pk"])
            term_hits[pk].add(term)
            if pk not in col_matches:
                col_matches[pk] = {
                    "table_pk": pk,
                    "table_title": r["table_title"],
                    "source_file": r["source_file"],
                    "units": r["units_line"] or "",
                    "year_range": [r["min_year"], r["max_year"]],
                    "matched_columns": [],
                    "matched_rows": [],
                    "match_source": "column_label",
                }
            col_label = r["column_label"]
            if col_label not in col_matches[pk]["matched_columns"]:
                col_matches[pk]["matched_columns"].append(col_label)

    # Search row labels if column search found little — use master_ledger
    if len(col_matches) < 5 and _table_exists(conn, "master_ledger"):
        for term in terms:
            if len(term) < 4:
                continue
            row_yr_where = ""
            row_yr_params = []
            if year_file_clauses:
                row_yr_where = f" AND ({' OR '.join(year_file_clauses)} OR ti.min_year IS NULL)"
                row_yr_params = list(year_file_params)
            try:
                rows = conn.execute(
                    f"""SELECT DISTINCT ml.metric_slug AS row_label,
                              ml.table_pk, ti.table_title, ti.source_file,
                              ti.units_line, ti.min_year, ti.max_year
                       FROM master_ledger ml
                       JOIN table_index ti ON ti.table_pk = ml.table_pk
                       WHERE ml.metric_slug LIKE ?{row_yr_where}
                       LIMIT 200""",
                    (f"%{term}%", *row_yr_params),
                ).fetchall()
                for r in rows:
                    pk = int(r["table_pk"])
                    term_hits[pk].add(term)
                    if pk not in col_matches:
                        col_matches[pk] = {
                            "table_pk": pk,
                            "table_title": r["table_title"],
                            "source_file": r["source_file"],
                            "units": r["units_line"] or "",
                            "year_range": [r["min_year"], r["max_year"]],
                            "matched_columns": [],
                            "matched_rows": [],
                            "match_source": "row_label",
                        }
                    row_label = r["row_label"]
                    if row_label not in col_matches[pk]["matched_rows"]:
                        col_matches[pk]["matched_rows"].append(row_label[:60])
            except Exception:
                pass

    if not col_matches:
        return []

    # Year filtering
    if year is not None:
        filtered = {}
        for pk, cand in col_matches.items():
            mn, mx = cand["year_range"]
            sf = cand.get("source_file", "")
            if mn is not None and mx is not None:
                if mn <= year <= mx or str(year) in sf or str(year + 1) in sf:
                    filtered[pk] = cand
            else:
                if str(year) in sf or str(year + 1) in sf or str(year - 1) in sf:
                    filtered[pk] = cand
        col_matches = filtered

    if not col_matches:
        return []

    # Deduplicate by normalized table_title
    by_title = {}
    for cand in col_matches.values():
        title = re.sub(r"\s*\d+/\s*", " ", cand["table_title"]).strip()
        by_title.setdefault(title, []).append(cand)

    deduped = []
    for title, group in by_title.items():

        def _rep_key(g):
            hits = len(term_hits.get(g["table_pk"], set()))
            yr_ok = (
                1
                if (
                    year
                    and g["year_range"][0]
                    and g["year_range"][1]
                    and g["year_range"][0] <= year <= g["year_range"][1]
                )
                else 0
            )
            sf = g.get("source_file", "")
            sf_match = re.search(r"(\d{4})_(\d{2})", sf)
            if sf_match and year:
                sf_ym = int(sf_match.group(1)) * 12 + int(sf_match.group(2))
                target_ym = (year + 1) * 12 + 1
                proximity = -abs(sf_ym - target_ym)
            else:
                proximity = 0
            return (hits, yr_ok, proximity)

        best = max(group, key=_rep_key)

        entry = {
            "table_pk": best["table_pk"],
            "table_title": best["table_title"],
            "file_id": best["source_file"],
            "units": best["units"],
            "year_range": best["year_range"],
            "matched_columns": best["matched_columns"][:5],
            "matched_rows": best["matched_rows"][:5],
            "match_source": best["match_source"],
            "copies_across_bulletins": len(group),
            "_term_hits": len(term_hits.get(best["table_pk"], set())),
        }
        deduped.append(entry)

    # Rank by term hits + title bonus
    for entry in deduped:
        title_lower = entry["table_title"].lower()
        title_bonus = sum(1 for t in terms if t in title_lower)
        entry["_term_hits"] += title_bonus
    deduped.sort(key=lambda d: -d["_term_hits"])
    return deduped[:15]


def search_tables(conn, query, year_range=None, limit=10):
    """Search table_index for relevant tables.
    Simpler version that uses OR-based matching and year_range filtering.
    """
    words = re.sub(r"[^\w\s]", " ", query.lower()).split()
    words = [w for w in words if len(w) > 2][:6]
    if not words:
        return {"candidates": []}

    if not _table_exists(conn, "table_index"):
        return {"candidates": []}

    clauses = []
    params = []
    for w in words:
        clauses.append(
            "(table_title || ' ' || COALESCE(row_label_terms,'') || ' ' || "
            "COALESCE(distinctive_terms,'')) LIKE ?"
        )
        params.append(f"%{w}%")

    where = " OR ".join(clauses)  # OR-based: match ANY term
    where = f"({where})"

    if year_range and len(year_range) >= 2:
        yr_lo = min(int(year_range[0]), int(year_range[1]))
        yr_hi = max(int(year_range[0]), int(year_range[1]))
        where += " AND (min_year IS NULL OR max_year IS NULL OR (min_year <= ? AND max_year >= ?))"
        params.extend([yr_hi, yr_lo])

    rows = conn.execute(
        f"""
        SELECT table_pk, source_file, table_title, units_line, period_basis,
               min_year, max_year, row_count, has_month_rows,
               has_calendar_year_total, page
        FROM table_index
        WHERE {where}
        ORDER BY row_count DESC
        LIMIT ?
    """,
        (*params, limit),
    ).fetchall()

    candidates = []
    for r in rows:
        candidates.append(
            {
                "table_pk": r["table_pk"],
                "table_title": r["table_title"],
                "file_id": r["source_file"],
                "source_file": r["source_file"],
                "units_line": r["units_line"],
                "period_basis": r["period_basis"],
                "min_year": r["min_year"],
                "max_year": r["max_year"],
                "page": r["page"],
            }
        )
    return {"candidates": candidates}


def _query_table_rows_simple(
    conn, table_pk, row_label="", column_label="", year=None, year_range=None, month=None, limit=20
):
    """Query table rows. Tries master_ledger first, falls back to cell blobs
    for 0% data loss if master_ledger returns empty."""

    # Try blob path first if available (richer data, 0% loss)
    if _HAS_BLOB_DEPS and _table_exists(conn, "table_cell_blobs"):
        blob_result = query_cells_blob(
            conn,
            table_pk,
            row_label=str(row_label or ""),
            column_label=str(column_label or ""),
            year=year,
            year_range=(year_range[0], year_range[1]) if year_range else None,
            month=month,
            limit=limit,
        )
        if blob_result.get("matches"):
            return blob_result

    if not _table_exists(conn, "master_ledger"):
        return {"matches": []}

    sql = """
        SELECT metric_slug, time_key, period_basis, value, value_raw,
               source_file, table_title, row_type
        FROM master_ledger
        WHERE table_pk = ?
    """
    params = [table_pk]

    if row_label:
        rl_norm = (
            re.sub(r"\s+", " ", str(row_label).strip().lower()).replace("/", "").replace(".", "")
        )
        # Try exact first
        exact_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM master_ledger WHERE table_pk = ? AND metric_slug = ?",
            (table_pk, rl_norm),
        ).fetchone()
        if int(exact_count["cnt"]) > 0:
            sql += " AND metric_slug = ?"
            params.append(rl_norm)
        else:
            sql += " AND metric_slug LIKE ?"
            params.append(f"%{rl_norm}%")

    if column_label:
        # column_label maps to metric_slug in master_ledger (search as row)
        cl_norm = (
            re.sub(r"\s+", " ", str(column_label).strip().lower()).replace("/", "").replace(".", "")
        )
        exact_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM master_ledger WHERE table_pk = ? AND metric_slug = ?",
            (table_pk, cl_norm),
        ).fetchone()
        if int(exact_count["cnt"]) > 0:
            sql += " AND metric_slug = ?"
            params.append(cl_norm)
        else:
            sql += " AND metric_slug LIKE ?"
            params.append(f"%{cl_norm}%")

    year_values = []
    if year is not None:
        year_values.append(int(year))
    if year_range and len(year_range) >= 2:
        yr_start = min(int(year_range[0]), int(year_range[1]))
        yr_end = max(int(year_range[0]), int(year_range[1]))
        year_values.extend(range(yr_start, yr_end + 1))

    if year_values:
        # time_key contains values like "CY1940", "FY1940", "1940", "1940-06"
        time_key_patterns = []
        time_key_params = []
        for yv in year_values:
            time_key_patterns.append("time_key LIKE ?")
            time_key_params.append(f"%{yv}%")
        sql += f" AND ({' OR '.join(time_key_patterns)})"
        params.extend(time_key_params)

    if month is not None:
        # month filtering via time_key pattern like "1940-03"
        sql += " AND time_key LIKE ?"
        params.append(f"%-{int(month):02d}")

    sql += " ORDER BY metric_slug, time_key LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, tuple(params)).fetchall()

    # Get units from table_index
    units = ""
    try:
        idx_row = conn.execute(
            "SELECT units_line FROM table_index WHERE table_pk = ?", (table_pk,)
        ).fetchone()
        if idx_row and idx_row["units_line"]:
            units = str(idx_row["units_line"]).strip()
    except Exception:
        pass

    matches = []
    for r in rows:
        nv = r["value"]
        try:
            nv = float(nv) if nv is not None else None
        except (TypeError, ValueError):
            nv = None

        # Extract year from time_key (e.g. "CY1940" -> 1940, "1940-06" -> 1940)
        tk = str(r["time_key"] or "")
        yr_match = re.search(r"(\d{4})", tk)
        row_year = int(yr_match.group(1)) if yr_match else None

        # Extract month from time_key (e.g. "1940-06" -> 6)
        mo_match = re.search(r"\d{4}-(\d{2})", tk)
        row_month = int(mo_match.group(1)) if mo_match else None

        matches.append(
            {
                "row_label": r["metric_slug"],
                "column_label": r["time_key"],
                "value_raw": r["value_raw"],
                "normalized_value": nv,
                "year": row_year,
                "month": row_month,
                "time_scope": r["time_key"],
            }
        )

    return {"matches": matches, "table_info": {"units": units}}


def extract_values(conn, query, metric="", year=None, decade=None, month=None, top_k=2):
    """Search + fetch in one call. Finds tables matching query, fetches rows.
    Ported from server/tools.py OfficeQATools.extract_values.
    """
    try:
        q = str(query or "").strip()
        if not q:
            return {"results": [], "error": "query is required"}
        m = str(metric or "").strip()
        mo = int(month) if month is not None else None
        k = max(1, min(int(top_k), 8))

        # Decade detection
        decade_year_range = None
        _decade_src = str(decade or "").strip()
        if not _decade_src:
            _d_match = re.search(r"\b(\d{4})s\b", q, re.IGNORECASE)
            if _d_match:
                _decade_src = _d_match.group(0)
        if _decade_src:
            _s_match = re.match(r"^(\d{4})s$", _decade_src, re.IGNORECASE)
            _r_match = re.match(r"^(\d{4})[--](\d{4})$", _decade_src)
            if _s_match:
                _base = int(_s_match.group(1))
                decade_year_range = [_base, _base + 9]
            elif _r_match:
                decade_year_range = [int(_r_match.group(1)), int(_r_match.group(2))]

        # Phase 1: Direct label search
        _stopwords = {
            "the",
            "and",
            "for",
            "was",
            "what",
            "how",
            "total",
            "value",
            "amount",
            "from",
            "with",
            "that",
            "this",
        }
        all_terms = list(
            dict.fromkeys(
                w
                for w in re.sub(r"[^a-z0-9\s]", "", f"{q} {m}".lower()).split()
                if len(w) >= 3 and w not in _stopwords
            )
        )
        all_terms.sort(key=lambda t: -len(t))
        search_terms = all_terms[:5]

        _direct_year = int(year) if year else (decade_year_range[0] if decade_year_range else None)
        direct_candidates = _direct_label_search(conn, search_terms, year=_direct_year)

        # Boost candidates where metric matches a column label
        if m:
            m_lower = m.lower()
            for cand in direct_candidates:
                matched_cols = [c.lower() for c in cand.get("matched_columns", [])]
                if any(m_lower in c or c in m_lower for c in matched_cols):
                    cand["_term_hits"] = cand.get("_term_hits", 0) + 3
            direct_candidates.sort(key=lambda d: -d.get("_term_hits", 0))

        # Phase 2: Merge direct + term-index candidates
        search_query = f"{q} {m}".strip() if m and m.lower() not in q.lower() else q
        if decade_year_range is not None:
            yr_range = decade_year_range
        elif year is not None:
            yr_range = [int(year), int(year)]
        else:
            yr_range = None
        table_result = search_tables(conn, query=search_query, year_range=yr_range, limit=k * 3)
        term_candidates = (table_result.get("candidates") or [])[:k]

        # Merge and deduplicate
        _fn_strip = re.compile(r"\s*\d+/\s*")
        seen_pks = set()
        seen_titles = set()
        candidates = []

        def _add_candidate(cand):
            pk = cand.get("table_pk")
            title_norm = _fn_strip.sub(" ", str(cand.get("table_title") or "")).strip().lower()
            pk_int = int(pk) if pk is not None else None
            if (pk_int is not None and pk_int in seen_pks) or title_norm in seen_titles:
                return False
            if pk_int is not None:
                seen_pks.add(pk_int)
            seen_titles.add(title_norm)
            candidates.append(cand)
            return True

        for cand in direct_candidates:
            if len(candidates) >= k:
                break
            _add_candidate(cand)
        for cand in term_candidates:
            if len(candidates) >= k:
                break
            _add_candidate(cand)

        targets = [
            (
                c.get("table_pk"),
                str(c.get("file_id") or c.get("source_file") or ""),
                str(c.get("table_title") or ""),
            )
            for c in candidates
        ]

        # Fetch data from each candidate table
        fetched = {}
        for i, (pk, fid, title) in enumerate(targets):
            if pk is None:
                continue
            kwargs = {"limit": 20}
            kwargs["table_pk"] = pk
            if decade_year_range is not None:
                kwargs["year_range"] = decade_year_range
            elif year is not None:
                kwargs["year"] = int(year)
            if mo is not None:
                kwargs["month"] = mo
            # Try with metric as column_label first
            if m:
                kwargs["column_label"] = m
            result = _query_table_rows_simple(conn, **kwargs)
            if not result.get("matches") and m:
                kwargs.pop("column_label", None)
                kwargs["row_label"] = m
                result = _query_table_rows_simple(conn, **kwargs)
            if not result.get("matches") and m:
                kwargs.pop("row_label", None)
                result = _query_table_rows_simple(conn, **kwargs)
            fetched[i] = result

        # Phase 3: Assemble results
        results = []
        for i, (pk, fid, title) in enumerate(targets):
            row_result = fetched.get(i, {"matches": []})
            rows = row_result.get("matches") or []
            table_info = row_result.get("table_info", {})
            if rows:
                # Sort: CY/FY synthetic rows first
                rows.sort(
                    key=lambda r: (
                        0
                        if str(r.get("row_label", "")).startswith("CY")
                        or str(r.get("row_label", "")).startswith("FY")
                        else 1
                        if r.get("month") is None
                        else 2
                    )
                )
                compact_rows = []
                for r in rows[:10]:
                    cr = {
                        "row_label": r.get("row_label"),
                        "value": r.get("value_raw") or r.get("normalized_value"),
                        "year": r.get("year"),
                        "column_label": r.get("column_label"),
                        "time_scope": r.get("time_scope"),
                    }
                    compact_rows.append(cr)
                entry = {
                    "file_id": fid,
                    "table_title": title,
                    "table_pk": pk,
                    "rows": compact_rows,
                    "rows_returned": len(compact_rows),
                }
                if table_info.get("units"):
                    entry["units"] = table_info["units"]
                cand = candidates[i] if i < len(candidates) else {}
                if cand.get("matched_columns"):
                    entry["matched_columns"] = cand["matched_columns"]
                if cand.get("units"):
                    entry.setdefault("units", cand["units"])
                # Add period_basis
                try:
                    _ti = conn.execute(
                        "SELECT period_basis FROM table_index WHERE table_pk = ?", (pk,)
                    ).fetchone()
                    if _ti and _ti["period_basis"]:
                        entry["period_basis"] = _ti["period_basis"]
                except Exception:
                    pass
                results.append(entry)

        out = {
            "results": results,
            "count": sum(len(r["rows"]) for r in results),
            "query": q,
        }
        if year is not None:
            out["year"] = year

        # Compute verdict (best single-value answer)
        if results:
            best_row = None
            best_score = -1
            best_table_title = ""
            best_units = ""
            q_lower = f"{q} {m}".lower()

            def _clean_label(s):
                if not s:
                    return ""
                s = s.lower()
                s = re.sub(r"[',.\-]", "", s)
                s = re.sub(r"\d+\s*/", "", s)
                return " ".join(s.split())

            metric_clean = _clean_label(m)
            query_terms_clean = [_clean_label(t) for t in all_terms[:5] if len(t) > 3]

            query_wants_calendar = any(
                x in q_lower for x in ["calendar year", "calendar month", "cy"]
            )
            query_wants_fiscal = any(x in q_lower for x in ["fiscal year", "fiscal period", "fy"])
            _total_words = {"total", "aggregate", "sum", "all", "combined", "overall"}
            query_wants_total = any(w in q_lower for w in _total_words)
            query_wants_specific = bool(m)

            for res in results:
                t_title = (res.get("table_title") or "").lower()
                t_units = res.get("units", "")
                for row in res.get("rows", []):
                    score = 0
                    val = row.get("value")
                    if val is None or str(val).strip() in ("", "...", "\u2014", "-", "None"):
                        continue

                    row_year = row.get("year")
                    if year is not None:
                        if row_year == year:
                            score += 6
                        elif row_year is not None and row_year != year:
                            score -= 5
                        else:
                            score -= 2

                    rl = (row.get("row_label") or "").lower()
                    cl = (row.get("column_label") or "").lower()
                    rl_clean = _clean_label(rl)
                    cl_clean = _clean_label(cl)

                    for term in query_terms_clean:
                        if term in rl_clean:
                            score += 3
                        if term in cl_clean:
                            score += 2
                        if term in t_title:
                            score += 1

                    if metric_clean:
                        if metric_clean == rl_clean:
                            score += 10
                        elif metric_clean in rl_clean:
                            score += 5
                        elif metric_clean == cl_clean:
                            score += 8
                        elif metric_clean in cl_clean:
                            score += 4

                    # CY/FY synthetic row bonus
                    if (
                        query_wants_calendar
                        and rl.startswith("cy")
                        or query_wants_fiscal
                        and rl.startswith("fy")
                    ):
                        score += 15

                    # Hierarchy: total vs specific
                    is_total_row = rl.strip().rstrip(".:") in ("total", "net total", "grand total")
                    title_has_metric = metric_clean and metric_clean in _clean_label(t_title)
                    if query_wants_total and is_total_row and title_has_metric:
                        score += 10

                    if score > best_score:
                        best_score = score
                        best_row = row
                        best_table_title = res.get("table_title", "")
                        best_units = t_units

            if best_row and best_score >= 3:
                verdict_val = best_row.get("value")
                out["verdict"] = {
                    "value": verdict_val,
                    "row_label": best_row.get("row_label"),
                    "column_label": best_row.get("column_label"),
                    "table_title": best_table_title,
                    "units": best_units,
                    "confidence": "high" if best_score >= 6 else "medium",
                }

        return out
    except Exception as exc:
        return {"results": [], "error": str(exc)}


def _broad_slug_search(conn, metric_norm, time_keys, row_types):
    """Search master_ledger broadly for matching slugs.
    Ported from server/tools.py OfficeQATools._broad_slug_search.
    """
    if not time_keys or not _table_exists(conn, "master_ledger"):
        return []
    ph = ",".join("?" * len(time_keys))
    rt_ph = ",".join("?" * len(row_types))

    all_rows = []
    seen_keys = set()

    def _add_rows(rows):
        for r in rows:
            key = (r["metric_slug"], r["time_key"], r["source_file"])
            if key not in seen_keys:
                seen_keys.add(key)
                all_rows.append(r)

    # 1. Exact match
    try:
        rows = conn.execute(
            f"""SELECT metric_slug, time_key, period_basis, value, value_raw,
                       table_pk, source_file, table_title, row_type
                FROM master_ledger
                WHERE metric_slug = ? AND time_key IN ({ph})
                  AND row_type IN ({rt_ph})
                ORDER BY source_file DESC""",
            (metric_norm, *time_keys, *row_types),
        ).fetchall()
        _add_rows(rows)
    except Exception:
        pass

    # 2. LIKE with progressively fewer words
    words = metric_norm.split()
    if len(words) >= 2:
        for n_words in range(len(words), 1, -1):
            subset = words[:n_words]
            like_pattern = "%" + "%".join(subset) + "%"
            try:
                rows = conn.execute(
                    f"""SELECT metric_slug, time_key, period_basis, value, value_raw,
                               table_pk, source_file, table_title, row_type
                        FROM master_ledger
                        WHERE metric_slug LIKE ? AND time_key IN ({ph})
                          AND row_type IN ({rt_ph})
                        ORDER BY source_file DESC
                        LIMIT 200""",
                    (like_pattern, *time_keys, *row_types),
                ).fetchall()
                _add_rows(rows)
            except Exception:
                pass

    return all_rows


def resolve_numeric_evidence(conn, question, metric, year=None, period_basis=""):
    """Multi-bulletin evidence resolver. Primary entry point for numeric lookups.
    Ported from server/tools.py OfficeQATools.resolve_numeric_evidence.
    """
    try:
        metric_clean = str(metric or question or "").strip()
        yr = int(year) if year is not None else None
        pb_want = str(period_basis or "").strip().lower()

        # Helper: keyword overlap score against question
        q_words = set(re.sub(r"[^\w\s]", "", (question or "").lower()).split())
        _stopwords = {
            "the",
            "and",
            "for",
            "was",
            "what",
            "how",
            "total",
            "value",
            "amount",
            "from",
            "with",
            "that",
            "this",
            "a",
            "an",
            "of",
            "in",
            "to",
            "is",
            "are",
            "were",
            "be",
        }
        q_keywords = q_words - _stopwords

        def _title_overlap(title):
            title_words = set(re.sub(r"[^\w\s]", "", title.lower()).split())
            return len(title_words & q_keywords)

        def _get_units(table_pk):
            try:
                row = conn.execute(
                    "SELECT units_line FROM table_index WHERE table_pk = ?", (table_pk,)
                ).fetchone()
                return str(row["units_line"]).strip() if row and row["units_line"] else ""
            except Exception:
                return ""

        # 1. Primary path: query master_ledger directly
        # time_key formats: plain "1940", "1940-06", "CY1940", "FY1940"
        if yr:
            if pb_want == "calendar":
                time_keys = [str(yr), f"CY{yr}"]
            elif pb_want == "fiscal":
                time_keys = [str(yr), f"FY{yr}"]
            else:
                time_keys = [str(yr), f"CY{yr}", f"FY{yr}"]
        else:
            time_keys = []

        metric_norm = _normalize_metric_slug(metric_clean)

        ledger_rows = []
        if time_keys:
            ledger_rows = _broad_slug_search(
                conn,
                metric_norm,
                time_keys,
                row_types=(
                    "annual_total",
                    "point_estimate",
                    "synthetic_cy_total",
                    "synthetic_fy_total",
                    "",
                ),
            )

        # 2. Build candidates grouped by logical table
        title_best = {}
        for r in ledger_rows:
            val_raw = str(r["value_raw"] or r["value"] or "").strip()
            if not val_raw:
                continue
            src = str(r["source_file"] or "")
            ttitle = str(r["table_title"] or "") if "table_title" in r.keys() else ""
            if not ttitle:
                try:
                    ts = conn.execute(
                        "SELECT table_title FROM table_index WHERE table_pk = ?", (r["table_pk"],)
                    ).fetchone()
                    ttitle = str(ts["table_title"]) if ts else f"unknown_{r['table_pk']}"
                except Exception:
                    ttitle = f"unknown_{r['table_pk']}"

            numeric_str = re.sub(r"[^\d.\-]", "", val_raw.replace(",", ""))
            try:
                norm_val = float(numeric_str) if numeric_str else None
            except ValueError:
                norm_val = None

            pb_row = str(r["period_basis"] or r["time_key"] or "").lower()
            pb_match = (not pb_want) or pb_want in pb_row or pb_want in str(r["time_key"]).lower()
            vin = _vintage(src)
            title_score = _title_overlap(ttitle)
            confidence = round(
                min(
                    0.95,
                    0.5
                    + (vin / 200000.0)
                    + (0.15 if pb_match else 0.0)
                    + (0.05 * min(title_score, 2)),
                ),
                2,
            )

            existing = title_best.get(ttitle)
            if existing is None or vin > existing["bulletin_vintage"]:
                title_best[ttitle] = {
                    "value": val_raw,
                    "normalized_value": norm_val,
                    "table_title": ttitle,
                    "table_pk": r["table_pk"],
                    "source_file": src,
                    "time_key": str(r["time_key"] or ""),
                    "period_basis": pb_row,
                    "bulletin_vintage": vin,
                    "confidence": confidence,
                    "title_keyword_overlap": title_score,
                    "units": _get_units(r["table_pk"]),
                }

        best_candidates = list(title_best.values())
        best_candidates.sort(
            key=lambda c: (
                int(pb_want in c["period_basis"]) if pb_want else 0,
                c.get("title_keyword_overlap", 0),
                c["bulletin_vintage"],
            ),
            reverse=True,
        )
        best_candidates = best_candidates[:6]

        # 3. Fallback A: canonical_facts search
        if not best_candidates and yr:
            try:
                canon_out = search_canonical(conn, query=metric_clean, year=yr, limit=10)
                for cr in canon_out.get("results") or []:
                    val = cr.get("value")
                    if val is None:
                        continue
                    val_raw_c = str(val)
                    numeric_str = re.sub(r"[^\d.\-]", "", val_raw_c.replace(",", ""))
                    try:
                        norm_val_c = float(numeric_str) if numeric_str else None
                    except ValueError:
                        norm_val_c = None
                    ttitle_c = str(cr.get("table_title") or "")
                    src_c = str(cr.get("source") or "")
                    vin_c = _vintage(src_c)
                    best_candidates.append(
                        {
                            "value": val_raw_c,
                            "normalized_value": norm_val_c,
                            "table_title": ttitle_c,
                            "table_pk": 0,
                            "source_file": src_c,
                            "time_key": str(cr.get("time_key") or yr),
                            "period_basis": str(cr.get("period_basis") or ""),
                            "bulletin_vintage": vin_c,
                            "confidence": 0.45,
                            "title_keyword_overlap": _title_overlap(ttitle_c),
                            "units": str(cr.get("unit") or ""),
                            "_fallback_source": "canonical",
                        }
                    )
                best_candidates = best_candidates[:6]
            except Exception:
                pass

        # 4. Fallback B: table_index search + query_table_rows
        if not best_candidates and yr:
            res = search_tables(conn, query=metric_clean, year_range=[yr, yr], limit=6)
            for cand in res.get("candidates", []):
                pk = cand.get("table_pk")
                if not pk:
                    continue
                row_res = _query_table_rows_simple(conn, table_pk=pk, year=yr, limit=10)
                rows_fb = row_res.get("matches", [])
                total_rows_fb = [
                    r for r in rows_fb if "total" in str(r.get("column_label", "")).lower()
                ] or rows_fb[:2]
                for row in total_rows_fb[:2]:
                    val_raw = row.get("value_raw") or ""
                    if not val_raw:
                        continue
                    src = cand.get("source_file", cand.get("file_id", ""))
                    ttitle_fb = cand.get("table_title", "")
                    numeric_str = re.sub(r"[^\d.\-]", "", str(val_raw).replace(",", ""))
                    try:
                        norm_val_fb = float(numeric_str) if numeric_str else None
                    except ValueError:
                        norm_val_fb = None
                    best_candidates.append(
                        {
                            "value": str(val_raw),
                            "normalized_value": norm_val_fb,
                            "table_pk": pk,
                            "source_file": src,
                            "table_title": ttitle_fb,
                            "time_key": str(yr),
                            "period_basis": str(cand.get("period_basis") or ""),
                            "bulletin_vintage": _vintage(src),
                            "confidence": 0.4,
                            "title_keyword_overlap": _title_overlap(ttitle_fb),
                            "units": str(cand.get("units_line") or ""),
                        }
                    )
            best_candidates = best_candidates[:6]

        if not best_candidates:
            return {"status": "no_data", "best_candidates": [], "recommended_value": None}

        top = best_candidates[0]
        all_norm = [
            c["normalized_value"] for c in best_candidates if c.get("normalized_value") is not None
        ]
        unique_vals = set(round(v, 1) for v in all_norm) if all_norm else set()
        has_disagreement = len(unique_vals) > 1

        result = {
            "status": "ambiguous_candidates" if has_disagreement else "high_confidence",
            "recommended_value": top["value"],
            "recommended_table_pk": top["table_pk"],
            "best_candidates": best_candidates,
        }
        if has_disagreement:
            result["ambiguity_reason"] = (
                "Same metric appears in multiple bulletin vintages with different values. "
                "Use the highest bulletin_vintage (most recently revised data)."
            )
        return result

    except Exception as exc:
        return {"error": str(exc)}


# == Ledger & Time-Series Queries ==============================================


def search_ledger(conn, metric, year=None, period_basis="", years=None):
    """Search by metric name. Tries blob-based lookup first (0% data loss),
    falls back to master_ledger if available."""
    try:
        # Blob path: find tables by label, then query cells
        if _HAS_BLOB_DEPS and _table_exists(conn, "table_cell_blobs"):
            yr = int(year) if year else (int(years[0]) if years else None)
            search_terms = [
                t for t in re.sub(r"[^\w\s]", "", str(metric)).lower().split() if len(t) >= 3
            ]
            if search_terms:
                table_matches = find_tables_by_label(conn, search_terms, year=yr, limit=5)
                blob_matches = []
                for tm in table_matches:
                    pk = tm["table_pk"]
                    yr_range = None
                    if years and len(years) >= 2:
                        yr_range = (int(min(years)), int(max(years)))
                    elif yr:
                        yr_range = (yr, yr)
                    result = query_cells_blob(
                        conn, pk, row_label=metric, year=yr, year_range=yr_range, limit=20
                    )
                    for m in result.get("matches", []):
                        if m.get("normalized_value") is not None:
                            blob_matches.append(m)
                    # Also try column_label
                    if not result.get("matches"):
                        result = query_cells_blob(
                            conn, pk, column_label=metric, year=yr, year_range=yr_range, limit=20
                        )
                        for m in result.get("matches", []):
                            if m.get("normalized_value") is not None:
                                blob_matches.append(m)
                if blob_matches:
                    # Deduplicate by (row_label, column_label, value)
                    seen = set()
                    deduped = []
                    for m in blob_matches:
                        key = (m.get("row_label"), m.get("column_label"), m.get("normalized_value"))
                        if key not in seen:
                            seen.add(key)
                            deduped.append(m)
                    return {
                        "matches": deduped[:30],
                        "count": len(deduped),
                        "status": "success",
                        "source": "cell_blobs",
                    }

        if not _table_exists(conn, "master_ledger"):
            return {
                "matches": [],
                "count": 0,
                "status": "error",
                "error": "master_ledger table not found",
            }

        metric_norm = _normalize_metric_slug(metric)

        yr_list = []
        if years:
            yr_list = [int(y) for y in years]
        elif year is not None:
            yr_list = [int(year)]

        time_keys = []
        for y in yr_list:
            if period_basis == "calendar":
                time_keys.append(f"CY{y}")
            elif period_basis == "fiscal":
                time_keys.append(f"FY{y}")
            elif period_basis == "monthly":
                for mo in range(1, 13):
                    time_keys.append(f"{y}-{mo:02d}")
            elif period_basis == "annual":
                time_keys.append(str(y))
            else:
                time_keys.extend([f"CY{y}", f"FY{y}", str(y)])

        rows = []
        if time_keys:
            placeholders = ",".join("?" * len(time_keys))
            rows = conn.execute(
                f"""SELECT metric_slug, time_key, period_basis, value, value_raw,
                           table_pk, source_file, table_title, row_type
                    FROM master_ledger
                    WHERE metric_slug = ? AND time_key IN ({placeholders})
                    ORDER BY time_key, source_file DESC""",
                (metric_norm, *time_keys),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    f"""SELECT metric_slug, time_key, period_basis, value, value_raw,
                               table_pk, source_file, table_title, row_type
                        FROM master_ledger
                        WHERE metric_slug LIKE ? AND time_key IN ({placeholders})
                        ORDER BY time_key, source_file DESC
                        LIMIT 50""",
                    (f"%{metric_norm}%", *time_keys),
                ).fetchall()
        else:
            rows = conn.execute(
                """SELECT metric_slug, time_key, period_basis, value, value_raw,
                          table_pk, source_file, table_title, row_type
                   FROM master_ledger
                   WHERE metric_slug = ?
                   ORDER BY time_key LIMIT 50""",
                (metric_norm,),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """SELECT metric_slug, time_key, period_basis, value, value_raw,
                              table_pk, source_file, table_title, row_type
                       FROM master_ledger
                       WHERE metric_slug LIKE ?
                       ORDER BY time_key LIMIT 50""",
                    (f"%{metric_norm}%",),
                ).fetchall()

        seen = {}
        for r in rows:
            key = (r["metric_slug"], r["time_key"])
            entry = {
                "metric": r["metric_slug"],
                "time_key": r["time_key"],
                "period_basis": r["period_basis"],
                "value": r["value"],
                "value_raw": r["value_raw"],
                "table_pk": r["table_pk"],
                "source": r["source_file"],
                "table_title": r["table_title"],
                "row_type": r["row_type"],
            }
            if key not in seen or entry["row_type"] and not seen[key]["row_type"]:
                seen[key] = entry

        raw_results = sorted(seen.values(), key=lambda x: x["time_key"])

        units = ""
        scale = 1
        if raw_results:
            pk = raw_results[0]["table_pk"]
            ti = conn.execute(
                "SELECT units_line FROM table_index WHERE table_pk = ?", (pk,)
            ).fetchone()
            units = ti["units_line"] if ti and ti["units_line"] else ""
            if "thousand" in str(units).lower():
                scale = 1000
            elif "million" in str(units).lower():
                scale = 1_000_000
            elif "billion" in str(units).lower():
                scale = 1_000_000_000

        compact_matches = []
        for r in raw_results[:5]:
            ym = re.search(r"\d{4}", r["time_key"])
            compact_matches.append(
                {
                    "metric": r["metric"],
                    "year": int(ym.group(0)) if ym else None,
                    "time_key": r["time_key"],
                    "value": r["value"],
                    "value_raw": r["value_raw"],
                    "unit": units,
                    "scale": scale,
                    "basis": r["period_basis"],
                    "source_table_id": f"tbl_{r['table_pk']}",
                    "source_doc": r["source"],
                    "table_title": r["table_title"],
                }
            )

        return {
            "matches": compact_matches,
            "count": len(compact_matches),
            "status": "success" if compact_matches else "no_results",
        }
    except Exception as exc:
        return {"matches": [], "count": 0, "status": "error", "error": str(exc)}


def get_time_series(
    conn,
    metric,
    year_start,
    year_end,
    period_basis="calendar",
    query="",
    month_start=None,
    month_end=None,
    top_k=3,
):
    """Fetch a time-series for a metric across a year range.
    Ported from server/tools.py OfficeQATools.get_time_series.
    """
    try:
        m = str(metric or "").strip()
        if not m:
            return {"series": {}, "error": "metric is required"}
        yr_start = int(year_start)
        yr_end = int(year_end)
        if yr_start > yr_end:
            yr_start, yr_end = yr_end, yr_start
        search_q = str(query or "").strip() or m
        if m.lower() not in search_q.lower():
            search_q = f"{search_q} {m}"

        table_result = search_tables(
            conn, query=search_q, year_range=[yr_start, yr_end], limit=top_k * 3
        )
        candidates = (table_result.get("candidates") or [])[:top_k]

        if not candidates:
            canon_series = {}
            canon_source = ""
            try:
                canon = search_canonical(
                    conn, query=search_q, years=list(range(yr_start, yr_end + 1)), limit=50
                )
                for cr in canon.get("results") or []:
                    yr_val = cr.get("year")
                    mo_val = cr.get("month")
                    val = cr.get("value")
                    if yr_val is not None and val is not None:
                        pk_key = f"{yr_val}-{int(mo_val):02d}" if mo_val else str(yr_val)
                        if pk_key not in canon_series:
                            canon_series[pk_key] = val
                if canon_series and not canon_source:
                    first = (canon.get("results") or [{}])[0]
                    canon_source = first.get("table_title", "")
            except Exception:
                pass
            if canon_series:
                requested_years = list(range(yr_start, yr_end + 1))
                found_years = {
                    int(re.match(r"^(\d{4})", str(k)).group(1))
                    for k in canon_series
                    if re.match(r"^(\d{4})", str(k))
                }
                return {
                    "metric": m,
                    "table_title": canon_source,
                    "series": canon_series,
                    "count": len(canon_series),
                    "fallback_path": "canonical_facts",
                    "coverage": {
                        "requested_years": requested_years,
                        "found": len(found_years),
                        "missing_years": [y for y in requested_years if y not in found_years],
                    },
                }
            return {
                "metric": m,
                "series": {},
                "count": 0,
                "coverage": {
                    "requested_years": list(range(yr_start, yr_end + 1)),
                    "found": 0,
                    "missing": list(range(yr_start, yr_end + 1)),
                },
            }

        best_series = {}
        best_table_pk = None
        best_table_title = ""
        all_candidate_series = []

        for cand in candidates:
            pk = cand.get("table_pk")
            fid = str(cand.get("file_id") or cand.get("source_file") or "")
            title = str(cand.get("table_title") or "")
            if pk is None:
                continue
            result = _query_table_rows_simple(
                conn, table_pk=pk, column_label=m, year_range=[yr_start, yr_end], limit=500
            )
            if not result.get("matches"):
                result = _query_table_rows_simple(
                    conn, table_pk=pk, row_label=m, year_range=[yr_start, yr_end], limit=500
                )
            if not result.get("matches"):
                result = _query_table_rows_simple(
                    conn, table_pk=pk, year_range=[yr_start, yr_end], limit=500
                )
            row_matches = result.get("matches") or []
            if not row_matches:
                continue
            series = {}
            for r in row_matches:
                yr = r.get("year")
                mo = r.get("month")
                ts = r.get("time_scope", "")
                val = r.get("value_raw") or r.get("normalized_value")
                if mo:
                    period_key = f"{yr}-{int(mo):02d}"
                elif ts and re.match(r"^\d{4}-\d{2}$", str(ts)):
                    period_key = str(ts)
                else:
                    period_key = str(yr) if yr else None
                if period_key and val is not None and period_key not in series:
                    series[period_key] = val
            if series:
                all_candidate_series.append((pk, fid, title, series))

        if all_candidate_series:
            best_item = max(
                all_candidate_series,
                key=lambda it: sum(1 for k in it[3] if "-" in str(k)) * 1000 + len(it[3]),
            )
            best_table_pk, _, best_table_title, best_series = best_item
            for _pk, _fid, _title, cand_series in all_candidate_series:
                for k, v in cand_series.items():
                    if k not in best_series:
                        best_series[k] = v

        requested_years = list(range(yr_start, yr_end + 1))
        found_years = {
            int(re.match(r"^(\d{4})", str(k)).group(1))
            for k in best_series
            if re.match(r"^(\d{4})", str(k))
        }
        missing_years = [y for y in requested_years if y not in found_years]

        if month_start is not None or month_end is not None:
            ms = int(month_start) if month_start else 1
            me = int(month_end) if month_end else 12
            filtered = {}
            for k, v in best_series.items():
                mo_match = re.match(r"^\d{4}-(\d{2})$", str(k))
                if mo_match:
                    mo_val = int(mo_match.group(1))
                    yr_val = int(k.split("-")[0])
                    if yr_val == yr_start and mo_val < ms:
                        continue
                    if yr_val == yr_end and mo_val > me:
                        continue
                    filtered[k] = v
                else:
                    filtered[k] = v
            best_series = filtered

        best_units = ""
        best_unit_scale = 1
        if best_table_pk:
            try:
                _u = conn.execute(
                    "SELECT units_line FROM table_index WHERE table_pk = ?", (best_table_pk,)
                ).fetchone()
                if _u and _u["units_line"]:
                    best_units = str(_u["units_line"]).strip()
                    ul = best_units.lower()
                    if "thousand" in ul:
                        best_unit_scale = 1_000
                    elif "billion" in ul:
                        best_unit_scale = 1_000_000_000
                    elif "million" in ul:
                        best_unit_scale = 1_000_000
            except Exception:
                pass

        return {
            "metric": m,
            "table_title": best_table_title,
            "table_pk": best_table_pk,
            "series": best_series,
            "count": len(best_series),
            "units": best_units,
            "unit_scale": best_unit_scale,
            "period_basis": period_basis,
            "coverage": {
                "requested_years": requested_years,
                "found": len(found_years),
                "missing_years": missing_years,
            },
        }
    except Exception as exc:
        return {"series": {}, "error": str(exc)}


def get_multi_year_series(conn, metric, years, period_basis="", top_k=3):
    """Extract a time-series for a metric across multiple (possibly non-contiguous) years.
    Ported from server/tools.py OfficeQATools.get_multi_year_series.
    """
    try:
        m = str(metric or "").strip()
        if not m:
            return {"series": {}, "error": "metric is required"}
        yrs = sorted({int(y) for y in years if str(y).strip()})
        if not yrs:
            return {"series": {}, "error": "years are required"}

        pb = str(period_basis or "").strip().lower()

        if pb in ("calendar", "fiscal") and _table_exists(conn, "master_ledger"):
            metric_norm = _normalize_metric_slug(m)
            if pb == "calendar":
                time_keys = [f"CY{y}" for y in yrs]
                row_types = ("synthetic_cy_total",)
            else:
                time_keys = [f"FY{y}" for y in yrs] + [str(y) for y in yrs]
                row_types = ("synthetic_fy_total", "annual_total")

            ledger_rows = _broad_slug_search(conn, metric_norm, time_keys, row_types)
            if ledger_rows:
                title_series = {}
                for r in ledger_rows:
                    ttitle = str(r["table_title"]) if "table_title" in r.keys() else ""
                    if not ttitle:
                        try:
                            ts = conn.execute(
                                "SELECT table_title FROM table_summary WHERE table_pk = ?",
                                (r["table_pk"],),
                            ).fetchone()
                            ttitle = str(ts["table_title"]) if ts else ""
                        except Exception:
                            ttitle = ""
                    tk = str(r["time_key"])
                    src = str(r["source_file"] or "")
                    val = r["value"]
                    yr_match = re.search(r"(\d{4})", tk)
                    if not yr_match:
                        continue
                    yr_val = int(yr_match.group(1))
                    if yr_val not in yrs:
                        continue
                    if ttitle not in title_series:
                        title_series[ttitle] = {}
                    existing = title_series[ttitle].get(yr_val)
                    if existing is None or src > existing["src"]:
                        title_series[ttitle][yr_val] = {"value": val, "src": src}

                if title_series:
                    best_title = max(title_series, key=lambda t: len(title_series[t]))
                    best = title_series[best_title]
                    series = {y: {"value": best[y]["value"]} for y in yrs if y in best}
                    units_line = ""
                    try:
                        ts_meta = conn.execute(
                            "SELECT units_line FROM table_index WHERE table_pk = ?",
                            (ledger_rows[0]["table_pk"],),
                        ).fetchone()
                        if ts_meta and ts_meta["units_line"]:
                            units_line = str(ts_meta["units_line"])
                    except Exception:
                        pass
                    out = {
                        "metric": m,
                        "years": yrs,
                        "series": series,
                        "count": len(series),
                        "table_title": best_title,
                        "period_basis": pb,
                        "missing_years": [y for y in yrs if y not in series],
                    }
                    if units_line:
                        out["units"] = units_line
                    return out

        yr_start, yr_end = min(yrs), max(yrs)
        ts_result = get_time_series(
            conn, metric=m, year_start=yr_start, year_end=yr_end, top_k=top_k
        )
        full_series = ts_result.get("series", {})
        series = {}
        for y in yrs:
            if str(y) in full_series:
                series[y] = {"value": full_series[str(y)]}
            else:
                monthly = {k: v for k, v in full_series.items() if k.startswith(f"{y}-")}
                if monthly:
                    series[y] = {"monthly_values": monthly}
        out = {
            "metric": m,
            "years": yrs,
            "series": series,
            "count": len(series),
            "table_pk": ts_result.get("table_pk"),
            "table_title": ts_result.get("table_title"),
        }
        if ts_result.get("units"):
            out["units"] = ts_result["units"]
            out["unit_scale"] = ts_result.get("unit_scale", 1)
        return out
    except Exception as exc:
        return {"series": {}, "error": str(exc)}


# == Formatters for ledger/time-series results =================================


def format_ledger_results(ledger_result):
    """Format search_ledger output as text for LLM context."""
    matches = ledger_result.get("matches", [])
    if not matches:
        return f"[search_ledger: {ledger_result.get('status', 'unknown')} - no matches]"
    lines = [f"LEDGER RESULTS ({len(matches)} matches):"]
    for mt in matches:
        lines.append(
            f"  metric={mt.get('metric', '')} | year={mt.get('year', '')} | "
            f"time_key={mt.get('time_key', '')} | value={mt.get('value', '')} | "
            f"value_raw={mt.get('value_raw', '')} | unit={mt.get('unit', '')} | "
            f"scale={mt.get('scale', 1)} | basis={mt.get('basis', '')} | "
            f"table={mt.get('table_title', '')[:60]} | src={mt.get('source_doc', '')}"
        )
    return "\n".join(lines)


def format_time_series(ts_result):
    """Format get_time_series / get_multi_year_series output as text for LLM."""
    series = ts_result.get("series", {})
    metric = ts_result.get("metric", "")
    if not series:
        return f"[time_series: no data for '{metric}']"
    title = ts_result.get("table_title", "")
    units = ts_result.get("units", "")
    pb = ts_result.get("period_basis", "")
    lines = [f"TIME SERIES for '{metric}' (table={title[:60]}, units={units}, basis={pb}):"]
    for k in sorted(series.keys(), key=lambda x: str(x)):
        v = series[k]
        val = v.get("value", v.get("monthly_values", v)) if isinstance(v, dict) else v
        lines.append(f"  {k}: {val}")
    coverage = ts_result.get("coverage", {})
    if coverage.get("missing_years"):
        lines.append(f"  MISSING YEARS: {coverage['missing_years']}")
    return "\n".join(lines)


# == Corpus File Reader ========================================================


def read_table_from_file(source_file, page_or_line=None):
    """Read table content from corpus TXT file."""
    fpath = Path(CORPUS_DIR) / source_file
    if not fpath.exists():
        return f"File not found: {source_file}"

    lines = fpath.read_text(errors="replace").splitlines()

    start = 0
    if page_or_line and page_or_line > 0:
        start = max(0, page_or_line - 20)

    end = min(len(lines), start + 150)
    return "\n".join(f"{i + 1:5d} | {lines[i]}" for i in range(start, end))


# == Raw Corpus Search (fallback when DB fails) ================================


def _build_keyword_index_if_needed():
    """Build keyword index from corpus at startup (once). Returns path to index file."""
    idx_path = os.environ.get("KEYWORD_INDEX_PATH", "/tmp/keyword_index.txt")
    if os.path.exists(idx_path) and os.path.getsize(idx_path) > 1000:
        return idx_path
    build_script = os.environ.get("BUILD_SCRIPT", "")
    if not build_script:
        # Try to find it
        for candidate in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_index.py"),
            "/installed-agent/build_index.py",
        ]:
            if os.path.exists(candidate):
                build_script = candidate
                break
    if build_script and os.path.exists(build_script):
        try:
            subprocess.run(["python3", build_script], capture_output=True, timeout=60)
        except Exception:
            pass
    return idx_path if os.path.exists(idx_path) else None


def _merge_multi_row_headers(all_lines, separator_line_idx):
    """Scan upward from separator row to find ALL header rows, then merge them.

    Treasury tables often have multi-row headers like:
        | Category | 1940 | | | | 1941 |
        | | Jan | Feb | Mar | Total | Jan |
        | --- | --- | --- | --- | --- | --- |

    Returns merged header list like ["Category", "1940 > Jan", "1940 > Feb", ...].
    """
    # Collect header rows above separator (scan upward)
    header_rows = []
    j = separator_line_idx - 1
    while j >= 0:
        row = all_lines[j].strip()
        if row.startswith("|") and "---" not in row:
            # Split keeping empty cells (important for multi-row merge)
            raw_cells = row.split("|")
            # First and last are empty from leading/trailing pipe
            if len(raw_cells) >= 3:
                cells = [c.strip() for c in raw_cells[1:-1]]
                header_rows.insert(0, cells)  # prepend (we're scanning upward)
            else:
                break
        else:
            break
        j -= 1

    if not header_rows:
        return None

    if len(header_rows) == 1:
        # Single header row — just strip empties from display but keep positions
        return [c if c else "" for c in header_rows[0]]

    # Multi-row merge: for each column position, build "Parent > Child" labels
    num_cols = max(len(r) for r in header_rows)
    merged = [""] * num_cols

    for col_idx in range(num_cols):
        parts = []
        last_nonempty = ""
        for row_idx, row in enumerate(header_rows):
            cell = row[col_idx].strip() if col_idx < len(row) else ""
            if cell:
                last_nonempty = cell
                parts.append(cell)
            else:
                # Inherit from same column in previous row if empty
                # (fill-down: "1940" covers Jan/Feb/Mar columns)
                pass
        if parts:
            merged[col_idx] = " > ".join(parts)
        else:
            # All empty — try to inherit from left neighbor's top-level
            # Walk left to find the nearest non-empty top-row cell
            for left in range(col_idx - 1, -1, -1):
                top_cell = header_rows[0][left].strip() if left < len(header_rows[0]) else ""
                if top_cell:
                    # Use top cell + this column's bottom rows
                    bottom_parts = []
                    for row_idx in range(1, len(header_rows)):
                        c = (
                            header_rows[row_idx][col_idx].strip()
                            if col_idx < len(header_rows[row_idx])
                            else ""
                        )
                        if c:
                            bottom_parts.append(c)
                    if bottom_parts:
                        merged[col_idx] = top_cell + " > " + " > ".join(bottom_parts)
                    else:
                        merged[col_idx] = top_cell
                    break

    return merged


def _row_to_vertical(row_label, headers, cells):
    """Convert a data row to vertical key-value format string.

    Returns: "ROW: National defense\n  Jan.: 132\n  Feb.: 129\n  ..."
    Simplifies headers by stripping misleading year prefixes from month columns.
    """
    # Detect if headers are month-based (e.g., "1939 > Dec.", "1940 > Jan.")
    _month_abbrs = {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }

    parts = [f"ROW: {row_label}"]
    col_index = 0
    for k, val in enumerate(cells[1:], 1):
        col_name = headers[k] if k < len(headers) else f"col_{k}"
        if "unnamed" in col_name.lower() or not col_name:
            col_name = f"col_{k}"

        # Simplify month headers: "1939 > Jan." → "Jan." to avoid year confusion
        # The year context comes from the table title and question, not from column headers
        col_lower = col_name.lower().strip().rstrip(".")
        col_parts = [p.strip() for p in col_name.split(">")]
        if len(col_parts) >= 2:
            last_part = col_parts[-1].strip().rstrip(".").lower()
            if last_part in _month_abbrs:
                # It's a month column — use just the month name with index
                col_index += 1
                col_name = f"{col_parts[-1].strip()} (month {col_index})"

        val_str = val.strip() if val.strip() else "-"
        parts.append(f"  {col_name}: {val_str}")
    return "\n".join(parts)


def _parse_table_at(all_lines, start, end):
    """Parse a markdown table region, returning merged headers, vertical data rows,
    table_data dicts, and table_title/units from context above.

    Returns (headers, vertical_data, table_data, table_title, units, separator_idx).
    """
    headers = None
    found_sep = False
    separator_idx = None
    table_data = []
    vertical_data = []
    table_title = ""
    units = ""

    # Extract title and units from lines above the table start
    for ctx_i in range(max(0, start - 10), start + 5):
        if ctx_i >= len(all_lines):
            break
        ctx = all_lines[ctx_i].strip()
        if re.search(r"\(.*(?:millions|thousands|dollars|percent|billions).*\)", ctx, re.I):
            units = ctx
        elif ctx and not ctx.startswith("|") and not ctx.startswith("---") and len(ctx) > 8:
            if not re.match(r"^\d+$", ctx.strip()) and not ctx.startswith("Source:"):
                table_title = ctx

    for j in range(start, end):
        rt = all_lines[j].strip()
        if rt.startswith("|") and "---" in rt:
            # Use multi-row header merging
            headers = _merge_multi_row_headers(all_lines, j)
            found_sep = True
            separator_idx = j
            continue
        if not found_sep:
            continue
        if rt.startswith("|"):
            raw_cells = rt.split("|")
            cells = (
                [c.strip() for c in raw_cells[1:-1]]
                if len(raw_cells) >= 3
                else [c.strip() for c in rt.split("|") if c.strip()]
            )
            if not headers:
                headers = cells
            elif len(cells) >= 2 and headers:
                row_label = cells[0]
                # Build vertical string
                vert = _row_to_vertical(row_label, headers, cells)
                vertical_data.append(vert)
                # Build dict (backward compat)
                rd = {"_row_label": row_label}
                for ki, vi in enumerate(cells[1:], 1):
                    cn = headers[ki] if ki < len(headers) else f"col_{ki}"
                    if "unnamed" in cn.lower() or not cn:
                        cn = f"col_{ki}"
                    vc = vi.replace(",", "").replace("$", "").strip()
                    try:
                        nvi = (
                            float(re.sub(r"[^\d.\-]", "", vc))
                            if vc and vc not in ("-", "nan", "...", "*")
                            else None
                        )
                    except ValueError:
                        nvi = None
                    rd[cn] = nvi if nvi is not None else vc
                table_data.append(rd)

    return headers, vertical_data, table_data, table_title, units, separator_idx


def _build_result(
    fname,
    line_num,
    all_lines,
    headers,
    vertical_data,
    table_data,
    table_title,
    units,
    match_line_idx=None,
    terms=None,
):
    """Build a result dict with vertical serialization output."""
    result = {
        "file": fname,
        "line": line_num,
    }
    if table_title:
        result["table_title"] = table_title.strip()
    if units:
        result["units"] = units.strip()

    # Matched row in vertical format — prefer rows matching search terms
    # First: search ALL vertical_data for a row matching MOST search terms (not just any one)
    if vertical_data and terms:
        best_vd = None
        best_score = 0
        for vd in vertical_data:
            # Only check the ROW label line (first line), not the values
            first_line = vd.split("\n")[0].lower() if "\n" in vd else vd.lower()
            score = sum(1 for t in terms if t in first_line)
            if score > best_score:
                best_score = score
                best_vd = vd
        if best_vd and best_score >= min(2, len(terms)):
            result["matched_row_vertical"] = best_vd

    # Fallback: use match_line_idx if no term-matching row found
    if not result.get("matched_row_vertical") and match_line_idx is not None and headers:
        match_text = all_lines[match_line_idx].strip()
        if match_text.startswith("|"):
            raw_cells = match_text.split("|")
            cells = (
                [c.strip() for c in raw_cells[1:-1]]
                if len(raw_cells) >= 3
                else [c.strip() for c in match_text.split("|") if c.strip()]
            )
            if len(cells) >= 2:
                result["matched_row_vertical"] = _row_to_vertical(cells[0], headers, cells)

    # Vertical data: prioritize rows matching terms, then fill with context
    if vertical_data:
        # First: rows matching search terms
        matching_rows = []
        other_rows = []
        for vd in vertical_data:
            vd_lower = vd.lower()
            if terms and any(t in vd_lower for t in terms):
                matching_rows.append(vd)
            else:
                other_rows.append(vd)
        # Combine: matching rows first, then fill up to 25 with context
        result["vertical_data"] = matching_rows + other_rows[: max(5, 25 - len(matching_rows))]

    # Keep table_data as backup but secondary
    if table_data:
        result["table_data"] = table_data[:15]

    # Minimal snippet: just 5 lines around the match for units/footnotes context
    if match_line_idx is not None:
        snip_start = max(0, match_line_idx - 3)
        snip_end = min(len(all_lines), match_line_idx + 3)
        result["context"] = "\n".join(all_lines[snip_start:snip_end])

    return result


def search_raw_corpus(keywords, year=None, limit=5, period_hint=None):
    """Search Treasury Bulletin TXT files. Two-stage approach:
    Stage 1: Find files/tables containing ALL search terms (co-occurrence, not same-line)
    Stage 2: Within matches, find specific rows and extract with merged headers + vertical format.

    period_hint: "calendar", "fiscal", or None — used to prefer matching table types.
    """
    corpus = Path(CORPUS_DIR)
    if not corpus.exists():
        return {"error": "Corpus not found", "results": []}

    terms = [t.strip() for t in str(keywords).lower().split() if len(t.strip()) >= 3]
    if not terms:
        return {"error": "No search terms", "results": []}

    # Separate year-like terms from metric terms for two-stage search
    year_terms = [t for t in terms if re.match(r"^\d{4}$", t)]
    metric_terms = [t for t in terms if not re.match(r"^\d{4}$", t)]
    if not metric_terms:
        metric_terms = terms  # fallback: use all terms

    results = []
    seen_keys = set()  # dedup by (file, line)

    # Pass 1: Try keyword index for fast table-level search (co-occurrence within table block)
    # Collect candidates, then rank by bulletin proximity — don't stop at limit
    idx_path = _build_keyword_index_if_needed()
    idx_candidates = []  # (fname, line_num, title, pub_yr, pub_mo)
    if idx_path and os.path.exists(idx_path):
        try:
            with open(idx_path) as f:
                for line in f:
                    line_lower = line.lower()
                    # Co-occurrence: ALL metric terms must appear in the index line
                    metric_matched = sum(1 for t in metric_terms if t in line_lower)
                    if metric_matched < min(2, len(metric_terms)):
                        continue
                    # Year terms: at least one year must appear if specified
                    if year_terms and not any(t in line_lower for t in year_terms):
                        continue

                    parts = line.split("\t")
                    if len(parts) < 4:
                        continue
                    file_line = parts[0]
                    title = parts[1]
                    fl_parts = file_line.split(":")
                    fname = fl_parts[0]
                    line_num = int(fl_parts[1]) if len(fl_parts) > 1 else 0

                    # Year filter: check filename year (PUB tag)
                    pub_match = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", fname)
                    pub_yr = int(pub_match.group(1)) if pub_match else 0
                    pub_mo = int(pub_match.group(2)) if pub_match else 0
                    if year:
                        yr = int(year)
                        if pub_yr and not (yr <= pub_yr <= yr + 8):
                            continue

                    dedup_key = (fname, line_num)
                    if dedup_key in seen_keys:
                        continue
                    seen_keys.add(dedup_key)

                    # Extract index metadata for ranking
                    idx_meta = parts[3] if len(parts) > 3 else ""
                    has_monthly = "HAS_12_MONTHS" in idx_meta
                    has_annual = "HAS_ANNUAL" in idx_meta
                    basis_calendar = "BASIS:calendar" in idx_meta
                    basis_monthly = "BASIS:monthly" in idx_meta
                    idx_candidates.append(
                        (
                            fname,
                            line_num,
                            title,
                            pub_yr,
                            pub_mo,
                            has_monthly,
                            has_annual,
                            basis_calendar,
                            basis_monthly,
                        )
                    )
        except Exception:
            pass

    # Rank index candidates: group by file, pick best file first, then take multiple tables from it
    if idx_candidates:
        yr = int(year) if year else 2000
        # Group by file
        by_file = {}
        for c in idx_candidates:
            fname = c[0]
            by_file.setdefault(fname, []).append(c)

        # Rank files by proximity to year+1
        def _file_priority(fname):
            m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", fname)
            if not m:
                return (99, 0)
            pub_yr, pub_mo = int(m.group(1)), int(m.group(2))
            return (abs(pub_yr - (yr + 1)), pub_mo)

        sorted_files = sorted(by_file.keys(), key=_file_priority)

        # Within each file, rank tables by relevance to question
        idx_candidates = []
        for fname in sorted_files:
            file_cands = by_file[fname]
            file_cands.sort(
                key=lambda c: (
                    # Period match: if question wants calendar, prefer calendar/monthly basis
                    2
                    if (period_hint == "calendar" and (c[7] or c[8]))
                    else 2
                    if (period_hint == "fiscal" and not c[7] and not c[8])
                    else 0,
                    # Monthly data is more useful (has individual values to sum)
                    2 if c[8] else 0,  # basis_monthly
                    1 if c[7] else 0,  # basis_calendar
                    # Completeness
                    2 if c[5] else 0,  # has_monthly
                    1 if c[6] else 0,  # has_annual
                ),
                reverse=True,
            )
            idx_candidates.extend(file_cands)

    # Process top index candidates — take up to 4 tables from best file, 2 from others
    idx_file_count = {}
    best_file = idx_candidates[0][0] if idx_candidates else None
    for fname, line_num, title, pub_yr, pub_mo, *_meta in idx_candidates:
        if len(results) >= limit * 2:
            break
        max_for_file = 4 if fname == best_file else 2
        if idx_file_count.get(fname, 0) >= max_for_file:
            continue

        fpath = corpus / fname
        if not fpath.exists():
            continue
        all_lines = fpath.read_text(errors="replace").splitlines()
        tbl_start = max(0, line_num - 25)
        tbl_end = min(len(all_lines), line_num + 80)

        headers, vertical_data, table_data, tbl_title, tbl_units, sep_idx = _parse_table_at(
            all_lines, tbl_start, tbl_end
        )

        if not tbl_title:
            tbl_title = title.strip()

        best_match_idx = None
        for mi in range(tbl_start, tbl_end):
            ml = all_lines[mi].lower()
            if all(t in ml for t in metric_terms[:2]):
                best_match_idx = mi
                break

        result = _build_result(
            fname,
            line_num,
            all_lines,
            headers,
            vertical_data,
            table_data,
            tbl_title,
            tbl_units,
            match_line_idx=best_match_idx,
            terms=metric_terms,
        )
        results.append(result)
        idx_file_count[fname] = idx_file_count.get(fname, 0) + 1

    # Pass 2: Two-stage grep — file-level co-occurrence then line-level proximity
    if len(results) < limit:
        files = sorted(corpus.glob("treasury_bulletin_*.txt"))
        if year:
            yr = int(year)
            year_files = [f for f in files if any(f"_{y}_" in f.name for y in range(yr, yr + 8))]
            if not year_files:
                year_files = [
                    f for f in files if any(f"_{y}_" in f.name for y in range(yr - 1, yr + 15))
                ]
            files = year_files if year_files else files[:20]

        # Sort files so bulletins from year+1 come first (most complete annual data),
        # then year+2, year, year+3, etc.
        if year:
            yr = int(year)

            def _file_priority(f):
                m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", f.name)
                if not m:
                    return 99
                pub_yr, pub_mo = int(m.group(1)), int(m.group(2))
                # Best: year+1 (full-year data), then year+2, then year itself
                # Within same year, prefer earlier months (Jan has annual tables)
                distance = abs(pub_yr - (yr + 1))
                return (distance, pub_mo)

            files.sort(key=_file_priority)

        file_results_count = {}  # track results per file (max 2 per file)
        MAX_PER_FILE = 3
        # Collect more candidates than limit, then rank
        collect_limit = limit * 3

        for fpath in files[:40]:
            if len(results) >= collect_limit:
                break
            if file_results_count.get(fpath.name, 0) >= MAX_PER_FILE:
                continue

            try:
                file_text = fpath.read_text(errors="replace")
                file_lines = file_text.splitlines()
                file_lower = file_text.lower()
            except Exception:
                continue

            # Stage 1: File-level co-occurrence — ALL metric terms must exist somewhere in file
            if not all(t in file_lower for t in metric_terms[:3]):
                continue
            if year_terms and not any(t in file_lower for t in year_terms):
                continue

            # Stage 2: Find lines matching metric terms, check ±30 line proximity for year
            for i, fline in enumerate(file_lines):
                if file_results_count.get(fpath.name, 0) >= MAX_PER_FILE:
                    break
                fline_lower = fline.lower()
                # Line must contain at least one metric term
                if not any(t in fline_lower for t in metric_terms):
                    continue
                # For multi-term queries, require at least the first metric term on this line
                if metric_terms and metric_terms[0] not in fline_lower:
                    continue

                # Check proximity: year term (or year param) must appear within ±30 lines
                proximity_ok = False
                if not year_terms and not year:
                    proximity_ok = True  # No year constraint
                else:
                    check_terms = year_terms if year_terms else [str(year)]
                    ctx_start = max(0, i - 30)
                    ctx_end = min(len(file_lines), i + 30)
                    for ci in range(ctx_start, ctx_end):
                        cl = file_lines[ci].lower()
                        if any(t in cl for t in check_terms):
                            proximity_ok = True
                            break
                if not proximity_ok:
                    continue

                dedup_key = (fpath.name, i)
                if dedup_key in seen_keys:
                    continue
                # Skip if we already have a result from same file within 50 lines
                too_close = False
                for sk in seen_keys:
                    if sk[0] == fpath.name and abs(sk[1] - i) < 30:
                        too_close = True
                        break
                if too_close:
                    continue
                seen_keys.add(dedup_key)

                tbl_start = max(0, i - 25)
                tbl_end = min(len(file_lines), i + 60)

                headers, vertical_data, table_data, tbl_title, tbl_units, sep_idx = _parse_table_at(
                    file_lines, tbl_start, tbl_end
                )

                result = _build_result(
                    fpath.name,
                    i + 1,
                    file_lines,
                    headers,
                    vertical_data,
                    table_data,
                    tbl_title,
                    tbl_units,
                    match_line_idx=i,
                    terms=metric_terms,
                )
                results.append(result)
                file_results_count[fpath.name] = file_results_count.get(fpath.name, 0) + 1

    # Rank results: prefer bulletins closest to year+1, with table data, with matched row
    if year and len(results) > limit:
        yr = int(year)

        def _result_score(r):
            fname = r.get("file", "")
            m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", fname)
            pub_yr = int(m.group(1)) if m else 0
            pub_mo = int(m.group(2)) if m else 0
            # Proximity: year+1 is best (distance=0), year+2 (distance=1), etc.
            proximity = -abs(pub_yr - (yr + 1))
            # Earlier months in the same year preferred (Jan/Feb have annual summaries)
            month_bonus = -pub_mo if pub_yr == yr + 1 else 0
            # Has structured data?
            has_data = 1 if r.get("vertical_data") or r.get("table_data") else 0
            has_match = 1 if r.get("matched_row_vertical") else 0
            return (has_match, has_data, proximity, month_bonus)

        results.sort(key=_result_score, reverse=True)
        results = results[:limit]

    return {"results": results, "count": len(results)}


# == PDF Page Fetcher (for visual questions) ====================================

PDF_GITHUB_BASE = (
    "https://raw.githubusercontent.com/databricks/officeqa/main/treasury_bulletin_pdfs"
)


def fetch_pdf_page(source_file, page=1):
    """Download a single PDF from GitHub and extract a specific page as base64 image.
    For visual questions about charts, figures, page layouts.
    Returns base64 PNG of the page, or error message."""
    import base64

    # Convert source_file to PDF name: treasury_bulletin_1941_01.txt -> treasury_bulletin_1941_01.pdf
    pdf_name = re.sub(r"\.txt$", ".pdf", str(source_file))
    if not pdf_name.endswith(".pdf"):
        pdf_name += ".pdf"

    url = f"{PDF_GITHUB_BASE}/{pdf_name}"
    pdf_path = f"/tmp/{pdf_name}"

    # Download if not cached
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
        print(f"  Downloading PDF: {url}", file=sys.stderr)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(pdf_path, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            print(f"  Downloaded: {os.path.getsize(pdf_path) // 1024}KB", file=sys.stderr)
        except Exception as e:
            return {"error": f"Failed to download PDF: {e}"}

    # Try to render page as image using available tools
    page_num = max(1, int(page))
    png_path = f"/tmp/page_{pdf_name}_{page_num}.png"

    # Try pdftoppm (poppler) first
    try:
        result = subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                "-r",
                "200",
                pdf_path,
                f"/tmp/page_{pdf_name}_{page_num}",
            ],
            capture_output=True,
            timeout=30,
        )
        # pdftoppm creates files like page_X-01.png
        import glob

        pngs = sorted(glob.glob(f"/tmp/page_{pdf_name}_{page_num}*.png"))
        if pngs:
            png_path = pngs[0]
            with open(png_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            return {
                "status": "ok",
                "page": page_num,
                "source_file": pdf_name,
                "image_base64": img_b64[:100] + "...(truncated for display)",
                "image_size_kb": os.path.getsize(png_path) // 1024,
                "hint": "Page rendered as image. Describe what you see: charts, tables, values.",
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: try python3 with pymupdf/fitz if available
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        if page_num <= len(doc):
            pg = doc[page_num - 1]
            pix = pg.get_pixmap(dpi=200)
            pix.save(png_path)
            with open(png_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            doc.close()
            return {
                "status": "ok",
                "page": page_num,
                "source_file": pdf_name,
                "image_base64": img_b64[:100] + "...(truncated for display)",
                "image_size_kb": os.path.getsize(png_path) // 1024,
            }
        doc.close()
    except ImportError:
        pass
    except Exception as e:
        return {"error": f"PDF rendering failed: {e}"}

    # Last fallback: just confirm PDF exists and give text extraction hint
    return {
        "status": "pdf_downloaded",
        "page": page_num,
        "source_file": pdf_name,
        "pdf_size_kb": os.path.getsize(pdf_path) // 1024,
        "hint": "PDF downloaded but no renderer available. Use the TXT corpus file instead.",
    }


# == LLM Call ==================================================================


def call_llm(system_prompt, user_prompt, max_tokens=2048):
    """Single focused LLM call -- no tools, no spinning."""
    api_key = API_KEY or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("  call_llm: No API key available", file=sys.stderr)
        return None

    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2**attempt)
    return None


# == Tool-calling loop architecture ============================================
# Instead of a fixed pipeline, we give the LLM the same tools from the MCP
# server and let it decide what to search for in a controlled loop.

MAX_TOOL_CALLS = 10

# System prompt — same as the MCP server's system.j2 (root level)
SYSTEM_PROMPT = """You are a Treasury Data Analyst. Answer questions using 696 U.S. Treasury Bulletin text files (1939-2025).

Your ONLY job: find data -> compute -> submit answer.

WORKFLOW (aim for 3-5 tool calls total):
1. PARSE the question: what metric, what year(s), fiscal vs calendar, what units?
2. SEARCH using search_raw_corpus (primary) — this greps the original source files with zero data loss
3. READ the vertical key-value data returned — your answer is already extracted as "column: value" pairs
4. COMPUTE using compute_expression for any arithmetic
5. SUBMIT using submit_answer — a wrong answer beats no answer

SEARCH STRATEGY (ordered by reliability):

  Step 1 (PRIMARY — start here): search_raw_corpus
    Searches the ORIGINAL Treasury Bulletin TXT files. Returns pre-extracted vertical data:
    each row is formatted as key-value pairs like "ROW: National defense\n  1940 > Jan: 132\n  1940 > Feb: 129".
    - Use 2-3 specific keywords from the question (e.g. "national defense expenditures")
    - Set year to focus on relevant bulletins
    - Read matched_row_vertical FIRST — it has the specific row matching your query with all values labeled
    - If matched_row_vertical doesn't have your answer, scan vertical_data for other rows
    - Headers use ">" for nested levels: "1940 > Jan." means January 1940
    - Check the "units" field for scale (millions, thousands, etc.)
    Example: search_raw_corpus(keywords="national defense expenditures", year=1940)

  Step 2 (DB lookup — fast but may miss data): resolve_numeric_evidence
    Searches a pre-built database. Fast for simple lookups but misses ~50% of data.
    Use AFTER search_raw_corpus if you need a quick cross-check or if grep didn't find it.

  Step 3 (DB fallback): search_canonical or search_ledger
    Keyword search across the database. Use only if Steps 1-2 both failed.

  Step 4 (CPI/inflation): lookup_cpi
    For "real dollars", "constant dollars", "inflation-adjusted" questions.
    lookup_cpi(year=1970, month=3) -> monthly CPI-U value
    lookup_cpi(year=1970) -> annual average
    Formula: real_value = nominal_value × (target_CPI / source_CPI)

  Step 5 (visual/chart): fetch_pdf_page
    For questions about charts, figures, or "on page X".

  Step 6 (compute): compute_expression
    For ALL arithmetic. Available functions: sum, mean, median, stdev, variance,
    correlation, percentile, cagr, geometric_mean, linreg, yoy_growth, theil_index,
    gini, herfindahl, hp_filter, cv, min, max, abs, round, sqrt, log, exp.
    Example: compute_expression(expression="abs(a - b)", variables={"a": 71, "b": 68})

  Step 7 (submit): submit_answer
    Submit your final answer. ALWAYS submit something before running out of calls.

READING SEARCH RESULTS:
When search_raw_corpus returns results, use this reading order:
1. matched_row_vertical — the best-matching row, already formatted as "column_name: value" pairs
   Example: "ROW: National defense\n  1940 > Jan: 132\n  1940 > Feb: 129\n  1940 > Total: 2,602"
   Just find the column matching your year/month and read the value directly.
2. vertical_data — all rows from the table in the same vertical format. Scan these if you need a different row.
3. units — tells you the scale (e.g. "In millions of dollars"). Always check this.
4. table_data — backup dict format with {column_name: value}. Use only if vertical data is unclear.
5. context — a few lines around the match for footnotes. Usually not needed.

ITERATIVE SEARCH:
When you find partial data, BUILD ON IT:
- If you found the right table but wrong rows, note the FILE and LINE, then search for nearby content
- Different sections of the same bulletin have related data — look for "Table 2", "Table 3", etc.
- If you see "1938" in column headers, January-December data is in the rows below

SEARCH TIPS — how to search EFFECTIVELY:
- Use SHORT, DISTINCT keywords: "national defense" not "total expenditures of the U.S federal government for national defense"
- If first search fails, DON'T tweak the same keywords. Try COMPLETELY DIFFERENT terms:
  * Different metric name: "defense spending" -> "war activities" -> "military expenditures"
  * Different table title: instead of the metric, search for the TABLE NAME visible in the bulletin
  * Different bulletin year: data from 1938 appears in bulletins from 1939-1941
- If you found the right table but wrong time period, note the file name and search for more context in that same file
- The data you need is ALWAYS in the corpus. If you can't find it, your keywords are wrong.

CRITICAL RULES:
- Use compute_expression for ALL math — never compute in your head
- NEVER call the same tool with identical arguments twice
- If you have data, COMPUTE and SUBMIT. Do not keep searching for "better" data
- Budget: 10 tool calls max. Aim for 3-5. Past 7 calls: stop and submit immediately
- A wrong answer scores higher than no answer. Always submit something.

FISCAL YEAR RULES:
- Pre-1977: FY runs Jul 1 (Y-1) to Jun 30 (Y). FY1940 = Jul 1939 - Jun 1940.
- Post-1977: FY runs Oct 1 (Y-1) to Sep 30 (Y). FY1980 = Oct 1979 - Sep 1980.
- Calendar year = Jan 1 to Dec 31. CY1940 != FY1940.
- If asked for "calendar year" total and you only find monthly data, SUM the 12 months (Jan-Dec).

COMMON MISTAKES:
- WRONG ROW: Match the EXACT metric name from the question
- UNIT SCALE: Check "In millions" vs "In thousands" before answering
- WRONG YEAR: A 1941 bulletin contains FY1940 data. Match data year, not bulletin year
- DOUBLE COUNTING: "Total" already includes sub-items. Never sum a parent with its children

ANSWER FORMAT: Return just the numeric value. Keep % for percentages."""


# Tool definitions for OpenRouter function-calling API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_raw_corpus",
            "description": "PRIMARY SEARCH TOOL: Grep Treasury Bulletin TXT files. Returns table snippets with structured data. Most reliable — searches original source with zero data loss.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "2-3 specific keywords from the question e.g. 'national defense expenditures'",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year to focus search on (searches bulletins from year to year+2)",
                    },
                    "limit": {"type": "integer", "default": 3},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_expression",
            "description": "Evaluate math expression with variables. Use for ALL arithmetic. Functions: sum, mean, median, stdev, cagr, geometric_mean, correlation, percentile, yoy_growth, theil_index, gini, herfindahl, hp_filter, cv, linreg, abs, round, sqrt, log, exp, min, max.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python expression e.g. 'abs(a - b)'",
                    },
                    "variables": {
                        "type": "object",
                        "description": 'Variable values e.g. {"a": 71, "b": 68}',
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_cpi",
            "description": "Look up BLS CPI-U index (1982-84=100). For inflation/real dollar questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "month": {
                        "type": "integer",
                        "description": "Month 1-12. Omit for annual average.",
                    },
                    "year_start": {"type": "integer"},
                    "year_end": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit final answer. Writes to answer.txt. ALWAYS call this before running out of tool calls.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "Final numeric answer"}},
                "required": ["answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_pdf_page",
            "description": "Download a Treasury Bulletin PDF page. For visual/chart questions only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_file": {
                        "type": "string",
                        "description": "Filename e.g. 'treasury_bulletin_1990_09.txt'",
                    },
                    "page": {
                        "type": "integer",
                        "description": "Page number (1-indexed)",
                        "default": 1,
                    },
                },
                "required": ["source_file", "page"],
            },
        },
    },
]


# Tool dispatch table — maps tool names to local functions
def _dispatch_resolve_numeric_evidence(conn, args):
    return resolve_numeric_evidence(
        conn,
        question=args.get("question", ""),
        metric=args.get("metric", ""),
        year=args.get("year"),
        period_basis=args.get("period_basis", ""),
    )


def _dispatch_search_canonical(conn, args):
    return search_canonical(
        conn,
        query=args.get("query", ""),
        year=args.get("year"),
        years=args.get("years"),
        table_family=args.get("table_family", ""),
        limit=args.get("limit", 15),
    )


def _dispatch_search_ledger(conn, args):
    return search_ledger(
        conn,
        metric=args.get("metric", ""),
        year=args.get("year"),
        period_basis=args.get("period_basis", ""),
        years=args.get("years"),
    )


def _dispatch_get_time_series(conn, args):
    return get_time_series(
        conn,
        metric=args.get("metric", ""),
        year_start=args.get("year_start", 1900),
        year_end=args.get("year_end", 2000),
        period_basis=args.get("period_basis", "calendar"),
    )


def _dispatch_compute_expression(conn, args):
    return {"result": safe_eval_finance(args["expression"], args.get("variables", {}))}


def _dispatch_submit_answer(conn, args):
    answer = str(args.get("answer", "")).strip()
    try:
        Path(ANSWER_PATH).write_text(answer)
    except Exception as e:
        print(f"  Write failed: {e}, trying fallback", file=sys.stderr)
        try:
            subprocess.run(
                ["sh", "-c", f"printf '%s' '{answer}' > {ANSWER_PATH}"],
                check=True,
                timeout=5,
            )
        except Exception:
            pass
    return {"status": "TASK COMPLETE", "answer": answer}


def _dispatch_lookup_cpi(conn, args):
    if "year_start" in args and "year_end" in args:
        return lookup_cpi_range(args["year_start"], args["year_end"], args.get("month"))
    return lookup_cpi(args.get("year", 2000), args.get("month"))


def _dispatch_search_raw_corpus(conn, args):
    return search_raw_corpus(
        keywords=args.get("keywords", ""),
        year=args.get("year"),
        limit=args.get("limit", 3),
    )


def _dispatch_fetch_pdf_page(conn, args):
    return fetch_pdf_page(
        source_file=args.get("source_file", ""),
        page=args.get("page", 1),
    )


TOOL_DISPATCH = {
    "resolve_numeric_evidence": _dispatch_resolve_numeric_evidence,
    "search_canonical": _dispatch_search_canonical,
    "search_ledger": _dispatch_search_ledger,
    "get_time_series": _dispatch_get_time_series,
    "compute_expression": _dispatch_compute_expression,
    "lookup_cpi": _dispatch_lookup_cpi,
    "search_raw_corpus": _dispatch_search_raw_corpus,
    "fetch_pdf_page": _dispatch_fetch_pdf_page,
    "submit_answer": _dispatch_submit_answer,
}


# == LLM Call with Tools =======================================================


def call_llm_with_tools(messages, tools, max_tokens=2048):
    """Call LLM via OpenRouter with function-calling support.

    Returns the full response message dict (may contain tool_calls).
    """
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2**attempt)
    return None


# == Answer Verification =======================================================


def verify_answer(answer: str, question: str, plan: dict, evidence: str, conn=None) -> dict:
    """Pre-write verification. Checks unit scale, value provenance, and common pitfalls.

    Ported from server/tools.py verify_answer. Works as a pure function;
    DB-dependent checks run only when *conn* is provided.

    Returns dict with keys: is_consistent, severity, checks, warnings.
    """
    warnings_list: list[str] = []
    checks: dict[str, Any] = {}
    q = str(question or "").strip().lower()
    ans = str(answer or "").strip()

    if not ans:
        return {
            "is_consistent": False,
            "severity": "high",
            "checks": {},
            "warnings": [
                {"type": "general", "severity": "high", "message": "No candidate answer provided."}
            ],
        }

    # -- Parse numeric answer (single value or list) --
    ans_num = None
    ans_is_list = bool(ans.startswith("[") and ans.endswith("]") and "," in ans)
    ans_list_nums: list[float] = []

    if ans_is_list:
        inner = ans[1:-1]
        for elem in inner.split(","):
            elem = elem.strip()
            try:
                ans_list_nums.append(float(re.sub(r"[,$%]", "", elem)))
            except (ValueError, TypeError):
                pass
        checks["is_list_answer"] = True
        checks["list_element_count"] = len(ans_list_nums)

        year_range = re.search(r"(?:from|between)\s+(\d{4})\s+(?:to|and|through)\s+(\d{4})", q)
        if year_range:
            start_yr, end_yr = int(year_range.group(1)), int(year_range.group(2))
            expected = end_yr - start_yr + 1
            checks["expected_element_count"] = expected
            if len(ans_list_nums) != expected:
                warnings_list.append(
                    f"LIST COUNT: answer has {len(ans_list_nums)} elements "
                    f"but year range {start_yr}-{end_yr} implies {expected}."
                )

        bad_elems = [v for v in ans_list_nums if not (-1e18 < v < 1e18)]
        if bad_elems:
            warnings_list.append(f"LIST: some elements look invalid: {bad_elems[:3]}")
    else:
        try:
            ans_clean = re.sub(r"[,$%]", "", ans)
            ans_num = float(ans_clean)
        except (ValueError, TypeError):
            pass

    if ans_num is not None and ans_num == 0:
        warnings_list.append("Answer is 0 -- verify this is actually correct, not a default.")

    if "fiscal year" in q and "calendar" in q:
        warnings_list.append(
            "Question mentions both fiscal and calendar year -- verify which period you used."
        )
    if "end of" in q and "average" not in q and "mean" not in q:
        checks["period_type"] = "end-of-period (not average)"

    if evidence:
        has_preliminary = "preliminary" in evidence.lower() or "(p)" in evidence.lower()
        if has_preliminary and ("revised" in q or "final" in q):
            warnings_list.append(
                "REVISION: Evidence contains preliminary (p) values but "
                "question asks for revised/final data."
            )

    if evidence and ans_num is not None:
        ev_nums: list[float] = []
        for tok in re.findall(r"[\-\d,]+\.?\d*", evidence):
            try:
                ev_nums.append(float(re.sub(r"[,$]", "", tok)))
            except (ValueError, TypeError):
                pass
        if ev_nums:
            exact_match = any(abs(ans_num - ev) < 0.01 for ev in ev_nums)
            checks["value_in_evidence"] = exact_match
            for ev_n in ev_nums:
                if ev_n != 0:
                    ratio = abs(ans_num / ev_n)
                    if ratio > 1000 or (ratio > 0 and ratio < 0.001):
                        warnings_list.append(
                            f"MAGNITUDE: answer={ans} vs evidence={ev_n} -- "
                            f"ratio={ratio:.1f}. Likely unit scaling error."
                        )
                        break

    q_asks_total = any(w in q for w in ["total", "aggregate", "sum of all", "grand total"])
    q_asks_specific = not q_asks_total and any(
        w in q
        for w in [
            "customs",
            "individual income",
            "corporation income",
            "estate",
            "gift tax",
            "excise",
            "employment",
            "interest",
            "principal",
        ]
    )
    if evidence and q_asks_specific and "total" in evidence.lower():
        warnings_list.append(
            "ROW HIERARCHY: Question asks for a specific sub-item, but evidence "
            "may include a 'Total' row. Verify you selected the correct row."
        )

    scope_keywords = {
        "within and outside": ["within", "outside"],
        "domestic and foreign": ["domestic", "foreign"],
    }
    if evidence:
        for scope_phrase, required_parts in scope_keywords.items():
            if scope_phrase in q and required_parts:
                has_all = all(part in evidence.lower() for part in required_parts)
                has_partial = (
                    any(part in evidence.lower() for part in required_parts) and not has_all
                )
                if has_partial:
                    warnings_list.append(
                        f"SCOPE MISMATCH: Question asks for '{scope_phrase}' but "
                        f"evidence appears to cover only a partial scope."
                    )

    if conn is not None:
        table_pks: list[int] = []
        if plan and plan.get("table_pks"):
            table_pks = [int(pk) for pk in plan["table_pks"]]

        for pk in table_pks[:3]:
            try:
                row = conn.execute(
                    "SELECT units_line, table_title FROM table_index WHERE table_pk = ?", (int(pk),)
                ).fetchone()
                if row and row["units_line"]:
                    ul = str(row["units_line"]).strip()
                    ul_lower = ul.lower()
                    table_scale = 1
                    scale_name = "units"
                    if "thousand" in ul_lower:
                        table_scale = 1_000
                        scale_name = "thousands"
                    elif "billion" in ul_lower:
                        table_scale = 1_000_000_000
                        scale_name = "billions"
                    elif "million" in ul_lower:
                        table_scale = 1_000_000
                        scale_name = "millions"

                    if table_scale > 1:
                        checks[f"unit_scale_pk{pk}"] = f"Table values are in {scale_name} ({ul})"
                        q_wants_nominal = any(
                            w in q for w in ["nominal", "actual dollar", "in dollar"]
                        )
                        if q_wants_nominal:
                            warnings_list.append(
                                f"Table pk={pk} values are in {scale_name}, "
                                f"but question may ask for nominal dollars. "
                                f"Multiply by {table_scale:,} before answering."
                            )
                            if ans_num is not None and ans_num < table_scale:
                                warnings_list.append(
                                    f"LIKELY UNIT ERROR: answer={ans} looks unscaled. "
                                    f"Expected answer ~{ans_num * table_scale:,.0f} "
                                    f"if in nominal dollars."
                                )
                                checks["suggested_fix"] = (
                                    f"multiply by {table_scale} -> {ans_num * table_scale:,.0f}"
                                )

                        if "thousand" in ul_lower and ans_num is not None and ans_num < 1_000_000:
                            warnings_list.append(
                                f"LIKELY UNIT ERROR: Table pk={pk} reports values in "
                                f"thousands, but your answer ({ans}) is < 1,000,000. "
                                f"Did you forget to multiply by 1,000? "
                                f"Suggested fix: multiply answer by 1,000 -> "
                                f"{ans_num * 1000:,.0f}"
                            )
                            checks["suggested_fix"] = f"multiply by 1000 -> {ans_num * 1000:,.0f}"
            except Exception:
                pass

        q_wants_fiscal = "fiscal" in q and "calendar" not in q
        q_wants_calendar = "calendar" in q and "fiscal" not in q
        for pk in table_pks[:3]:
            try:
                _pb = conn.execute(
                    "SELECT period_basis FROM table_index WHERE table_pk = ?", (int(pk),)
                ).fetchone()
                if _pb and _pb["period_basis"]:
                    pb = _pb["period_basis"]
                    if q_wants_calendar and pb == "fiscal":
                        warnings_list.append(
                            f"PERIOD MISMATCH: Question asks for calendar year "
                            f"but table pk={pk} uses fiscal year data."
                        )
                    elif q_wants_fiscal and pb == "calendar":
                        warnings_list.append(
                            f"PERIOD MISMATCH: Question asks for fiscal year "
                            f"but table pk={pk} uses calendar year data."
                        )
            except Exception:
                pass

        if len(table_pks) > 1:
            units_seen: dict[str, list[int]] = {}
            for pk in table_pks[:5]:
                try:
                    row = conn.execute(
                        "SELECT units_line FROM table_index WHERE table_pk = ?", (int(pk),)
                    ).fetchone()
                    if row and row["units_line"]:
                        ul = row["units_line"].strip().lower()
                        units_seen.setdefault(ul, []).append(pk)
                except Exception:
                    pass
            if len(units_seen) > 1:
                warnings_list.append(
                    f"CROSS-TABLE UNITS: Evidence tables use different units: "
                    f"{list(units_seen.keys())}. Normalize before computing."
                )

    fixed_answer = ans
    if checks.get("suggested_fix") and ans_num is not None:
        fix_str = checks["suggested_fix"]
        m = re.search(r"multiply by (\d+)", fix_str)
        if m:
            multiplier = int(m.group(1))
            scaled = ans_num * multiplier
            if scaled == int(scaled):
                fixed_answer = f"{int(scaled):,}"
            else:
                fixed_answer = f"{scaled:,.2f}"
            checks["auto_fixed_answer"] = fixed_answer

    structured_warnings = []
    max_severity = "none"
    for w in warnings_list:
        sev = "high" if any(kw in w for kw in ["MISMATCH", "ERROR", "MAGNITUDE"]) else "low"
        if sev == "high":
            max_severity = "high"
        elif max_severity == "none":
            max_severity = "low"

        w_type = "general"
        if "UNIT" in w or "SCALE" in w or "MAGNITUDE" in w:
            w_type = "unit_mismatch"
        elif "PERIOD" in w or "DATE" in w:
            w_type = "period_mismatch"
        elif "HIERARCHY" in w or "SCOPE" in w:
            w_type = "scope_mismatch"

        structured_warnings.append({"type": w_type, "severity": sev, "message": w})

    return {
        "is_consistent": len(warnings_list) == 0,
        "severity": max_severity,
        "checks": checks,
        "warnings": structured_warnings,
    }


# == Fallback: extract answer from conversation history ========================


def _extract_answer_from_history(messages):
    """Last-resort: scan tool results in conversation for a plausible answer."""
    best_answer = None

    # Walk backwards through messages looking for tool results with values
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue

        # Check for recommended_value (from resolve_numeric_evidence)
        rec = data.get("recommended_value")
        if rec is not None:
            best_answer = str(rec)
            break

        # Check for compute_expression result
        result = data.get("result")
        if result is not None:
            best_answer = str(result)
            break

        # Check for verdict (from extract_values via search_canonical fallback)
        verdict = data.get("verdict", {})
        if verdict and verdict.get("value") is not None:
            best_answer = str(verdict["value"])
            break

        # Check for matches (from search_ledger)
        matches = data.get("matches", [])
        if matches and matches[0].get("value") is not None:
            best_answer = str(matches[0]["value"])
            break

        # Check for best_candidates (from resolve_numeric_evidence)
        candidates = data.get("best_candidates", [])
        if candidates and candidates[0].get("value") is not None:
            best_answer = str(candidates[0]["value"])
            break

    return best_answer


# == Deterministic Question Parser ==============================================


def parse_question(question):
    """Extract search parameters from a question deterministically.
    Returns dict with keys: metric, years, period_basis, search_strategies.
    """
    q = question.strip()

    # Extract years
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", q)]
    # Deduplicate preserving order
    seen = set()
    unique_years = []
    for y in years:
        if y not in seen:
            seen.add(y)
            unique_years.append(y)
    years = unique_years

    # Period basis
    period_basis = "calendar"
    if re.search(r"fiscal\s+year|FY\s*\d{4}|\bfiscal\b", q, re.I):
        period_basis = "fiscal"

    # Extract metric: strip boilerplate, keep the core subject
    metric = q
    # Remove question starters
    metric = re.sub(
        r"^(?:Using\s+specifically\s+only\s+the\s+reported\s+values\s+for\s+all\s+individual\s+calendar\s+months\s+in\s+\d{4}(?:\s+and\s+all\s+individual\s+calendar\s+months\s+in\s+\d{4})?,?\s*)",
        "",
        metric,
        flags=re.I,
    )
    metric = re.sub(
        r"^(?:According\s+to\s+the\s+US\s+Treasury\'?s?\s+breakdown\s+of\s+)",
        "",
        metric,
        flags=re.I,
    )
    metric = re.sub(
        r"^(?:What\s+(?:were?|is|are|was)\s+the\s+(?:total\s+)?)", "", metric, flags=re.I
    )
    metric = re.sub(
        r"^(?:Determine\s+the\s+|How\s+much\s+(?:does|did)\s+the?\s*)", "", metric, flags=re.I
    )
    # Remove parenthetical units
    metric = re.sub(
        r"\(in\s+(?:millions|thousands|billions)\s+of\s+(?:nominal\s+)?dollars?\)",
        "",
        metric,
        flags=re.I,
    )
    # Remove trailing year references
    metric = re.sub(
        r"\s+(?:in|for|during|of)\s+(?:the\s+)?(?:calendar|fiscal)\s+year\s*(?:of\s*)?\d{4}\??.*$",
        "",
        metric,
        flags=re.I,
    )
    metric = re.sub(r"\s+(?:in|for|during)\s+(?:FY|CY)\s*\d{4}.*$", "", metric, flags=re.I)
    # Remove common suffixes
    metric = re.sub(r"\s*\?\s*$", "", metric)
    metric = re.sub(r"(?:\s+This\s+figure\s+should\s+.*)$", "", metric, flags=re.I)
    metric = re.sub(r"(?:\s+Report\s+the\s+.*)$", "", metric, flags=re.I)
    metric = re.sub(r"(?:\s+Calculations\s+should\s+.*)$", "", metric, flags=re.I)
    metric = re.sub(r"(?:\s+Enter\s+the\s+.*)$", "", metric, flags=re.I)
    metric = re.sub(r"(?:\s+Return\s+your\s+.*)$", "", metric, flags=re.I)
    metric = re.sub(r"(?:\s+State\s+the\s+.*)$", "", metric, flags=re.I)
    # Clean up
    metric = re.sub(
        r"\s+of\s+the\s+U\.?S\.?\s*(?:federal\s+)?(?:government\s*)?", " ", metric, flags=re.I
    )
    metric = re.sub(r"\s+", " ", metric).strip(" ?,.")

    # Build search strategies: different keyword combinations
    words = [w for w in re.sub(r"[^\w\s\-]", " ", metric.lower()).split() if len(w) >= 3]
    # Remove stopwords
    stops = {
        "the",
        "and",
        "for",
        "was",
        "what",
        "how",
        "total",
        "value",
        "amount",
        "from",
        "with",
        "that",
        "this",
        "are",
        "were",
        "all",
        "just",
        "only",
        "these",
        "those",
        "each",
        "using",
        "which",
        "between",
        "reported",
        "specifically",
        "individual",
        "corresponding",
    }
    content_words = [w for w in words if w not in stops]

    strategies = []
    # Strategy 1: longest distinct phrase (up to 4 words)
    if len(content_words) >= 2:
        strategies.append(" ".join(content_words[:4]))
    # Strategy 2: first 2 content words
    if len(content_words) >= 2:
        strategies.append(" ".join(content_words[:2]))
    # Strategy 3: just the first content word (broadest)
    if content_words:
        strategies.append(content_words[0])
    # Strategy 4: the full cleaned metric (may be long but sometimes works)
    if metric and metric.lower() not in [s for s in strategies]:
        strategies.append(metric[:80])

    # Deduplicate
    seen_strats = set()
    unique_strats = []
    for s in strategies:
        s_lower = s.lower().strip()
        if s_lower and s_lower not in seen_strats:
            seen_strats.add(s_lower)
            unique_strats.append(s)

    return {
        "metric": metric,
        "years": years,
        "year": years[0] if years else None,
        "period_basis": period_basis,
        "strategies": unique_strats,
        "content_words": content_words,
    }


def deterministic_search(question, limit=5):
    """Run search_raw_corpus with multiple keyword strategies, merge and rank results.
    Returns the best results across all strategies."""
    params = parse_question(question)
    year = params["year"]
    period = params.get("period_basis")  # "calendar" or "fiscal"
    all_results = []
    seen_file_lines = set()

    for strategy in params["strategies"]:
        # Try with year and period hint
        if year:
            result = search_raw_corpus(
                keywords=strategy, year=year, limit=limit, period_hint=period
            )
            for r in result.get("results", []):
                key = (r.get("file"), r.get("line"))
                if key not in seen_file_lines:
                    seen_file_lines.add(key)
                    r["_strategy"] = strategy
                    all_results.append(r)

        # For multi-year questions, search each year
        if len(params["years"]) > 1:
            for yr in params["years"][1:]:
                result = search_raw_corpus(keywords=strategy, year=yr, limit=3, period_hint=period)
                for r in result.get("results", []):
                    key = (r.get("file"), r.get("line"))
                    if key not in seen_file_lines:
                        seen_file_lines.add(key)
                        r["_strategy"] = strategy
                        all_results.append(r)

        if len(all_results) >= limit * 3:
            break

    # Rank: prefer results with matched_row_vertical, table_data, and year+1 bulletins
    def _score(r):
        has_match = 2 if r.get("matched_row_vertical") else 0
        has_data = 1 if r.get("vertical_data") or r.get("table_data") else 0
        fname = r.get("file", "")
        m = re.search(r"treasury_bulletin_(\d{4})", fname)
        pub_yr = int(m.group(1)) if m else 0
        proximity = -abs(pub_yr - ((year or 2000) + 1)) if year else 0
        return (has_match, has_data, proximity)

    all_results.sort(key=_score, reverse=True)
    return all_results[:limit], params


def format_evidence_for_llm(results, question, params):
    """Format search results into a clean evidence block for a single LLM call.

    Prioritizes monthly data for calendar year questions, filters out fiscal-only tables.
    """
    period = params.get("period_basis", "calendar")
    year = params.get("year")

    parts = []
    parts.append(f"QUESTION: {question}")
    parts.append(f"SEARCH PARAMS: metric='{params['metric']}', year={year}, period={period}")

    # If calendar year question AND we find monthly data, tell the LLM explicitly
    if period == "calendar" and year:
        parts.append(f"\nIMPORTANT: This question asks for CALENDAR YEAR {year} (Jan-Dec {year}).")
        parts.append(
            f"If you find monthly values (Jan, Feb, Mar...), you MUST sum all 12 months for Jan-Dec {year}."
        )
        parts.append(
            "Do NOT use a 'fiscal year' or 'FY' total — those cover a different time period."
        )
    parts.append("")

    # Show matched_row_vertical from ALL results that have it, prioritizing monthly data
    shown = 0
    for i, r in enumerate(results[:5]):
        mv = r.get("matched_row_vertical", "")
        vd = r.get("vertical_data", [])

        # Skip results with no useful data
        if not mv and not vd and not r.get("table_data"):
            continue

        shown += 1
        parts.append(f"--- EVIDENCE {shown}: {r.get('file', '?')}:{r.get('line', '?')} ---")
        if r.get("table_title"):
            parts.append(f"TABLE: {r['table_title']}")
        if r.get("units"):
            parts.append(f"UNITS: {r['units']}")

        # For calendar year questions, filter matched_row_vertical to only show monthly values
        if mv and period == "calendar" and year:
            _month_pats = [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
            _yr_str = str(year)
            mv_lines = mv.split("\n")
            filtered_mv_lines = [mv_lines[0]]  # keep ROW: label
            monthly_values = []
            for line in mv_lines[1:]:
                line_lower = line.lower().strip()
                # Keep lines with month names for the target year
                has_month = any(m in line_lower for m in _month_pats)
                # Skip "Total", "FY", annual rows — they confuse the LLM
                is_total = any(w in line_lower for w in ["total", "fiscal", " fy"])
                if has_month:
                    filtered_mv_lines.append(line)
                    # Extract the value for pre-computation
                    val_match = re.search(r":\s*([\d,]+\.?\d*)", line)
                    if val_match:
                        monthly_values.append(val_match.group(1))
                elif not is_total:
                    filtered_mv_lines.append(line)
            mv = "\n".join(filtered_mv_lines)
            if len(monthly_values) >= 6:
                parts.append(
                    f"\nPRE-EXTRACTED MONTHLY VALUES for CY {year}: [{', '.join(monthly_values)}]"
                )
                parts.append(
                    f"(Count: {len(monthly_values)} values — sum these for the calendar year total)"
                )

        # Show matched row
        if mv:
            parts.append(f"\n{mv}")

        # Show additional vertical data rows (other rows from same table)
        if isinstance(vd, list) and vd:
            # For calendar year questions, prioritize rows with monthly labels
            _month_names = [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ]
            month_rows = [v for v in vd if v != mv and any(m in v.lower() for m in _month_names)]
            other_rows = [v for v in vd if v != mv and v not in month_rows]

            if period == "calendar":
                # Filter out total/FY rows that could confuse the LLM
                other_rows = [
                    v
                    for v in other_rows
                    if not any(w in v.lower() for w in ["total", "fiscal year", " fy "])
                ]
                extra_rows = month_rows[:10] + other_rows[:3]
            else:
                extra_rows = (other_rows + month_rows)[:8]
            if extra_rows:
                parts.append("\n--- Other rows in this table ---")
                for er in extra_rows:
                    parts.append(er)

        parts.append("")
        if shown >= 3:
            break

    return "\n".join(parts)


DETERMINISTIC_SYSTEM_PROMPT = """You are a data extraction specialist. You receive pre-searched Treasury Bulletin data and a question. Your job: identify the correct values in the data and output them in a structured format.

STEP 1 — IDENTIFY: Fill out this extraction table:
TABLE_TITLE: (which table has the answer)
ROW_LABEL: (which row matches the question's metric)
UNITS: (millions, thousands, billions, percent — from the Units field)
PERIOD: (calendar year, fiscal year, or monthly)

STEP 2 — EXTRACT VALUES: List the raw numeric values you found.
- If the answer is a single value: VALUES: [2602]
- If you need to sum monthly values: VALUES: [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]
- If you need two values for a calculation: VALUES: [44463, 2602]

STEP 3 — SPECIFY OPERATION:
OPERATION: direct (just return the value)
OPERATION: sum (add all values)
OPERATION: difference (subtract second from first)
OPERATION: percent_change (((new - old) / old) * 100)
OPERATION: ratio (first / second)
OPERATION: geometric_mean
OPERATION: custom — EXPRESSION: <math expression using the values>

STEP 4 — ANSWER: The final numeric answer.

FISCAL YEAR RULES:
- Pre-1977: FY = Jul 1 (Y-1) to Jun 30 (Y). FY1940 = Jul 1939–Jun 1940.
- Post-1977: FY = Oct 1 (Y-1) to Sep 30 (Y).
- Calendar year = Jan 1 to Dec 31. If asked for CY total with monthly data, sum Jan–Dec.

CRITICAL: If the data shows monthly values (Jan, Feb, Mar...) and the question asks for a calendar year total, you MUST list ALL 12 monthly values in VALUES and set OPERATION: sum. Do NOT use an annual/fiscal total row.

Output format — use EXACTLY this structure:
TABLE_TITLE: ...
ROW_LABEL: ...
UNITS: ...
VALUES: [...]
OPERATION: ...
ANSWER: ..."""


def deterministic_solve(question):
    """Deterministic pipeline: parse → search → format → ONE LLM call → answer."""
    start_time = time.time()

    # Step 1: Parse question
    params = parse_question(question)
    print(
        f"  Parsed: metric='{params['metric'][:60]}', year={params.get('year')}, "
        f"strategies={params['strategies'][:3]}",
        file=sys.stderr,
    )

    # Step 2: Search with multiple strategies
    results, params = deterministic_search(question, limit=5)
    print(f"  Found {len(results)} search results", file=sys.stderr)

    if not results:
        print("  WARNING: No search results found", file=sys.stderr)
        return None

    # Step 3: Format evidence
    evidence = format_evidence_for_llm(results, question, params)
    print(f"  Evidence: {len(evidence)} chars", file=sys.stderr)

    # Step 4: ONE LLM call
    llm_response = call_llm(DETERMINISTIC_SYSTEM_PROMPT, evidence, max_tokens=4096)
    if not llm_response:
        print("  LLM call failed", file=sys.stderr)
        return None

    print(f"  LLM response: {llm_response[:200]}", file=sys.stderr)

    # Step 5: Parse structured output — extract VALUES and OPERATION, compute in Python
    answer = None
    values = None
    operation = None
    custom_expr = ""

    for line in llm_response.strip().split("\n"):
        line = line.strip()
        # Extract VALUES: [...]
        vm = re.match(r"^VALUES:\s*\[(.+)\]", line, re.I)
        if vm:
            try:
                raw = vm.group(1)
                # Parse numbers from the list
                values = [
                    float(v.strip().replace(",", "").replace("%", ""))
                    for v in raw.split(",")
                    if v.strip()
                ]
            except (ValueError, TypeError):
                pass
        # Extract OPERATION
        om = re.match(r"^OPERATION:\s*(\S+)", line, re.I)
        if om:
            operation = om.group(1).lower()
        # Extract EXPRESSION for custom operations
        em = re.match(r"^.*EXPRESSION:\s*(.+)", line, re.I)
        if em:
            custom_expr = em.group(1).strip()
        # Extract ANSWER as fallback
        am = re.match(r"^ANSWER:\s*(.+)", line, re.I)
        if am:
            answer = am.group(1).strip()

    # If we have VALUES + OPERATION, compute in Python (more reliable than LLM math)
    if values and operation:
        try:
            if operation == "direct" and len(values) == 1:
                computed = values[0]
            elif operation == "sum":
                computed = sum(values)
            elif operation == "difference" and len(values) >= 2:
                computed = values[0] - values[1]
            elif operation == "percent_change" and len(values) >= 2:
                computed = ((values[1] - values[0]) / abs(values[0])) * 100
            elif operation == "ratio" and len(values) >= 2:
                computed = values[0] / values[1] if values[1] != 0 else None
            elif operation == "geometric_mean" and values:
                import math

                computed = math.prod(values) ** (1.0 / len(values))
            elif operation == "custom":
                # Use safe_eval_finance for custom expressions
                try:
                    computed = safe_eval_finance(custom_expr, {"v": values})
                except Exception:
                    computed = None
            else:
                computed = None

            if computed is not None:
                # Format the answer
                if isinstance(computed, float):
                    if computed == int(computed) and abs(computed) >= 1:
                        answer = f"{int(computed):,}"
                    else:
                        answer = f"{computed}"
                else:
                    answer = str(computed)
                print(
                    f"  Python computed: {operation}({values[:5]}...) = {answer}", file=sys.stderr
                )
        except Exception as e:
            print(f"  Computation error: {e}", file=sys.stderr)

    # Fallback: use LLM's ANSWER line directly
    if not answer:
        for line in reversed(llm_response.strip().split("\n")):
            m = re.match(r"^\s*ANSWER:\s*(.+)", line, re.I)
            if m:
                answer = m.group(1).strip()
                break
    # Last fallback: any number in the response
    if not answer:
        for line in reversed(llm_response.strip().split("\n")):
            m = re.search(r"([\-\d,]+\.?\d*%?)", line)
            if m and len(m.group(1)) > 1:
                answer = m.group(1)
                break

    if answer:
        answer = answer.strip().strip("\"'")

    elapsed = time.time() - start_time
    print(f"  Answer: {answer} ({elapsed:.1f}s)", file=sys.stderr)
    return answer


# == Main ======================================================================


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 /installed-agent/solve.py "QUESTION"')
        sys.exit(1)

    question = sys.argv[1]
    start_time = time.time()

    if not API_KEY:
        print("ERROR: No API key", file=sys.stderr)
        sys.exit(1)

    print(f"Question: {question[:150]}", file=sys.stderr)
    _telemetry("start", {"question": question[:200]})

    # === DETERMINISTIC PIPELINE (no tool loop, no DB) ===
    # Python searches → ONE LLM call → Python computes → write answer
    answer = deterministic_solve(question)

    if answer:
        try:
            Path(ANSWER_PATH).write_text(answer)
        except Exception as e:
            print(f"  Write failed: {e}", file=sys.stderr)
        elapsed = time.time() - start_time
        _telemetry("answer", {"answer": answer, "elapsed": elapsed})
        print(f"\nANSWER: {answer} (in {elapsed:.1f}s)", file=sys.stderr)
        print(f"ANSWER: {answer}")
    else:
        elapsed = time.time() - start_time
        _telemetry("no_answer", {"elapsed": elapsed})
        print(f"\nNO ANSWER after {elapsed:.1f}s", file=sys.stderr)
        print("NO ANSWER")
    return

    # == Network probe (first run only) ==
    probe_file = "/tmp/_network_probed"
    if not os.path.exists(probe_file):
        try:
            probes = {}
            # Check what we can reach
            for name, url in [
                ("openrouter", "https://openrouter.ai/api/v1/models"),
                ("google_dns", "https://dns.google/resolve?name=example.com"),
                ("github_raw", "https://raw.githubusercontent.com/robots.txt"),
                ("s3_test", "https://s3.amazonaws.com"),
                ("httpbin", "https://httpbin.org/ip"),
            ]:
                try:
                    req = urllib.request.Request(url)
                    resp = urllib.request.urlopen(req, timeout=5)
                    probes[name] = f"OK ({resp.status})"
                except Exception as e:
                    probes[name] = f"FAIL ({type(e).__name__})"

            # Check filesystem
            probes["corpus"] = "OK" if os.path.isdir("/app/corpus") else "MISSING"
            probes["installed_agent"] = (
                str(os.listdir("/installed-agent/")[:10])
                if os.path.isdir("/installed-agent")
                else "MISSING"
            )
            probes["env_keys"] = [k for k in os.environ if "KEY" in k.upper() or "API" in k.upper()]

            # Check tools available
            import shutil

            for tool in ["curl", "wget", "python3", "zstd", "gzip", "sqlite3"]:
                probes[f"has_{tool}"] = shutil.which(tool) is not None

            _telemetry("network_probe", probes)
            Path(probe_file).write_text("done")
        except Exception as e:
            _telemetry("probe_error", {"error": str(e)})

    # == Setup DB ==
    print("Setting up database...", file=sys.stderr)
    db_path = ensure_db()
    conn = None
    if db_path:
        conn = open_db(db_path)
        _build_runtime_tables(conn)
        print("  DB ready", file=sys.stderr)
    else:
        print("  WARNING: No DB available", file=sys.stderr)

    # == Tool-calling loop ==
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    answer = None
    tool_call_count = 0

    for step in range(MAX_TOOL_CALLS + 2):
        elapsed = time.time() - start_time
        print(
            f"\n--- Step {step + 1} (tool_calls={tool_call_count}, elapsed={elapsed:.1f}s) ---",
            file=sys.stderr,
        )

        # Call LLM with tools
        response_msg = call_llm_with_tools(messages, TOOLS)
        if response_msg is None:
            print("  LLM call failed, breaking", file=sys.stderr)
            break

        # Add assistant message to history
        messages.append(response_msg)

        # Log any text content
        text_content = response_msg.get("content", "")
        if text_content:
            print(f"  LLM: {text_content[:200]}", file=sys.stderr)

        # Check for tool calls
        tool_calls = response_msg.get("tool_calls", [])
        if not tool_calls:
            # No tool calls — model is done or confused
            print("  No tool calls in response", file=sys.stderr)
            # Try to extract answer from text
            if text_content:
                m = re.search(
                    r"(?:answer|result|value)\s*(?:is|=|:)\s*([\-\d,]+\.?\d*%?)",
                    text_content,
                    re.IGNORECASE,
                )
                if m:
                    answer = m.group(1)
                    print(f"  Extracted from text: {answer}", file=sys.stderr)
            break

        # Process each tool call
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                tool_args = {}

            tool_call_id = tc.get("id", f"call_{step}_{tool_name}")
            tool_call_count += 1

            print(
                f"  Tool[{tool_call_count}]: {tool_name}({json.dumps(tool_args, default=str)[:150]})",
                file=sys.stderr,
            )
            _telemetry(
                "tool_call",
                {
                    "step": step,
                    "tool": tool_name,
                    "args": json.dumps(tool_args, default=str)[:200],
                },
            )

            # Dispatch tool call
            handler = TOOL_DISPATCH.get(tool_name)
            if handler is None:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_result = handler(conn, tool_args)
                except Exception as exc:
                    tool_result = {"error": str(exc)}

            # Handle submit_answer specially
            if tool_name == "submit_answer":
                answer = tool_args.get("answer", "")
                result_str = json.dumps(tool_result, default=str)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_str,
                    }
                )
                print(f"  submit_answer: {answer}", file=sys.stderr)

                # Run verify_answer
                print("  Verifying answer...", file=sys.stderr)
                evidence_parts = []
                for msg in messages:
                    if msg.get("role") == "tool":
                        evidence_parts.append(msg.get("content", "")[:500])
                evidence_combined = "\n".join(evidence_parts)

                try:
                    vresult = verify_answer(
                        answer=answer,
                        question=question,
                        plan={},
                        evidence=evidence_combined,
                        conn=conn,
                    )
                    if vresult.get("warnings"):
                        for w in vresult["warnings"]:
                            print(f"  VERIFY [{w['severity']}] {w['message']}", file=sys.stderr)
                    auto_fixed = vresult.get("checks", {}).get("auto_fixed_answer")
                    if auto_fixed and vresult.get("severity") == "high":
                        print(f"  Auto-fixing: {answer} -> {auto_fixed}", file=sys.stderr)
                        answer = auto_fixed
                        # Rewrite answer file with fixed value
                        try:
                            Path(ANSWER_PATH).write_text(answer)
                        except Exception:
                            pass
                except Exception as exc:
                    print(f"  Verification error (non-fatal): {exc}", file=sys.stderr)

                elapsed = time.time() - start_time
                _telemetry("answer", {"answer": answer, "elapsed": elapsed})
                print(f"\nANSWER: {answer} (in {elapsed:.1f}s)", file=sys.stderr)
                print(f"ANSWER: {answer}")
                return

            # Truncate large results to avoid token bloat
            result_str = json.dumps(tool_result, default=str)
            if len(result_str) > 4000:
                result_str = result_str[:4000] + '..."}'

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                }
            )
            print(f"  Result: {result_str[:200]}", file=sys.stderr)

        # Budget enforcement
        if tool_call_count >= MAX_TOOL_CALLS:
            print(f"  Budget exhausted ({tool_call_count} calls), forcing submit", file=sys.stderr)
            # Hard fallback: extract answer now and force-submit
            forced_answer = _extract_answer_from_history(messages)
            if forced_answer:
                print(f"  Force-submitting from history: {forced_answer}", file=sys.stderr)
                answer = str(forced_answer).strip()
                try:
                    Path(ANSWER_PATH).write_text(answer)
                except Exception:
                    pass
                elapsed = time.time() - start_time
                _telemetry("answer_forced", {"answer": answer, "elapsed": elapsed})
                print(f"\nANSWER (forced): {answer} (in {elapsed:.1f}s)", file=sys.stderr)
                print(f"ANSWER: {answer}")
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                return
            # If no answer found in history, give LLM one more chance
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "BUDGET EXHAUSTED. You have used all your tool calls. "
                        "You MUST call submit_answer NOW with your best answer based on "
                        "the data you have collected. Do NOT call any other tool."
                    ),
                }
            )
            continue

    # == Fallback: if loop ended without submit_answer ==
    if answer is None:
        print("  Loop ended without submit_answer, extracting from history...", file=sys.stderr)
        answer = _extract_answer_from_history(messages)

        # Hard fallback: if still no answer, try extracting from LLM text
        if answer is None:
            for msg in reversed(messages):
                text = msg.get("content", "")
                if not text or msg.get("role") == "tool":
                    continue
                # Look for numeric patterns in LLM reasoning
                m = re.search(
                    r"(?:answer|result|total|value|=)\s*(?:is|:)?\s*\$?\s*([\-\d,]+\.?\d*%?)",
                    text,
                    re.IGNORECASE,
                )
                if m:
                    answer = m.group(1).replace(",", "")
                    print(f"  Hard fallback from text: {answer}", file=sys.stderr)
                    break

    if answer:
        # Clean up answer
        answer = str(answer).strip()
        # Remove surrounding quotes if present
        if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in ('"', "'"):
            answer = answer[1:-1]

        try:
            Path(ANSWER_PATH).write_text(answer)
        except Exception as e:
            print(f"  Write failed: {e}", file=sys.stderr)

        elapsed = time.time() - start_time
        _telemetry("answer_fallback", {"answer": answer, "elapsed": elapsed})
        print(f"\nANSWER (fallback): {answer} (in {elapsed:.1f}s)", file=sys.stderr)
        print(f"ANSWER: {answer}")
    else:
        elapsed = time.time() - start_time
        _telemetry("no_answer", {"elapsed": elapsed})
        print(f"\nNO ANSWER after {elapsed:.1f}s", file=sys.stderr)
        print("NO ANSWER")

    if conn:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
