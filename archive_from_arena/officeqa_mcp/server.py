#!/usr/bin/env python3
"""OfficeQA tools - callable directly without MCP dependency"""

import difflib
import json
import os
import re
import sys


def clean(v):
    v = re.sub(r"\s*[0-9]+/\s*$", "", v)
    v = re.sub(r"^[rp]/\s*", "", v)
    m = re.match(r"^\(([0-9,.]+)\)$", v)
    if m:
        v = "-" + m.group(1)
    return v.strip()


def to_num(v):
    v = clean(v).replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(v)
    except:
        return None


def parse_table(lines):
    p = sum(l.count("|") for l in lines)
    t = sum(l.count("\t") for l in lines)
    d = "|" if p > t else "\t"
    rows = []
    for n, l in enumerate(lines):
        pts = [c.strip() for c in l.split(d) if c.strip()]
        if len(pts) >= 2:
            rows.append((n, pts))
    if not rows:
        return [], [], ""
    h = rows[0][1]
    data = rows[1:]
    units = ""
    for l in lines[:15]:
        m = re.search(r"(millions?|billions?|thousands?|percent)", l, re.I)
        if m:
            units = m.group(0).lower()
            break
    return h, data, units


def extract_cell(filepath, row_pattern=None, col_pattern=None, fuzzy=False):
    try:
        with open(filepath, errors="replace") as f:
            lines = f.readlines()
        h, data, units = parse_table(lines)
        if not h:
            return {"error": "No table found"}
        result = []
        if units:
            result.append(f"[units: {units}]")
        for n, pts in data:
            label = pts[0]
            if row_pattern:
                if fuzzy:
                    ratio = difflib.SequenceMatcher(
                        None, row_pattern.lower(), label.lower()
                    ).ratio()
                    if ratio < 0.5:
                        continue
                elif row_pattern.lower() not in label.lower():
                    continue
            for j in range(1, len(pts)):
                cn = h[j] if j < len(h) else f"c{j}"
                if col_pattern and col_pattern.lower() not in cn.lower():
                    continue
                raw = pts[j] if j < len(pts) else ""
                if not raw or raw == "nan":
                    continue
                num = to_num(raw)
                if num is not None:
                    result.append(f"{label} | {cn} = {num}")
                else:
                    result.append(f"{label} | {cn} = {clean(raw)}")
        return {"results": result}
    except Exception as e:
        return {"error": str(e)}


def list_rows(filepath):
    try:
        with open(filepath, errors="replace") as f:
            lines = f.readlines()
        h, data, units = parse_table(lines)
        if not h:
            return {"error": "No table found"}
        result = [f"[units: {units}]" if units else ""]
        for n, pts in data:
            result.append(f"[{n + 1}] {pts[0]}")
        return {"rows": result}
    except Exception as e:
        return {"error": str(e)}


def list_cols(filepath):
    try:
        with open(filepath, errors="replace") as f:
            lines = f.readlines()
        h, data, units = parse_table(lines)
        if not h:
            return {"error": "No table found"}
        result = [f"[units: {units}]" if units else ""]
        for j, c in enumerate(h):
            result.append(f"[{j}] {c}")
        return {"columns": result}
    except Exception as e:
        return {"error": str(e)}


def grep_files(query, directory="/app/resources"):
    try:
        result = []
        if not os.path.isdir(directory):
            return {"error": f"Directory not found: {directory}"}
        for fp in sorted(os.listdir(directory)):
            if not fp.endswith(".txt"):
                continue
            full_path = os.path.join(directory, fp)
            try:
                with open(full_path, errors="replace") as f:
                    lines = f.readlines()
                for i, l in enumerate(lines):
                    if re.search(query, l, re.I):
                        ctx = "".join(lines[max(0, i - 1) : i + 2]).rstrip()
                        result.append(f"[{fp}:{i + 1}]\n{ctx}")
            except:
                continue
        return {"matches": result if result else ["No matches"]}
    except Exception as e:
        return {"error": str(e)}


def batch_extract(row_pattern, directory="/app/resources"):
    try:
        result = []
        if not os.path.isdir(directory):
            return {"error": f"Directory not found: {directory}"}
        for fp in sorted(os.listdir(directory)):
            if not fp.endswith(".txt"):
                continue
            full_path = os.path.join(directory, fp)
            try:
                with open(full_path, errors="replace") as f:
                    lines = f.readlines()
                for l in lines:
                    if re.search(row_pattern, l, re.I):
                        vals = re.findall(r"[\d,]+\.?\d*", l)
                        if vals:
                            result.append(f"{fp}: {' | '.join(vals)}")
                        break
            except:
                continue
        return {"extractions": result if result else ["No matches"]}
    except Exception as e:
        return {"error": str(e)}


def get_cpi():
    cpi_data = {
        1913: 9.9,
        1914: 10.0,
        1915: 10.1,
        1916: 10.9,
        1917: 12.8,
        1918: 15.1,
        1919: 17.3,
        1920: 20.0,
        1921: 17.9,
        1922: 16.8,
        1923: 17.1,
        1924: 17.1,
        1925: 17.5,
        1926: 17.7,
        1927: 17.4,
        1928: 17.2,
        1929: 17.2,
        1930: 16.7,
        1931: 15.2,
        1932: 13.6,
        1933: 12.9,
        1934: 13.4,
        1935: 13.7,
        1936: 13.9,
        1937: 14.4,
        1938: 14.1,
        1939: 13.9,
        1940: 14.0,
        1941: 14.7,
        1942: 16.3,
        1943: 17.3,
        1944: 17.6,
        1945: 18.0,
        1946: 19.5,
        1947: 22.3,
        1948: 24.1,
        1949: 23.8,
        1950: 24.1,
        1951: 26.0,
        1952: 26.5,
        1953: 26.7,
        1954: 26.9,
        1955: 26.8,
        1956: 27.2,
        1957: 28.1,
        1958: 28.9,
        1959: 29.1,
        1960: 29.6,
        1961: 29.9,
        1962: 30.2,
        1963: 30.6,
        1964: 31.0,
        1965: 31.5,
        1966: 32.4,
        1967: 33.4,
        1968: 34.8,
        1969: 36.7,
        1970: 38.8,
        1971: 40.5,
        1972: 41.8,
        1973: 44.4,
        1974: 49.3,
        1975: 53.8,
        1976: 56.9,
        1977: 60.6,
        1978: 65.2,
        1979: 72.6,
        1980: 82.4,
        1981: 90.9,
        1982: 96.5,
        1983: 99.6,
        1984: 103.9,
        1985: 107.6,
        1986: 109.6,
        1987: 113.6,
        1988: 118.3,
        1989: 124.0,
        1990: 130.7,
        1991: 136.2,
        1992: 140.3,
        1993: 144.5,
        1994: 148.2,
        1995: 152.4,
        1996: 156.9,
        1997: 160.5,
        1998: 163.0,
        1999: 166.6,
        2000: 172.2,
        2001: 177.1,
        2002: 179.9,
        2003: 184.0,
        2004: 188.9,
        2005: 195.3,
        2006: 201.6,
        2007: 207.3,
        2008: 215.3,
        2009: 214.5,
        2010: 218.1,
        2011: 224.9,
        2012: 229.6,
        2013: 233.0,
        2014: 236.7,
        2015: 237.0,
        2016: 240.0,
        2017: 245.1,
        2018: 251.1,
        2019: 255.7,
        2020: 258.8,
        2021: 271.0,
        2022: 292.7,
        2023: 304.7,
        2024: 314.2,
    }
    return {"cpi": cpi_data, "base": 1982, "formula": "real = nominal * (target_cpi / source_cpi)"}


def main():
    """Minimal MCP stdio interface - no mcp library dependency"""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            msg = json.loads(line)
            method = msg.get("method", "")
            params = msg.get("params", {})

            if method == "extract":
                result = extract_cell(
                    params.get("filepath"),
                    params.get("row"),
                    params.get("col"),
                    params.get("fuzzy", False),
                )
            elif method == "rows":
                result = list_rows(params.get("filepath"))
            elif method == "cols":
                result = list_cols(params.get("filepath"))
            elif method == "grep":
                result = grep_files(params.get("query"), params.get("directory", "/app/resources"))
            elif method == "batch":
                result = batch_extract(
                    params.get("row_pattern"), params.get("directory", "/app/resources")
                )
            elif method == "cpi":
                result = get_cpi()
            else:
                result = {"error": f"Unknown method: {method}"}

            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")


if __name__ == "__main__":
    main()
