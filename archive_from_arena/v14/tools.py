#!/usr/bin/env python3
"""Local CLI tools for OfficeQA Arena. No MCP, no dependencies.

Usage:
  python3 /installed-agent/tools.py cpi <year>
  python3 /installed-agent/tools.py fy <fiscal_year>
  python3 /installed-agent/tools.py calc "<expression>" [var=value ...]
"""

import ast
import math
import statistics
import sys

# CPI-U Annual Averages (BLS, base 1982-84=100)
CPI = {
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


def _safe_eval(expression, variables=None):
    variables = variables or {}

    def _collect(args):
        r = []
        for a in args:
            if isinstance(a, ast.List):
                r.extend(_ev(e) for e in a.elts)
            else:
                r.append(_ev(a))
        return r

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"unknown variable: {node.id}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return _ev(node.operand)
        if isinstance(node, ast.BinOp):
            L, R = _ev(node.left), _ev(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return L + R
            if isinstance(op, ast.Sub):
                return L - R
            if isinstance(op, ast.Mult):
                return L * R
            if isinstance(op, ast.Div):
                if R == 0:
                    raise ValueError("division by zero")
                return L / R
            if isinstance(op, ast.Pow):
                return L**R
            if isinstance(op, ast.Mod):
                return L % R
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "abs":
                return abs(_ev(node.args[0]))
            if fn == "round":
                a = [_ev(x) for x in node.args]
                return round(a[0], int(a[1])) if len(a) == 2 else round(a[0])
            if fn in ("min", "max"):
                return min(_collect(node.args)) if fn == "min" else max(_collect(node.args))
            if fn == "sqrt":
                return math.sqrt(_ev(node.args[0]))
            if fn == "log":
                a = [_ev(x) for x in node.args]
                return math.log(*a)
            if fn == "exp":
                return math.exp(_ev(node.args[0]))
            if fn == "sum":
                return sum(_collect(node.args))
            if fn == "mean":
                v = _collect(node.args)
                return sum(v) / len(v)
            if fn == "stdev":
                return statistics.stdev(_collect(node.args))
            if fn == "pstdev":
                return statistics.pstdev(_collect(node.args))
            if fn == "median":
                return statistics.median(_collect(node.args))
            if fn == "pct_change":
                a = [_ev(x) for x in node.args]
                if a[0] == 0:
                    raise ValueError("division by zero")
                return (a[1] - a[0]) / a[0] * 100
            if fn == "cagr":
                a = [_ev(x) for x in node.args]
                return ((a[1] / a[0]) ** (1.0 / a[2]) - 1) * 100
            if fn == "variance":
                return statistics.variance(_collect(node.args))
            if fn == "pvariance":
                return statistics.pvariance(_collect(node.args))
            if fn == "geometric_mean":
                return statistics.geometric_mean(_collect(node.args))
            if fn == "harmonic_mean":
                return statistics.harmonic_mean(_collect(node.args))
            if fn == "correlation":
                a = _collect(node.args)
                n = len(a) // 2
                return statistics.correlation(a[:n], a[n:])
            if fn == "linear_regression":
                a = _collect(node.args)
                n = len(a) // 2
                r = statistics.linear_regression(a[:n], a[n:])
                return r.slope
        raise ValueError(f"unsupported: {ast.dump(node)}")

    expr = expression.strip().replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return _ev(tree)


def main():
    if len(sys.argv) < 2:
        print("Usage: tools.py <cpi|fy|calc> [args...]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "cpi":
        yr = int(sys.argv[2])
        val = CPI.get(yr)
        if val is None:
            print(f"CPI not available for {yr}. Range: 1913-2024.")
        else:
            print(f"{val}")
    elif cmd == "fy":
        fy = int(sys.argv[2])
        if fy <= 1976:
            print(f"FY{fy}: {fy - 1}-07-01 to {fy}-06-30 (pre-1977 Jul-Jun)")
        else:
            print(f"FY{fy}: {fy - 1}-10-01 to {fy}-09-30 (post-1976 Oct-Sep)")
    elif cmd == "calc":
        expr = sys.argv[2]
        vs = {}
        for arg in sys.argv[3:]:
            k, v = arg.split("=", 1)
            vs[k] = float(v)
        result = _safe_eval(expr, vs)
        print(f"{result}")
    elif cmd == "ols":
        xs = [float(x) for x in sys.argv[2].split(",")]
        ys = [float(y) for y in sys.argv[3].split(",")]
        r = statistics.linear_regression(xs, ys)
        print(f"slope={r.slope:.6f}")
        print(f"intercept={r.intercept:.6f}")
        if len(sys.argv) > 4:
            px = float(sys.argv[4])
            print(f"predict({px})={r.intercept + r.slope * px:.6f}")
        corr = statistics.correlation(xs, ys)
        print(f"r={corr:.6f}")
        print(f"r2={corr**2:.6f}")
    else:
        print(f"Unknown command: {cmd}. Use: cpi, fy, calc")
        sys.exit(1)


if __name__ == "__main__":
    main()
