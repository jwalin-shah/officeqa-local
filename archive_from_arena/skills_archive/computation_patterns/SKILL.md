# Computation Patterns for compute_expression

These are the common statistical operations asked in Treasury bulletin questions.

Single-series statistics:
  sum([v1, v2, ...])
  mean([v1, v2, ...])
  median([v1, v2, ...])
  stdev([v1, v2, ...])
  geometric_mean([v1, v2, ...])
  cv([v1, v2, ...])                    — coefficient of variation
  theil_index([v1, v2, ...])           — Theil index of dispersion
  percentile([v1, v2, ...], 25)        — Q1 (Tukey exclusive median)
  mad([v1, v2, ...])                   — median absolute deviation

Two-series:
  correlation([x1,x2,...], [y1,y2,...])
  linreg([x1,x2,...], [y1,y2,...])     — returns [slope, intercept]

Growth and change:
  cagr(start_value, end_value, n_years)
  (end - start) / abs(start) * 100     — percent change

Transforms:
  boxcox([v1, v2, ...], lambda)
  interpolate([x1,x2,...], [y1,y2,...], target_x)

H-spread = percentile(values, 75) - percentile(values, 25)
Winsorized range (10%): sort values, trim top/bottom 10%, then max - min
