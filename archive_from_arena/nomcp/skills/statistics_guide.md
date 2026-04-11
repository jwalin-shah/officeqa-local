# Statistical Operations Guide

For complex questions involving statistics. ALL computation via python3.

## Pearson Correlation
```bash
python3 -c "
import statistics
x = [v1, v2, v3, ...]
y = [w1, w2, w3, ...]
n = len(x)
mx, my = sum(x)/n, sum(y)/n
cov = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y)) / n
sx = (sum((xi-mx)**2 for xi in x)/n)**0.5
sy = (sum((yi-my)**2 for yi in y)/n)**0.5
print(round(cov/(sx*sy), 4))
"
```

## Linear Regression (slope, intercept)
```bash
python3 -c "
x = [1,2,3,4,5,6,7,8,9,10,11,12]  # months
y = [v1, v2, ...]  # values
n = len(x)
mx, my = sum(x)/n, sum(y)/n
slope = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y)) / sum((xi-mx)**2 for xi in x)
intercept = my - slope*mx
predicted = slope*13 + intercept  # predict month 13
print(round(predicted, 2))
"
```

## Standard Deviation (population, n denominator)
```bash
python3 -c "
vals = [v1, v2, v3, ...]
n = len(vals)
mean = sum(vals)/n
sd = (sum((v-mean)**2 for v in vals)/n)**0.5
print(round(sd, 4))
"
```
Note: Use /n for POPULATION stdev, /(n-1) for SAMPLE stdev. Check what the question asks.

## Skewness (sample Pearson)
```bash
python3 -c "
vals = [v1, v2, ...]
n = len(vals)
mean = sum(vals)/n
sd = (sum((v-mean)**2 for v in vals)/n)**0.5
skew = sum(((v-mean)/sd)**3 for v in vals) / n
print(round(skew, 4))
"
```

## Excess Kurtosis
```bash
python3 -c "
vals = [v1, v2, ...]
n = len(vals)
mean = sum(vals)/n
sd = (sum((v-mean)**2 for v in vals)/n)**0.5
kurt = sum(((v-mean)/sd)**4 for v in vals) / n - 3
print(round(kurt, 4))
"
```

## Geometric Mean
```bash
python3 -c "
vals = [v1, v2, ...]
product = 1
for v in vals: product *= v
print(round(product**(1/len(vals)), 2))
"
```

## CAGR
```bash
python3 -c "
start = 100; end = 200; years = 5
cagr = (end/start)**(1/years) - 1
print(round(cagr*100, 2))  # as percentage
"
```

## Variance
```bash
python3 -c "
vals = [v1, v2, ...]
n = len(vals)
mean = sum(vals)/n
var = sum((v-mean)**2 for v in vals) / n  # population
print(round(var, 5))
"
```

## H-Spread (IQR)
```bash
python3 -c "
vals = sorted([v1, v2, ...])
n = len(vals)
q1_idx = (n+1)*0.25 - 1
q3_idx = (n+1)*0.75 - 1
# Linear interpolation
import math
def percentile(data, p):
    k = (len(data)-1)*p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c: return data[int(k)]
    return data[f]*(c-k) + data[c]*(k-f)
q1 = percentile(vals, 0.25)
q3 = percentile(vals, 0.75)
print(round(q3 - q1, 2))
"
```

## Gini Coefficient
```bash
python3 -c "
vals = sorted([v1, v2, ...])
n = len(vals)
total = sum(vals)
cum = 0
area = 0
for i, v in enumerate(vals):
    cum += v
    area += cum/total - (i+1)/n
gini = area / (n/2)
print(round(abs(gini), 3))
"
```

## Zipf Exponent (log-log regression)
```bash
python3 -c "
import math
vals = sorted([v1, v2, ...], reverse=True)
n = len(vals)
log_rank = [math.log(i+1) for i in range(n)]
log_val = [math.log(v) for v in vals]
mx = sum(log_rank)/n
my = sum(log_val)/n
slope = sum((x-mx)*(y-my) for x,y in zip(log_rank,log_val)) / sum((x-mx)**2 for x in log_rank)
print(round(-slope, 3))
"
```
