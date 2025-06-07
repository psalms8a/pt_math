"""Pure Python exp approximation using polynomial and range reduction.
"""

LN2 = 0.6931471805599453
LOG2_E = 1.4426950408889634

# Maclaurin series coefficients for exp(x) up to x^7 / 7!
C7 = 1.0 / 5040.0
C6 = 1.0 / 720.0
C5 = 1.0 / 120.0
C4 = 1.0 / 24.0
C3 = 1.0 / 6.0
C2 = 0.5
C1 = 1.0
C0 = 1.0

# thresholds for overflow/underflow based on float32 limits
MAX_X = 88.7    # ln(FLT_MAX)
MIN_X = -103.972 # ln(FLT_MIN)

def approx_exp(x: float) -> float:
    """Approximate exp(x) for float inputs using only basic operations."""
    # handle NaN and infinities
    if x != x:
        return x  # NaN
    if x > MAX_X:
        return float('inf')
    if x < MIN_X:
        return 0.0

    # range reduction: x = n * ln2 + r, where r in [-ln2/2, ln2/2]
    n = int(x * LOG2_E + (0.5 if x >= 0 else -0.5))
    r = x - n * LN2

    # polynomial approximation of e^r
    p = C7
    p = p * r + C6
    p = p * r + C5
    p = p * r + C4
    p = p * r + C3
    p = p * r + C2
    p = p * r + C1
    p = p * r + C0

    # multiply by 2**n using repeated multiplication
    if n > 0:
        for _ in range(n):
            p *= 2.0
    elif n < 0:
        for _ in range(-n):
            p *= 0.5
    return p

if __name__ == "__main__":
    # small manual test
    for val in [-20, -10, -1, 0, 1, 10, 20]:
        print(val, approx_exp(val))
