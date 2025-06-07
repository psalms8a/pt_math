import numpy as np
import math
from approx_exp import approx_exp, MAX_X, MIN_X

NUM_SAMPLES = 1000000
# Uniform samples across a very wide range
inputs = np.random.uniform(-120.0, 120.0, size=NUM_SAMPLES).astype(np.float32)
# Append some special cases
specials = np.array([
    float('nan'), float('inf'), float('-inf'),
    MAX_X + 1.0, MIN_X - 1.0, 0.0, 1.0, -1.0
], dtype=np.float32)
inputs = np.concatenate((inputs, specials))

# Reference implementation using math.exp but with explicit overflow/underflow handling
ref = []
for x in inputs:
    if math.isnan(float(x)):
        ref.append(float('nan'))
    elif x > MAX_X:
        ref.append(float('inf'))
    elif x < MIN_X:
        ref.append(0.0)
    else:
        ref.append(float(math.exp(float(x))))
ref = np.array(ref, dtype=np.float32)

res = np.array([approx_exp(float(x)) for x in inputs], dtype=np.float32)

# Compute ULP error only for finite results within range
mask = (inputs >= MIN_X) & (inputs <= MAX_X) & np.isfinite(inputs)
ulps = np.abs(
    np.frombuffer(res[mask].tobytes(), dtype=np.uint32).astype(np.int64) -
    np.frombuffer(ref[mask].tobytes(), dtype=np.uint32).astype(np.int64)
)

nan_mismatch = np.sum(np.isnan(ref) != np.isnan(res))
inf_mismatch = np.sum(np.isinf(ref) != np.isinf(res))

print(f"Tested {len(inputs)} samples")
print(f"Within range max ULP: {ulps.max()}")
print(f"Within range mean ULP: {ulps.mean():.5f}")
print(f"NaN mismatches: {nan_mismatch}, Inf mismatches: {inf_mismatch}")
