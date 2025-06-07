import numpy as np
from approx_exp import approx_exp

NUM_SAMPLES = 100000
inputs = np.linspace(-20, 20, NUM_SAMPLES, dtype=np.float32)

# reference using double precision
ref = np.exp(inputs.astype(np.float64)).astype(np.float32)
res = np.array([approx_exp(float(x)) for x in inputs], dtype=np.float32)

def ulp_diff(a, b):
    ai = np.frombuffer(np.float32(a).tobytes(), dtype=np.uint32)
    bi = np.frombuffer(np.float32(b).tobytes(), dtype=np.uint32)
    return np.abs(ai.astype(np.int64) - bi.astype(np.int64))

ulps = ulp_diff(res, ref)
max_ulp = ulps.max()
mean_ulp = ulps.mean()
print(f"max ULP error: {max_ulp}")
print(f"mean ULP error: {mean_ulp:.5f}")
