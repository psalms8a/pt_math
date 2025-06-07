import numpy as np
try:
    import torch
except ImportError:
    raise SystemExit('PyTorch is not installed')

NUM_SAMPLES = 100000
# range of inputs for exp
inputs = np.linspace(-20, 20, NUM_SAMPLES, dtype=np.float32)

# torch computation (float32)
xt = torch.from_numpy(inputs)
with torch.no_grad():
    yt = torch.exp(xt).numpy()

# reference using double precision
ref = np.exp(inputs.astype(np.float64)).astype(np.float32)

def ulp_diff(a, b):
    ai = np.frombuffer(np.float32(a).tobytes(), dtype=np.uint32)
    bi = np.frombuffer(np.float32(b).tobytes(), dtype=np.uint32)
    return np.abs(ai.astype(np.int64) - bi.astype(np.int64))

ulps = ulp_diff(yt, ref)

max_ulp = ulps.max()
mean_ulp = ulps.mean()
print(f"max ULP error: {max_ulp}")
print(f"mean ULP error: {mean_ulp:.4f}")

np.savetxt('exp_accuracy.csv',
           np.column_stack((inputs, ulps)),
           delimiter=',',
           header='input,ulp_error',
           comments='')
print('results saved to exp_accuracy.csv')
