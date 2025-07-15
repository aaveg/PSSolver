import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose shapes
dyn_shape_0 = 22  # Change as needed
# stat_shape_0 = 8  # Change as needed

dyn = torch.randn(dyn_shape_0, 256, 256, device=device)
dyn2 = dyn.clone()
# stat = torch.randn(stat_shape_0, 256, 256, device=device)
# combined = torch.cat([stat, dyn], dim=0)
lhat = torch.randn(3,256,256, device= device)
# Create lhat2 by padding lhat with zeros to match dyn's first dimension, leave lhat unchanged

cache = torch.stack([torch.zeros(256,256, device=device) for _ in range(dyn_shape_0-4)])
def generate_lhat():
    """
    Generates lhat2 by padding a random lhat tensor with zeros to match dyn's first dimension.
    Returns both lhat and lhat2.
    """
    lhat = torch.stack([torch.randn(256,256, device=device) for _ in range(4)])
    return lhat

def generate_lhat2():
    """
    Generates lhat2 by padding a random lhat tensor with zeros to match dyn's first dimension.
    Returns both lhat and lhat2.
    """
    lhat = torch.stack([torch.randn(256,256, device=device) for _ in range(4)])
    lhat2 = torch.cat([lhat, cache], dim=0)
    return lhat2

# Example usage:
# lhat, lhat2 = generate_lhat2(dyn.shape[0], 3, 256, 256, device=device)

steps = 100000
torch.manual_seed(42)

# First loop: per-matrix FFT/IFFT
start = time.time()
for _ in range(steps):
    lhat = generate_lhat()
    dyn[0:4] += lhat
    dyn[0:4] /= (1+lhat)
torch.cuda.synchronize()
end = time.time()
print(f"Loop 1 (per-matrix): {end - start:.4f} seconds")

torch.manual_seed(42)
# Second loop: batch FFT/IFFT
start = time.time()
for _ in range(steps):
    lhat2 = generate_lhat2()
    dyn2 += lhat2
    dyn2 /= (1+lhat2)
torch.cuda.synchronize()
end = time.time()
print(f"Loop 2 (batch): {end - start:.4f} seconds")

print(torch.equal(dyn, dyn2))
