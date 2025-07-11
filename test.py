import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dyn = torch.randn(4, 256, 256, device=device)
stat = torch.randn(2, 256, 256, device=device)
combined = torch.cat([stat, dyn], dim=0)

steps = 100000
# First loop: per-matrix FFT/IFFT
start = time.time()
for _ in range(steps):
    for d in dyn:
        d_fft = torch.fft.fft2(d)
        d_ifft = torch.fft.ifft2(d_fft).real
    for s in stat:
        s_fft = torch.fft.fft2(s)
        s_ifft = torch.fft.ifft2(s_fft).real
end = time.time()
print(f"Loop 1 (per-matrix): {end - start:.4f} seconds")

# Second loop: batch FFT/IFFT
start = time.time()
for _ in range(steps):
    d_fft = torch.fft.fft2(dyn)
    d_ifft = torch.fft.ifft2(d_fft).real

    s_fft = torch.fft.fft2(stat)
    s_ifft = torch.fft.ifft2(s_fft).real
end = time.time()
print(f"Loop 2 (batch): {end - start:.4f} seconds")

# Third loop: combined FFT/IFFT
start = time.time()
for _ in range(steps):
    dyn_fft = torch.fft.fft2(combined)
    dyn_ifft = torch.fft.ifft2(dyn_fft).real
end = time.time()
print(f"Loop 3 (combined): {end - start:.4f} seconds")

# Third loop: combined and sliced FFT/IFFT
start = time.time()
for _ in range(steps):
    d_fft = torch.fft.fft2(combined[2:])
    d_ifft = torch.fft.ifft2(d_fft).real

    s_fft = torch.fft.fft2(combined[:2])
    s_ifft = torch.fft.ifft2(s_fft).real
end = time.time()
print(f"Loop 3 (combined): {end - start:.4f} seconds")