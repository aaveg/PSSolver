import torch
import time

# Parameters
n_steps = 100000
shape = (100, 100)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generate two random fields on GPU
field1 = torch.rand(*shape, device=device)
field2 = torch.rand(*shape, device=device)

# Method 1: FFT two separate fields
start1 = time.time()
for _ in range(n_steps):
    fft1 = torch.fft.fft2(field1)
    fft2 = torch.fft.fft2(field2)
torch.cuda.synchronize()
end1 = time.time()
time_separate = end1 - start1

# Method 2: Stack and batch FFT
start2 = time.time()
for _ in range(n_steps):
    fields = torch.stack([field1, field2], dim=0)
    fft_batch = torch.fft.fft2(fields, dim=(-2, -1))
torch.cuda.synchronize()
end2 = time.time()
time_batch = end2 - start2

print(f"Separate FFTs time: {time_separate:.4f} seconds")
print(f"Batch FFT time:     {time_batch:.4f} seconds")