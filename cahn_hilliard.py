from solver import System
import torch
import time


N = 128
L = 128
dt = 0.1
steps = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

solver = System(N=N, L=L, dt=dt, device=device, record_every_n_steps = steps/100)

# # --- Parameters ---
a = -1
b = 1
k = 4

# # --- Add active fields ---
solver.model.add_active_field(
    "u",
    init = 0.1 * torch.randn((N, N)),
    L_hat = -solver.q2 * (a + k*solver.q2),
    N_hat = lambda R: -solver.q2 * b * torch.fft.fft2(R["u"].f**3)
)

start_time = time.time()
solver.run(steps)
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")
