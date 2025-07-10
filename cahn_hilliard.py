from solver import System
import torch
import time


N = 256
L = 256
dt = 0.1
steps = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

solver = System(N=N, L=L, dt=dt, device=device, record_every_n_steps = steps/100)

# # --- Parameters ---
a = -2
b = 1
k = 4

# # --- Add active fields ---
solver.model.add_active_field(
    "u",
    init = 0.1 * torch.randn((N, N)),
    L_hat = -solver.q2 * (a + k*solver.q2),
    N_hat = lambda R: -solver.q2 * b * torch.fft.fft2(R["u"].f**3)
)

traj = []
start = time.time()
for i in range(steps):
    solver.run(1)
    if i % solver.record_every_n_steps == 0:
        traj.append(solver.model.fields['u'].f)
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
traj = torch.stack(traj)


solver.visualize_pygame(data = traj)

