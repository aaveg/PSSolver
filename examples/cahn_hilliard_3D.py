from pssolver import SpectralSolver
import torch
import time
from tqdm import trange


class CH_NLmodel(torch.nn.Module):
    def forward(self, fields,params): 
        u = fields['u']  
        q2 = fields.q2

        output = - q2 * b * torch.fft.fftn(u**3, dim=[-1,-2,-3])
        return output.unsqueeze(0)


N = 256
L = 128
dt = 0.1
steps = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
solver = SpectralSolver(shape=(N,N,N), L=L, dt=dt, device=device, record_every_n_steps = steps//100)

# # --- Parameters ---
a = -2
b = 1
k = 4

# # --- Add active fields ---
solver.model.add_dynamic_field(
    "u",
    init = 0.1 * torch.randn((N, N, N)),
    L_hat = -solver.q2 * (a + k*solver.q2)
)

solver.model.set_nonlinear_model(CH_NLmodel())
solver.build()

traj = []
start = time.time()

for i in trange(steps//solver.record_every_n_steps, desc="Solving"):
    solver.run(solver.record_every_n_steps)
    traj.append(solver.model.fields['u'])
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")

traj = torch.stack(traj)


# solver.visualize_pygame(data = traj)
# solver.visualize3d(data = traj)

