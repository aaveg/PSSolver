from pssolver import SpectralSolver
import torch
import time
from tqdm import trange


class CH_NLmodel(torch.nn.Module):
    def forward(self, fields, params): 
        u = fields['u']  
        q2 = fields.q2
        b = params['b']

        output = - q2 * b * torch.fft.fft2(u**3)
        return output.unsqueeze(0)

 
N = 256
L = 256
dt = 0.1
steps = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
solver = SpectralSolver(shape=(N,N), L=L, dt=dt, device=device, batch_size= 3 )

# # --- Parameters ---
a = -2
b = 1
k = 4

# # --- Add active fields ---
solver.model.add_dynamic_field(
    "u",
    init = 0.1* torch.randn(N, N),
    L_hat = -solver.q2 * (a + k*solver.q2)
)

solver.model.set_nonlinear_model(CH_NLmodel())
solver.model.parameters.set_param('b', 1)
solver.build()

traj = []
start = time.time()
for i in trange(steps):
    solver.run(1)
    if i % (steps//100) == 0:
        traj.append(solver.model.fields['u'])
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
traj = torch.stack(traj).permute(1,0,2,3)

print(traj.shape)
solver.visualize(data = traj[0])
# solver.visualize(data = traj[1])
# solver.visualize(data = traj[2])

