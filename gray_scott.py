from solver import System
import torch
import time
from tqdm import trange


def init_u(N):
    u = torch.ones(N, N)
    r = 10  # half size of perturbation square
    u[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.50
    u += 0.02 * torch.randn(N, N)
    return u

def init_v(N):
    v = torch.zeros(N, N)
    r = 10
    v[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.25
    v += 0.02 * torch.randn(N, N)
    return v

class NonlinearModel(torch.nn.Module):
    def forward(self, fields): 
        u = fields['u']  
        v = fields['v']

        F = 0.06
        k = 0.062
        vsq = v**2
        out0 = -u * vsq + F * (1 - u)
        out1 = u * vsq - (F + k) * v

        return torch.fft.fft2(torch.stack([out0, out1]))  


N = 128
L = 128
dt = 0.1
steps = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

solver = System(shape = (N,N), L=L, dt=dt, device=device, record_every_n_steps = steps/100)

# # --- Parameters ---
du = 0.16
dv = 0.08
F = 0.06
k = 0.062

# --- Add active fields ---
solver.model.add_dynamic_field(
    "u",
    init = init_u(N),
    L_hat = -solver.q2 * (du)
)
# --- Add active fields ---
solver.model.add_dynamic_field(
    "v",
    init =  init_v(N),
    L_hat = -solver.q2 * (dv)
)


solver.model.set_nonlinear_model(NonlinearModel())
solver.build()

traj = []
start = time.time()
for i in trange(steps):
    solver.run(1)
    if i % solver.record_every_n_steps == 0:
        traj.append(solver.model.fields['u'])
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
traj = torch.stack(traj)


solver.visualize_pygame(data = traj)

