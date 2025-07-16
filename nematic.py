import time
import torch
from solver import System
from tqdm import trange

def Q_init(shape, seed = 42,):
    Nx,Ny = shape
    generator = torch.Generator().manual_seed(seed)
    angle_radians = torch.tensor(torch.pi / 3)
    nx_ = torch.cos(angle_radians)
    ny_ = torch.sin(angle_radians)
    Qxx = (nx_ * nx_ - 0.5) + 0.01*(2 *torch.rand((Nx, Ny), generator=generator) - 1)
    Qxy = nx_ * ny_ + 0.01*(2 * torch.rand((Nx, Ny), generator=generator) - 1)

    return Qxx , Qxy
     

class NonlinearModel(torch.nn.Module):
    def forward(self, fields): 
        Qxx = fields['Qxx']  
        Qxy = fields['Qxy']
        Qsq = Qxx**2 + Qxy**2

        ux = fields['ux']
        uy = fields['uy']
        w  = fields['w']
        gxQxx = fields['gradx_Qxx']
        gyQxx = fields['grady_Qxx']
        gxQxy = fields['gradx_Qxy']
        gyQxy = fields['grady_Qxy']
        Axx = fields['Axx']
        Axy = fields['Axy']

        lam = 1
        out0 =  -a4*Qsq*Qxx - ux*gxQxx - uy*gyQxx - 2*Qxy*w + lam*Axx #lam*iqx*ux #lam*Axx 
        out1 =  -a4*Qsq*Qxy - ux*gxQxy - uy*gyQxy + 2*Qxx*w + lam*Axy #0.5*lam*(iqx*uy + iqy*ux) #lam*Axy 
        
        return torch.fft.fft2(torch.stack([out0, out1]))  

class Static_compute_fn(torch.nn.Module):

    ### avoid repeating same calculations
    def forward(self, fields): 
        # Cache Projection operator on first forward call
        if not hasattr(self, 'P'):
            qx = fields.qx
            qy = fields.qy
            iqx = 1j * qx
            iqy = 1j * qy
            q2 = fields.q2
            self.iqx = iqx
            self.iqy = iqy
            P = torch.zeros((2, 2, *q2.shape), dtype=torch.cfloat, device=q2.device)
            P[0, 0] = 1 - (qx * qx) / q2
            P[0, 1] = - (qx * qy) / q2
            P[1, 0] = - (qy * qx) / q2
            P[1, 1] = 1 - (qy * qy) / q2
            self.P = P * 1/(fric+eta*q2)

            sig_to_f = torch.zeros((2, 2, *q2.shape), dtype=torch.cfloat, device=q2.device)
            sig_to_f[0, 0] = iqx
            sig_to_f[0, 1] = iqy
            sig_to_f[1, 0] = -iqy
            sig_to_f[1, 1] = iqx
            self.sig_to_f = sig_to_f


        Qxx = fields['Qxx']  
        Qxy = fields['Qxy']
        Q = torch.stack([Qxx, Qxy], dim=0)
        sig =  beta * alpha * Q  
        sig_hat = torch.fft.fft2(sig)

        f_hat = torch.einsum('ijxy,jxy->ixy', self.sig_to_f, sig_hat)  
        u_hat = torch.einsum('abij,bij->aij', self.P, f_hat)

        # u_hat = 0*u_hat
        ux_hat = u_hat[0]
        uy_hat = u_hat[1]
        w_hat = 0.5*(self.iqx*uy_hat - self.iqy*ux_hat)  

        Axx_hat = self.iqx * ux_hat
        Axy_hat = 0.5*(self.iqx*uy_hat + self.iqy*ux_hat)
        
        Qxx_hat = fields['Qxx.hat']
        Qxy_hat = fields['Qxy.hat']   
        gradx_Qxx_hat = self.iqx * Qxx_hat
        grady_Qxx_hat = self.iqy * Qxx_hat
        gradx_Qxy_hat = self.iqx * Qxy_hat
        grady_Qxy_hat = self.iqy * Qxy_hat
        # return order is [ux, uy, w, Axx, Axy, gxQxx, gyQxx, gxQxy, gyQxy]
        return torch.stack([
            ux_hat,
            uy_hat,
            w_hat,
            Axx_hat,
            Axy_hat,
            gradx_Qxx_hat,
            grady_Qxx_hat,
            gradx_Qxy_hat,
            grady_Qxy_hat
        ])




seed = 42
N = 128
L = 128
dt = 0.01
steps = 25000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

solver = System(shape = (N,N), L=L, dt=dt, device=device, record_every_n_steps = 100)

Qxx_0, Qxy_0 = Q_init(shape = (N,N), seed = seed)

# # --- Parameters ---
a2 = -1
a4 = 1
KQ = 4

beta = -1
alpha = 0.4


# --- Add active fields ---
solver.model.add_dynamic_field(
    "Qxx",
    init = Qxx_0,
    L_hat = -( a2 + solver.q2 * KQ)
)
solver.model.add_dynamic_field(
    "Qxy",
    init =  Qxy_0,
    L_hat = -( a2 + solver.q2 * KQ)
)

fric = 0.01
eta = 1
# --- Add static fields ---
solver.model.add_static_field("ux")
solver.model.add_static_field("uy")
solver.model.add_static_field("w")
solver.model.add_static_field("Axx")
solver.model.add_static_field("Axy")
solver.model.add_static_field("gradx_Qxx")
solver.model.add_static_field("grady_Qxx")
solver.model.add_static_field("gradx_Qxy")
solver.model.add_static_field("grady_Qxy")


solver.model.set_nonlinear_model(NonlinearModel())
solver.model.set_static_compute_model(Static_compute_fn())

solver.build()
print(solver.model.fields.dyn_count)
print(solver.model.fields.name_to_idx)

traj = []
start = time.time()
for i in trange(steps):
    if i % solver.record_every_n_steps == 0:
        snapshot = torch.stack([solver.model.fields[name].clone().detach().cpu() for name in ["Qxx", "Qxy"]])
        traj.append(snapshot)
    solver.run(1)
    
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
traj = torch.stack(traj).permute(1,0,2,3)

# traj = torch.stack([torch.sqrt(Qxx_0**2 + Qxy_0**2)])

qxx = traj[0]  # shape: (time, nx, ny)
qxy = traj[1]  # shape: (time, nx, ny)
# Calculate scalar order parameter s
s = torch.sqrt(qxx**2 + qxy**2)  # shape: (time, nx, ny)

solver.visualize(data = s)
