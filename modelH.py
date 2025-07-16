import time
import torch
from solver import System

class NonlinearModel(torch.nn.Module):
    def forward(self, fields): 
        phi = fields['phi']  
        q2 = fields.q2
        
        ux = fields['ux']
        uy = fields['uy']
        gxphi = fields['gradx_phi']
        gyphi = fields['grady_phi']
       
        out0  =  - b* q2* torch.fft.fft2(phi**3) + torch.fft.fft2(- ux*gxphi - uy*gyphi)
        
        return out0.unsqueeze(0)  

class Static_compute_fn(torch.nn.Module):
    def forward(self, fields): 
        ### avoid repeating same calculations

        # Cache coefficients and linear term on first forward call
        if not hasattr(self, 'P'):
            qx = fields.qx
            qy = fields.qy
            iqx = 1j * qx
            iqy = 1j * qy
            q2 = fields.q2
            self.iqx = iqx
            self.iqy = iqy

            # Stokes flow projection operator (in Fourier space)
            # Projects a vector field onto its divergence-free component
            P = torch.zeros((2, 2, *q2.shape), dtype=torch.cfloat, device=q2.device)
            P[0, 0] = 1 - (qx * qx) / q2
            P[0, 1] = - (qx * qy) / q2
            P[1, 0] = - (qy * qx) / q2
            P[1, 1] = 1 - (qy * qy) / q2
            self.P = P * 1/(eta*q2)


        sigxx =  -k/2 * (fields['gradx_phi']**2 - fields['grady_phi']**2)  
        sigxy =  -k * (fields['gradx_phi'] * fields['grady_phi'])  
        sigxx_hat, sigxy_hat = torch.fft.fft2(torch.stack([sigxx,sigxy]))

        # Create sig tensor with shape (2, N, N)
        sig_hat = torch.zeros((2,2, *sigxx.shape), dtype=sigxy_hat.dtype, device = sigxy_hat.device)
        sig_hat[0,0] = sigxx_hat
        sig_hat[0,1] = sigxy_hat
        sig_hat[1,0] = sigxy_hat
        sig_hat[1,1] = -sigxx_hat
         
        f_hat = torch.einsum('jxy,ijxy->ixy', torch.stack([self.iqx,self.iqy]), sig_hat)  
        u_hat = torch.einsum('abij,bij->aij', self.P, f_hat) 

        ux_hat = u_hat[0]
        uy_hat = u_hat[1]
        
        phi_hat = fields['phi.hat']
        gradx_phi_hat = self.iqx * phi_hat
        grady_phi_hat = self.iqy * phi_hat

        # return order is [ux, uy, gxphi, gyphi]
        return torch.stack([
            ux_hat,
            uy_hat,
            gradx_phi_hat,
            grady_phi_hat,
        ])




seed = 42
N = 128
L = 128
dt = 0.1
steps = 20000
device = 'cuda' if torch.cuda.is_available() else 'cpu'

solver = System(shape = (N,N), L=L, dt=dt, device=device, record_every_n_steps = steps/100)

# # --- Parameters ---
a = -1
b = 1
k = 4
eta = 1

# --- Add active fields ---
solver.model.add_dynamic_field(
    "phi",
    init =  0.1 * torch.randn((N, N)),
    L_hat = - solver.q2*(a + k*solver.q2)
)

# --- Add static fields ---
solver.model.add_static_field("ux")
solver.model.add_static_field("uy")
solver.model.add_static_field("gradx_phi")
solver.model.add_static_field("grady_phi")


solver.model.set_nonlinear_model(NonlinearModel())
solver.model.set_static_compute_model(Static_compute_fn())

solver.build()
print(solver.model.fields.dyn_count)
print(solver.model.fields.name_to_idx)

traj = []
start = time.time()
for i in range(steps):
    if i % solver.record_every_n_steps == 0:
        snapshot = torch.stack([solver.model.fields[name].clone().detach().cpu() for name in solver.model.fields.name_to_idx])
        traj.append(snapshot)

    solver.run(1)
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
traj = torch.stack(traj).permute(1,0,2,3)

solver.visualize(data = traj[0])
