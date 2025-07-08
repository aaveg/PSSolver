import torch
import torch.fft

# class PseudoSpectralSolver2D:
#     def __init__(self, N, L=2 * torch.pi, dt=0.01, device='cpu',record_every_n_steps = 1000):
#         self.N = N
#         self.L = L
#         self.dt = dt
#         self.device = device
#         self.record_every_n_steps = record_every_n_steps

#         # Spatial grid (not strictly needed but kept for extensibility)
#         x = torch.linspace(0, L - L / N, N, device=device)
#         self.X, self.Y = torch.meshgrid(x, x, indexing='ij')

#         # Wavenumbers
#         k = torch.fft.fftfreq(N, d=L/N).to(device) * 2 * torch.pi
#         self.kx, self.ky = torch.meshgrid(k, k, indexing='ij')
#         self.k2 = self.kx**2 + self.ky**2

#         self.k2[0, 0] = 1e-10  # avoid div by 0 if using 1/k^2 later

#         # Current field in real and spectral space
#         self.u = None
#         self.u_hat = None
#         self.L_hat_fn = None
#         self.N_hat_fn = None

#         self.L_hat = None
#         self.mask = self._make_dealias_mask(N, device)
        
#     def set_initial_condition(self, u0):
#         """
#         Sets the initial condition for the solver.
#         u0: torch.Tensor of shape (N, N), real space initial field
#         """
#         self.u = u0.to(self.device)
#         self.u_hat = torch.fft.fft2(self.u)

#     def set_linear_func(self, L_hat_fn):
#         """
#         Sets and precomputes the linear term in spectral space.
#         linear_term_func: function that takes (kx, ky, k2) and returns L_hat
#         """
#         self.L_hat_fn = L_hat_fn
#         # Precompute linear operator
#         self.L_hat = L_hat_fn(self.kx, self.ky, self.k2)
#         # Precompute ETDRK2 coefficients
#         self._precompute_etdrk2()

#     def set_nonlinear_func(self, N_hat_fn):
#         """
#         Adds a nonlinear term function to the solver.
#         nonlinear_term_func: function that takes u (real space) and returns nonlinear term (real space)
#         """
#         self.N_hat_fn = N_hat_fn


#     def _make_dealias_mask(self, N, device):
#         # Frequency components for FFT (e.g., -N/2 to N/2-1)
#         k = torch.fft.fftfreq(N, d=1/N).to(device)
#         kx, ky = torch.meshgrid(k, k, indexing="ij")

#         # Apply 2/3 rule: zero out all wavenumbers beyond N/3
#         cutoff = N // 3
#         mask = (kx.abs() < cutoff) & (ky.abs() < cutoff)
#         return mask

#     def step(self):
#         dt = self.dt

#         # Compute nonlinear and linear terms
#         N_hat = self.N_hat_fn(self.kx, self.ky, self.k2, self.u)
#         # N_hat = self.dealias(N_hat)
#         L_hat = self.L_hat

#         # Forward Euler update in spectral space
#         # self.u_hat += dt * (L_hat + N_hat)
#         self.u_hat += dt * (N_hat)#*self.mask
#         self.u_hat /= (1- dt*L_hat)
        

#         self.u = torch.fft.ifft2(self.u_hat).real
#         # self.u_hat = torch.fft.fft2(self.u)


#     def _precompute_etdrk2(self):
#         dt = self.dt
#         L = self.L_hat

#         # # Ensure L is a tensor, and broadcast if needed
#         # if L.ndim == 0:
#         #     L = L * torch.ones_like(self.k2)  # broadcast to shape [N, N]

#         # self.L_hat = L  # store back the expanded tensor
#         self.E = torch.exp(dt * L)

#         # Avoid division by zero in phi1
#         L_safe = torch.where(L == 0, torch.tensor(1.0, device=L.device), L)
#         self.phi1 = (self.E - 1) / L_safe
#         self.phi1[L == 0] = dt

#     def step_etdrk1(self):
#         if self.N_hat_fn is not None:
#             N_hat = self.N_hat_fn(self.kx, self.ky, self.k2, self.u)
#         else:
#             N_hat = torch.zeros_like(self.u_hat)

#         # Final update
#         self.u_hat = self.E * self.u_hat +  self.phi1 * N_hat
#         # u^n → real space
#         self.u = torch.fft.ifft2(self.u_hat).real


#     def step_etdrk2(self):

#         # N(u^n)
#         N1_hat = self.N_hat_fn(self.kx, self.ky, self.k2, self.u)*self.mask
#         # N1_hat = torch.fft.fft2(N1)

#         # ũ = E * u_hat + dt * phi1 * N1_hat
#         u_hat_tilde = self.E * self.u_hat +  self.phi1 * N1_hat

#         # N(ũ)
#         u_tilde = torch.fft.ifft2(u_hat_tilde).real
#         N2_hat = self.N_hat_fn(self.kx, self.ky, self.k2, u_tilde)*self.mask
#         # N2_hat = torch.fft.fft2(N2)

#         # Final update
#         self.u_hat = self.E * self.u_hat +  self.phi1 * 0.5 * (N1_hat + N2_hat)
#         # self.u_hat = self.dealias(self.u_hat)
#         # u^n → real space
#         self.u = torch.fft.ifft2(self.u_hat).real

#     def run(self, steps, callback = None):
#         traj = []
#         for i in range(steps):
#             self.step()
#             self.u_hat = torch.fft.fft2(self.u)
#             if i%self.record_every_n_steps == 0:
#                 traj.append(self.u)

#         return torch.stack(traj)




from integrator import SemiImplicitEulerIntegrator
from PDEModel import PDEModel
from Field import Field

class System:
    def __init__(self, N, L=2 * torch.pi, dt=0.01, device='cuda', record_every_n_steps=1000):

        self.N = N
        self.L = L
        self.dt = dt
        self.device = device
        self.record_every_n_steps = record_every_n_steps

        self._init_q_space()

        self.integrator = SemiImplicitEulerIntegrator(dt, self.qx, self.qy, self.q2)
        self.model = PDEModel(N, device)


    def _init_q_space(self):
        # Spatial grid (not strictly needed but kept for extensibility)
        x = torch.linspace(0, self.L - self.L / self.N, self.N, device=self.device)
        self.X, self.Y = torch.meshgrid(x, x, indexing='ij')

        # Wavenumbers
        q = torch.fft.fftfreq(self.N, d=self.L/self.N).to(self.device) * 2 * torch.pi
        self.qx, self.qy = torch.meshgrid(q, q, indexing='ij')
        self.q2 = self.qx**2 + self.qy**2
        self.q2[0, 0] = 1e-10  # avoid div by 0 if using 1/q^2 later

    
    def run(self, steps, callback = None):
        for step in range(steps):
            self.integrator.step(self.model)
            if callback is not None:
                callback(self,step)

