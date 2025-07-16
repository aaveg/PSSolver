import torch

class TimeIntegrator:
    def __init__(self, dt, qx, qy, q2):
        self.dt = dt
        self.qx = qx
        self.qy = qy
        self.q2 = q2

    def step(self):
        raise NotImplementedError("Implement in subclass")


class SemiImplicitEulerIntegrator(TimeIntegrator):
    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(dt, qx, qy, q2)
        self.model = model
        self.denom = 1 - model.fields.L_hat *self.dt # cache for denominator in the update step = (1 - L_hat*dt)
        self.dyn_count = self.model.fields.dyn_count
        self.stat_count = self.model.fields.stat_count

        self.step_count = 0

    def step(self):
        N_hats = self.model.compute_nonlinear() 

        # Use in-place operations to reduce memory allocations and improve speed
        dyn_fields = self.model.fields.spectral[:self.dyn_count]
        dyn_fields.add_(self.dt * N_hats)
        dyn_fields.div_(self.denom)
        
        if self.stat_count != 0:
            self.model.fields.spectral[self.dyn_count:] = self.model.compute_static() 
        
        # self.model.fields.spectral *= self.model.fields.dealiasing_mask

        self.model.fields.spatial = self.model.fields.ifftn() # calculate spatial from spectral

        self.step_count += 1

        # spectral cleanup. Taking fft after ifft is critical for stability. 
        if self.step_count % 20 == 0:
            self.model.fields.spectral = self.model.fields.fftn() 
            self.step_count = 0
