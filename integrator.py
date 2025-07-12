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
    def __init__(self, dt, qx, qy, q2):
        super().__init__(dt, qx, qy, q2)

    def step(self, model):
        N_hats = model.compute_nonlinear()
        L_hats = model.fields.L_hat

        dyn_idx = model.fields.dyn_idx
        # fields = model.fields
        # print(model.fields.spectral[model.fields.dyn_idx])
        # model.fields.spectral[dyn_idx] = (model.fields.spectral[dyn_idx] + self.dt * N_hats) / (1 - self.dt * L_hats)
        model.fields.spectral[:model.fields.field_count] = (model.fields.spectral[:model.fields.field_count] + self.dt * N_hats) / (1 - self.dt * L_hats)

        # Use in-place operations to reduce memory allocations and improve speed
        # for idx in dyn_idx:
        #     model.fields.spectral[idx] = (model.fields.spectral[idx] + self.dt * N_hats[idx]) / (1 - self.dt * L_hats[idx])

        # model.fields.spectral[dyn_idx] += (self.dt * N_hats) 
        # model.fields.spectral[dyn_idx] /= (1 - self.dt * L_hats)
        
        # model.fields.spectral = (model.fields.spectral + self.dt * N_hats) / (1 - self.dt * L_hats)

        # model.fields.spectral += self.dt * N_hats
        # model.fields.spectral /= (1 - self.dt * L_hats)



        # print(model.fields.spectral[model.fields.dyn_idx])
        # for name, field in fields.items():
        #     L_hat = field.L_hat  # = field.linear_term(self.qx, self.qy, self.q2)
        #     N_hat = N_hats[name]

        #     field.f_hat = (field.f_hat + self.dt * N_hat) / (1 - self.dt * L_hat)
        
        model.fields.spatial = model.fields.ifftn() #torch.fft.ifft2(model.fields.spectral).real
        model.fields.spectral = model.fields.fftn() #torch.fft.fft2(field.f)
