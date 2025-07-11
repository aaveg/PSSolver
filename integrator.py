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
        fields = model.fields

        for name, field in fields.items():
            L_hat = field.L_hat  # = field.linear_term(self.qx, self.qy, self.q2)
            N_hat = N_hats[name]

            field.f_hat = (field.f_hat + self.dt * N_hat) / (1 - self.dt * L_hat)
        
        field.f = torch.fft.ifft2(field.f_hat).real
        field.f_hat = torch.fft.fft2(field.f)
