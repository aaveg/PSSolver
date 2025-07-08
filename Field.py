import torch



class Field:
    def __init__(self, name, N, dynamic: bool, device):
        """
        f0: initial field in real space (torch.Tensor)
        L_hat_fn: function (kx, ky, k2, u) -> linear term in spectral space
        nonlinear_fn: function (kx, ky, k2, u) -> nonlinear term in spectral space
        """
        self.name = name
        self.device = device
        self.f = torch.zeros((N, N), device=self.device)
        self.f_hat = torch.fft.fft2(self.f)

        self.L_hat = None
        self.is_dynamic = dynamic


    def set_initial_condition(self, f0):
        self.f = f0.to(self.device)
        self.f_hat = torch.fft.fft2(self.f)
    
    def set_L_hat(self, L_hat):
        self.L_hat = L_hat

    # def update_spatial(self):
    #     self.f = torch.fft.ifft2(self.f_hat).real

    # def update_spectral(self):
    #     self.f_hat = torch.fft.fft2(self.f)

    # def linear_term(self, kx, ky, k2):
    #     if self.L_hat_fn is not None:
    #         return self.L_hat_fn(kx, ky, k2, self.f)
    #     else:
    #         return torch.zeros_like(self.f_hat)

    # def nonlinear_term(self, kx, ky, k2):
    #     if self.nonlinear_fn is not None:
    #         return self.nonlinear_fn(kx, ky, k2, self.f)
    #     else:
    #         return torch.zeros_like(self.f_hat)
        

    # def set_nonlinear_fn(self, nonlinear_fn):
    #     self.nonlinear_fn = nonlinear_fn
