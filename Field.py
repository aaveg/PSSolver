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



