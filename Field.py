import torch



# class Field:
#     def __init__(self, name, N, dynamic: bool, device):
#         """
#         f0: initial field in real space (torch.Tensor)
#         L_hat_fn: function (kx, ky, k2, u) -> linear term in spectral space
#         nonlinear_fn: function (kx, ky, k2, u) -> nonlinear term in spectral space
#         """
#         self.name = name
#         self.device = device
#         self.f = torch.zeros((N, N), device=self.device)
#         self.f_hat = torch.fft.fft2(self.f)

#         self.L_hat = None
#         self.is_dynamic = dynamic


#     def set_initial_condition(self, f0):
#         self.f = f0.to(self.device)
#         self.f_hat = torch.fft.fft2(self.f)
    
#     def set_L_hat(self, L_hat):
#         self.L_hat = L_hat




class Fields:
    def __init__(self, field_names, shape, device="cuda", dtype=torch.float32):
        """
        field_names: list of str, e.g., ['u', 'v']
        shape: (Nx, Ny)
        """
        self.names = field_names
        self.name_to_index = {name: i for i, name in enumerate(field_names)}
        self.Nx, self.Ny = shape
        self.device = device
        self.dtype = dtype

        self.data = torch.zeros((len(field_names), self.Nx, self.Ny), 
                                device=device, dtype=dtype)

    def __getitem__(self, key):
        """Access a field by name or index"""
        if isinstance(key, str):
            return self.data[self.name_to_index[key]]
        elif isinstance(key, int):
            return self.data[key]
        else:
            raise KeyError("Key must be field name (str) or index (int)")

    def __setitem__(self, key, value):
        """Update a field by name or index"""
        if isinstance(key, str):
            self.data[self.name_to_index[key]] = value
        elif isinstance(key, int):
            self.data[key] = value
        else:
            raise KeyError("Key must be field name (str) or index (int)")

    def fft2(self):
        """Return batched FFT of all fields"""
        return torch.fft.fft2(self.data)

    def ifft2(self):
        """Return batched IFFT (real part) of all fields"""
        return torch.fft.ifft2(self.data).real

    def to(self, device):
        """Move to a new device"""
        self.device = device
        self.data = self.data.to(device)
        return self

    def clone(self):
        """Return a new Fields object with cloned data"""
        new = Fields(self.names, (self.Nx, self.Ny), self.device, self.dtype)
        new.data = self.data.clone()
        return new

    def keys(self):
        return self.names

    def values(self):
        return [self[name] for name in self.names]

    def items(self):
        return [(name, self[name]) for name in self.names]