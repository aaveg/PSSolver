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



# class Field:
#     def __init__(self, shape, device="cuda", dtype=torch.float32):
#         """
#         Base class for a field.
#         name: str, field name
#         shape: tuple, spatial shape
#         """
#         self.name = name
#         self.shape = shape
#         self.device = device
#         self.dtype = dtype
#         self.spatial = torch.zeros(shape, device=device, dtype=dtype)
#         self.spectral = torch.fft.fftn(self.spatial)
#         self.is_dynamic = False

#     def set_initial_condition(self, f0):
#         self.spatial = f0.to(self.device)
#         self.spectral = torch.fft.fftn(self.spatial)

#     def fftn(self):
#         self.spectral = torch.fft.fftn(self.spatial)
#         return self.spectral

#     def ifftn(self):
#         self.spatial = torch.fft.ifftn(self.spectral).real
#         return self.spatial

#     def to(self, device):
#         self.device = device
#         self.spatial = self.spatial.to(device)
#         self.spectral = self.spectral.to(device)
#         return self

# class DynamicField(Field):
#     def __init__(self, name, shape, device="cuda", dtype=torch.float32):
#         super().__init__(name, shape, device, dtype)
#         self.is_dynamic = True
#         self.L_hat = None

#     def set_L_hat(self, L_hat):
#         self.L_hat = L_hat

# class StaticField(Field):
#     def __init__(self, name, shape, device="cuda", dtype=torch.float32):
#         super().__init__(name, shape, device, dtype)
#         self.is_dynamic = False



class Fields:
    def __init__(self, shape, device="cuda", dtype=torch.float32):
        """
        field_names: list of str, e.g., ['u', 'v']
        shape: tuple of length 1, 2, or 3 (e.g., (Nx,), (Nx, Ny), (Nx, Ny, Nz))
        """
        self.shape = shape
        self.device = device
        self.dtype = dtype

        self.num_dyn_fields = 0
        self.num_stat_fields = 0
        self.field_names = {}

        self.spatial_name_to_index = {} #{name: i for i, name in enumerate(field_names)}
        self.spectral_name_to_index = {} #{f"{name}.hat": i for i, name in enumerate(field_names)}

        self.dim = len(shape)

        self.spatial = torch.zeros( (2, 0) + shape ) # shape: (dynamics_or_static = 2, number_of_fields = 0, Nx, Ny, Nz)
        self.spectral = self.fftn()

    # def __getitem__(self, key):
    #     """Access a field by name (optionally with '.hat') """
    #     if isinstance(key, str):
    #         if key.endswith('.hat'):
    #             return self.spectral[self.spectral_name_to_index[key]]
    #         else:
    #             return self.spatial[self.spatial_name_to_index[key]]
    #     else:
    #         raise KeyError("Key must be a field name (str), optionally ending with '.hat'")

    # def __setitem__(self, key, value):
    #     """Update a field by name (optionally with '.hat') or index"""
    #     if isinstance(key, str):
    #         if key.endswith('.hat'):
    #             self.spectral[self.spectral_name_to_index[key]] = value
    #         else:
    #             self.spatial[self.spatial_name_to_index[key]] = value
    #     else:
    #         raise KeyError("Key must be field name (str)")

    def fftn(self):
        """Return batched N-dimensional FFT of all fields"""
        return torch.fft.fftn(self.spatial, dim=tuple(range(2, 2 + self.dim)))

    def ifftn(self):
        """Return batched N-dimensional IFFT (real part) of all fields"""
        return torch.fft.ifftn(self.spatial, dim=tuple(range(2, 2 + self.dim))).real

    def to(self, device):
        """Move to a new device"""
        self.device = device
        self.spatial = self.spatial.to(device)
        return self

    def keys(self):
        return self.names

    def values(self):
        return [self[name] for name in self.names]

    def items(self):
        return [(name, self[name]) for name in self.names]
