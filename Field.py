import torch


class Fields:
    def __init__(self, shape, device="cuda", dtype=torch.float32):
        """
        field_names: list of str, e.g., ['u', 'v']
        shape: tuple of length 1, 2, or 3 (e.g., (Nx,), (Nx, Ny), (Nx, Ny, Nz))
        """
        self.shape = shape
        self.device = device
        self.dtype = dtype

        self.qx = None
        self.qy = None 
        self.qz = None
        self.q2 = None

        self.dim = len(shape)

        self.name_to_idx = {}
        self.dyn_count = 0
        self.stat_count = 0

        self.spatial = torch.zeros((0,) + shape, device=self.device, dtype=self.dtype)  # shape: (number_of_fields = 0, Nx, Ny, Nz)
        self.L_hat   = torch.zeros((0,) + shape, device=self.device, dtype=self.dtype)  # shape: (number_of_fields = 0, Nx, Ny, Nz) -> L_hat for static fields =0 (done for speed optimization)
        self.spectral = torch.zeros((0,) + shape, device=self.device, dtype=self.dtype)
        self.dealiasing_mask = self.create_dealiasing_mask(shape)

    # Create a dealiasing mask for the given shape (2/3 rule)
    def create_dealiasing_mask(self, shape):
        mask = torch.ones(shape, dtype=torch.bool, device=self.device)
        for axis, n in enumerate(shape):
            cutoff = int(n * 1 / 3)
            idx = [slice(None)] * self.dim
            # Set False for indices outside the central 2/3 region
            if cutoff > 0:
                idx_low = idx.copy()
                idx_low[axis] = slice(0, cutoff)
                mask[tuple(idx_low)] = False

                idx_high = idx.copy()
                idx_high[axis] = slice(n - cutoff, n)
                mask[tuple(idx_high)] = False
        return mask

    
    def set_wavenumbers(self, qx=None, qy=None, qz=None, q2=None):
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.q2 = q2

    def __getitem__(self, key):
        """Access a field by name (optionally with '.hat') """
        if isinstance(key, str):
            if key.endswith('.hat'):
                return self.spectral[self.name_to_idx[key[:-4]]]
            else:
                return self.spatial[self.name_to_idx[key]]
        else:
            raise KeyError("Key must be a field name (str), optionally ending with '.hat'")


    def fftn(self):
        """Return batched N-dimensional FFT of all fields"""
        return torch.fft.fftn(self.spatial, dim=tuple(range(1, 1 + self.dim)))

    def ifftn(self):
        """Return batched N-dimensional IFFT (real part) of all fields"""
        return torch.fft.ifftn(self.spectral, dim=tuple(range(1, 1 + self.dim))).real


    def keys(self):
        return list(self.name_to_idx.keys())

    def values(self):
        return [self[name] for name in self.name_to_idx.keys()]

    def items(self):
        return [(name, self[name]) for name in self.name_to_idx.keys()]
