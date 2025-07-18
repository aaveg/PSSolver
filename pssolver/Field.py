import torch

class Parameters:
    def __init__(self):
        self._params = {}

    def set_param(self, name, value):
        """Set a parameter by name, avoiding duplicates."""
        if name in self._params:
            raise KeyError(f"Parameter '{name}' already exists.")
        self._params[name] = value

    def update_param(self, name, value):
        """Update an existing parameter by name."""
        if name in self._params:
            self._params[name] = value
        else:
            raise KeyError(f"Parameter '{name}' does not exist.")

    def get_param(self, name):
        """Get a parameter by name."""
        return self[name]

    def __getitem__(self, key):
        """Return the value for the given key from internal parameters."""
        if key not in self._params:
            raise KeyError(f"Parameter '{key}' does not exist.")
        return self._params[key]
        
    def keys(self):
        return list(self._params.keys())

    def values(self):
        return list(self._params.values())

    def items(self):
        return list(self._params.items())


class Fields:
    def __init__(self, shape, device="cuda", dtype=torch.float32, batch_size = 1):
        """
        field_names: list of str, e.g., ['u', 'v']
        shape: tuple of length 1, 2, or 3 (e.g., (Nx,), (Nx, Ny), (Nx, Ny, Nz))
        """
        self.shape = shape
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self.qx = None
        self.qy = None 
        self.qz = None
        self.q2 = None

        self.dim = len(shape)

        self.name_to_idx = {}
        self.dyn_count = 0
        self.stat_count = 0

        self.spatial  = None  # shape: (Batch, number_of_fields, Nx, Ny, Nz)
        self.L_hat    = None  # shape: (Batch, number_of_dynamic_fields, Nx, Ny, Nz)
        self.spectral = None 

    
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
        return torch.fft.fftn(self.spatial, dim=tuple(range(2, 2 + self.dim)))

    def ifftn(self):
        """Return batched N-dimensional IFFT (real part) of all fields"""
        return torch.fft.ifftn(self.spectral, dim=tuple(range(2, 2 + self.dim))).real


    def keys(self):
        return list(self.name_to_idx.keys())

    def values(self):
        return [self[name] for name in self.name_to_idx.keys()]

    def items(self):
        return [(name, self[name]) for name in self.name_to_idx.keys()]
