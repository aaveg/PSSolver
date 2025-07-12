import torch
from Field import Fields

class PDEModel:
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device

        self.fields = Fields(shape = shape, device = device)
        # self.dyn_name_to_idx = {}
        # self.dyn_count = 0 
        # self.dyn_inits = []
        # self.dyn_L_hat = []
        self.nl_eqns = []

        # self.stat_name_to_idx = {}
        # self.stat_count = 0 
        self.stat_compute_fns = []


    def add_dynamic_field(self, name, init, L_hat, N_hat):
        self.fields.name_to_idx[name] = self.fields.field_count 
        self.fields.dyn_idx.append(self.fields.field_count)
        self.fields.field_count+=1
        self.fields.spatial = torch.cat([self.fields.spatial, init.unsqueeze(0).to(self.device)], dim=0)
        self.fields.spectral = self.fields.fftn()
        self.fields.L_hat = torch.cat([self.fields.L_hat, L_hat.unsqueeze(0).to(self.device)], dim=0)
        self.nl_eqns.append(N_hat)

    def add_static_field(self, name, compute_fn):
        self.fields.name_to_idx[name] = self.fields.field_count 
        self.fields.stat_idx.append(self.fields.field_count)
        self.fields.field_count+=1
        self.fields.spatial = torch.cat([self.fields.spatial, torch.zeros(self.shape).unsqueeze(0)], dim=0)
        self.stat_compute_fns.append(compute_fn)

    def compute_static(self):
        outputs = [fn(self.fields) for fn in self.stat_compute_fns]
        return torch.stack(outputs)  

    def compute_nonlinear(self):
        outputs = [fn(self.fields) for fn in self.nl_eqns]
        return torch.stack(outputs)  

    # def add_dynamic_field(self, name, init, L_hat, N_hat):
    #     if (name in self.dyn_name_to_idx) or (name in self.stat_name_to_idx) :
    #         raise ValueError(f"Field name '{name}' already exists.")
    #     if init.shape != self.shape:
    #         raise ValueError(f"Initial condition for '{name}' has shape {init.shape}, expected {self.shape}.")
    #     if L_hat.shape != self.shape:
    #         raise ValueError(f"L_hat for '{name}' has shape {L_hat.shape}, expected {self.shape}.")
        
    #     self.dyn_name_to_idx[name] = self.dyn_count 
    #     self.dyn_count += 1
    #     self.dyn_inits.append(init)
    #     self.dyn_L_hat.append(L_hat)

    #     self.nl_eqns[name] = N_hat


    # def add_passive_field(self, name, compute_fn):
    #     if (name in self.dyn_name_to_idx) or (name in self.stat_name_to_idx) :
    #         raise ValueError(f"Field name '{name}' already exists.")

    #     self.stat_name_to_idx[name] = self.stat_count 
    #     self.stat_count += 1
    #     self.stat_compute_fns[name] = compute_fn

    # def finalize(self):
    #     self.stat_idx_offset = self.dyn_count
    #     self.all_inits = torch.stack(
    #         [torch.stack(self.dyn_inits)] +
    #         [torch.zeros((self.stat_count, *self.shape), device=self.device)],
    #     ).to(self.device)

    # def compute_nonlinear(self):
    #     results = {}
    #     for name, eq in self.nl_eqns.items():
    #         results[name] = eq(self.fields)  

    #     return results

    # def update_stat_fields(self):
    #     for name, compute_fn in self.passive_compute_fns.items():
    #         self.fields[name].f = compute_fn(self.fields)
