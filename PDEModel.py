import torch
from Field import Field

class PDEModel:
    def __init__(self, N, device):
        self.N = N
        self.device = device
        self.dynamic_fields={}
        self.fields = {}
        self.nl_eqns = {}
        self.passive_compute_fns={}


    def add_active_field(self, name, init, L_hat, N_hat):
        if name in self.fields:
            raise ValueError(f"Field '{name}' already exists.")

        dynamic_field = Field(name, self.N, dynamic = True, device = self.device)
        dynamic_field.set_initial_condition(init)
        dynamic_field.set_L_hat(L_hat)
        self.fields[name] = dynamic_field
        self.dynamic_fields[name] = dynamic_field
        self.nl_eqns[name] = N_hat


    def add_passive_field(self, name, compute_fn):
        if name in self.fields:
            raise ValueError(f"Field '{name}' already exists.")

        passive_field = Field(name, self.N, dynamic = False, device = self.device)
        self.fields[name] = passive_field
        self.passive_compute_fns[name] = compute_fn

    def compute_nonlinear(self):
        results = {}
        for name, eq in self.nl_eqns.items():
            results[name] = eq(self.fields)  

        return results

    def update_passive_fields(self):
        for name, compute_fn in self.passive_compute_fns.items():
            self.fields[name].f = compute_fn(self.fields)
