import torch
from Field import Fields

class PDEModel:
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device

        self.fields = Fields(shape = shape, device = device)
        self.dynamic_fields= self.fields[0]
        self.passive_fields= self.fields[1]
        # self.fields = {}
        self.nl_eqns = {}
        self.passive_compute_fns={}


    def add_dynamic_field(self, name, init, L_hat, N_hat):
        if name in self.fields.field_names:
            raise ValueError(f"Field '{name}' already exists.")

        self.fields.field_names{name} = (0,self.fields.num_dyn_fields)
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
