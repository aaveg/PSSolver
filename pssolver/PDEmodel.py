import torch
from .Field import Fields, Parameters
import inspect


class PDEModel:
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device
        self.fields = Fields(shape = shape, device = device)
        self.parameters = Parameters()
        self.num_fields = 0
        self.dyn_fields = []
        self.stat_fields = []
        self.nlmodel = None
        self.static_model = None


    def add_dynamic_field(self, name, init, L_hat):
        if init.shape != self.shape:
            raise ValueError(f"Initial value for field '{name}' must have shape {self.shape}, got {init.shape}")
        self.dyn_fields.append([name, init, L_hat])

    def add_static_field(self, name):
        self.stat_fields.append([name])  

    def set_nonlinear_model(self, model):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Nonlinear model must be a torch.nn.Module")
        self.nlmodel = model

    def set_static_compute_model(self, model):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Static compute model must be a torch.nn.Module")
        self.static_model = model

    def build(self):
        if not self.dyn_fields:
            raise ValueError("No dynamic fields added. Add at least one dynamic field.")
        
        # Check for duplicate names
        all_names = [entry[0] for entry in self.dyn_fields + self.stat_fields]
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate field names detected")

        count = 0
        inits = []
        L_hats = []
        for entry in self.dyn_fields:
            self.fields.name_to_idx[entry[0]] = count 
            count += 1
            inits.append(entry[1])
            L_hats.append(entry[2])
        
        self.fields.dyn_count = count
 
        for entry in self.stat_fields:
            self.fields.name_to_idx[entry[0]] = count 
            count += 1
            inits.append(torch.zeros(self.shape))
        self.fields.stat_count = count - self.fields.dyn_count

        self.fields.spatial = torch.stack(inits).to(self.device)
        self.fields.spectral = self.fields.fftn()
        self.fields.L_hat = torch.stack(L_hats).to(self.device)

        if self.nlmodel is None:
            self.nlmodel = ZeroModel()
        else:
            # Validate nonlinear model
            try:
                sig = inspect.signature(self.nlmodel.forward)
                if len(sig.parameters) != 2:  # fields, parameters
                    raise TypeError("Nonlinear model's forward method must accept two input parameters: fields and parameters")
                test_output = self.nlmodel(self.fields, self.parameters)
                expected_shape = (self.fields.dyn_count, *self.shape)
                if test_output.shape != expected_shape:
                    raise ValueError(f"Nonlinear model output shape {test_output.shape} doesn't match expected {expected_shape}")
            except Exception as e:
                raise RuntimeError("Error occurred during nonlinear model validation.") from e


        if self.static_model is None:
            self.static_model = ZeroModel()
        else:
            # Validate static model
            try:
                sig = inspect.signature(self.static_model.forward)
                if len(sig.parameters) != 2:  # fields, parameters
                    raise TypeError("Static model's forward method must accept two input parameters: fields and parameters")
                test_output = self.static_model(self.fields, self.parameters)
                expected_shape = (self.fields.stat_count, *self.shape)
                if test_output.shape != expected_shape:
                    raise ValueError(f"Static model output shape {test_output.shape} doesn't match expected {expected_shape}")
            except Exception as e:
                raise RuntimeError("Error occurred during static model validation.") from e
        

        
    
    def get_field_order(self):
        """Return the order of dynamic fields for reference when writing nonlinear models."""
        return [entry[0] for entry in self.dyn_fields]

    def compute_static(self):
        return self.static_model(self.fields, self.parameters)

    def compute_nonlinear(self):
        return self.nlmodel(self.fields, self.parameters)


class ZeroModel(torch.nn.Module):
    def forward(self, fields,params):
        return 0
