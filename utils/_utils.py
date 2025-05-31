import torch
import warnings
from torch.quasirandom import SobolEngine
from torch.distributions.multivariate_normal import MultivariateNormal


def device_manager(device=None):
    if device == None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return device

def dtype_manager(dtype=None):
    if dtype == None:
        dtype = torch.float
    return dtype

class TensorManager:
    def __init__(self, device=None, dtype=None):
        self.device = device_manager(device=device)
        self.dtype = dtype_manager(dtype=dtype)
        
    def standardise_tensor(self, tensor):
        return tensor.to(self.device, self.dtype)
    
    def standardise_device(self, tensor):
        return tensor.to(self.device)
    
    def ones(self, n_samples, n_dims=None):
        if n_dims == None:
            return self.standardise_tensor(torch.ones(n_samples))
        else:
            return self.standardise_tensor(torch.ones(n_samples, n_dims))
    
    def zeros(self, n_samples, n_dims=None):
        if n_dims == None:
            return self.standardise_tensor(torch.zeros(n_samples))
        else:
            return self.standardise_tensor(torch.zeros(n_samples, n_dims))
    
    def rand(self, n_dims, n_samples, qmc=True):
        if qmc:
            random_samples = SobolEngine(n_dims, scramble=True).draw(n_samples)
        else:
            random_samples = torch.rand(n_samples, n_dims)
        return self.standardise_tensor(random_samples)
    
    def arange(self, length):
        return self.standardise_device(torch.arange(length))
    
    def null(self):
        return self.standardise_device(torch.tensor([]))
    
    def tensor(self, x):
        return self.standardise_tensor(torch.tensor(x))
    
    def randperm(self, length):
        return self.standardise_device(torch.randperm(length))
    
    def multinomial(self, weights, n):
        return self.standardise_device(torch.multinomial(weights, n))
    
    def numpy(self, x):
        return x.detach().cpu().numpy()
    
    def from_numpy(self, x):
        return self.standardise_tensor(torch.from_numpy(x))
    
    def is_cuda(self):
        if self.device == torch.device('cuda'):
            return True
        else:
            return False
