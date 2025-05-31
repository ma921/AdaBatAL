import torch

def berkenkamp_1d(x):
    x = x.squeeze()
    y1 = 0.6 * x
    y2 = torch.distributions.Normal(0.2, 0.08).log_prob(x).exp() / 8
    y = torch.atleast_1d(y1 + y2)
    return y