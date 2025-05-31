import torch
from utils._rchq import recombination
from utils._rchq_lp import lp_recombination
from utils._kernel import Kernel
from fast_pytorch_kmeans import KMeans


def landmark_points(X_cand, n_nys):
    kmeans = KMeans(n_clusters=n_nys, mode='euclidean')
    kmeans.fit(X_cand.view(-1).unsqueeze(-1))
    X_nys = kmeans.centroids
    return X_nys

def batch_query(gp, X_cand, X_nys, batch_size=5, tolerance=1e-8, fixed_batch=True):
    kernel = Kernel(gp)
    pred_post = gp.posterior(X_cand)
    mean, stddev = pred_post.mean.detach().flatten(), pred_post.stddev.detach().flatten()
    w_uncertainty = (mean - mean.min()) * stddev
    if fixed_batch:
        idx_kq, w_kq = recombination(
            X_cand,          # random samples for recombination
            X_nys,           # number of samples used for approximating kernel with Nystrom method
            batch_size,      # number of samples finally returned
            kernel,          # kernel
            X_cand.device,
            X_cand.dtype,
            #init_weights=w_uncertainty,
        )
    else:
        idx_kq, w_kq = lp_recombination(
            X_cand,             # random samples for recombination
            X_nys,              # number of samples used for approximating kernel with Nystrom method
            batch_size,         # number of samples finally returned
            kernel,             # kernel
            X_cand.device,      # device
            X_cand.dtype,       # data type; torch.float or torch.double. We recommend torch.double for precision
            ep_lp=tolerance,
            init_weights=w_uncertainty,
        )
    return X_cand[idx_kq]

def query_and_update(X, Y, X_new, true_func):
    X_new = X_new.view(-1).unsqueeze(-1)
    Y_new = true_func(X_new).unsqueeze(-1)
    X = torch.vstack([X, X_new])
    Y = torch.vstack([Y, Y_new])
    return X, Y, X_new, Y_new
