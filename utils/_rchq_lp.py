import torch
import numpy as np
import gurobipy as gp
from ._utils import TensorManager


def lp_recombination(
    pts_rec,         # random samples for recombination
    pts_nys,         # number of samples used for approximating kernel with Nystrom method
    num_pts,         # number of samples finally returned
    kernel,          # kernel
    device,          # device
    dtype,           # data type; torch.float or torch.double. We recommend torch.double for precision
    ratio=2,           # ratio in recombination: if set to be None, it runs a full LP
    init_weights=None,  # initial weights of the sample for recombination
    ep_lp=1e-8,
    ep_torch=torch.finfo().eps,
    calc_obj=None,   # a function for calculating an additional objective function
    calc_prob=None,  # a function for calculating the acceptance probability
):
    """
    Args:
        - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
        - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
        - num_pts: int, number of samples finally returned. In BASQ context, this is equivalent to batch size
        - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)
        - device: torch.device, cpu or cuda
        - init_weights: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        - calc_obj: a function that returns a Tensor of objective values for all the input points

    Returns:
        - x: torch.tensor, the sparcified samples from pts_rec. The number of samples are determined by self.batch_size
        - w: torch.tensor, the positive weights for kernel quadrature as discretised summation.
    """
    #print(ep_lp, 1)
    tm = TensorManager(device=device, dtype=dtype)
    return rc_kernel_svd(pts_rec, pts_nys, num_pts, kernel, tm, ratio, mu=init_weights, ep_lp=ep_lp, ep_torch=ep_torch, calc_obj=calc_obj, calc_prob=calc_prob)

def ker_svd_sparsify(pt, s, kernel):
    _U, S, _ = torch.svd_lowrank(kernel(pt, pt), q=s)
    U = -1 * _U.T  # Hermitian
    return S, U


def rc_kernel_svd(samp, pt, s, kernel, tm, ratio, mu=None, ep_lp=1e-8, ep_torch=1e-8, calc_obj=None, calc_prob=None):
    #print(ep_lp, 2)
    num_eigs = s - 1 - int(calc_prob is not None)
    # Nystrom method
    eigs, U = ker_svd_sparsify(pt, num_eigs, kernel)
    w_star, idx_star = Mod_Tchernychova_Lyons(
        samp, eigs, U, pt, kernel, tm, ratio, mu=mu, ep_lp=ep_lp, ep_torch=ep_torch, calc_obj=calc_obj, calc_prob=calc_prob
    )
    return idx_star, w_star


def Mod_Tchernychova_Lyons(samp, eigs, U_svd, pt_nys, kernel, tm, ratio=2, mu=None, ep_lp=1e-8, ep_torch=1e-8, calc_obj=None, calc_prob=None, DEBUG=False):
    """
    This function is a modified Tcherynychova_Lyons from
    https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py
    """
    N = len(samp)
    n, length = U_svd.shape
    if ratio is None:
        number_of_sets = N
    else:
        number_of_sets = min(N, max(ratio, 2) * (n + 1))

    # obj = torch.zeros(N).to(device)
    if mu is None:
        mu = tm.ones(N) / N

    idx_story = tm.arange(N)
    idx_story = idx_story[mu != 0]
    remaining_points = len(idx_story)

    use_prob = False if calc_prob is None else True
    if use_prob:
        prob = calc_prob(samp)
    use_obj = False if calc_obj is None else True
    if use_obj:
        obj = calc_obj(samp)

    while True:
        if remaining_points <= n + 1:
            idx_star = tm.arange(len(mu))[mu > 0]
            w_star = mu[idx_star]
            return w_star, idx_star

        elif n + 1 < remaining_points <= number_of_sets:
            X_mat = U_svd @ kernel(pt_nys, samp[idx_story])
            if use_prob:
                X_mat = torch.cat((X_mat, torch.reshape(
                    prob[idx_story], (1, -1))), 0)
            if use_obj:
                X_mat = torch.cat((X_mat, torch.reshape(
                    obj[idx_story], (1, -1))), 0)
            w_star, idx_star = LP(X_mat, torch.clone(
                mu[idx_story]), eigs, tm, ep_lp=ep_lp, ep_torch=ep_torch, use_obj=use_obj, use_prob=use_prob)

            idx_story = idx_story[idx_star]
            mu[:] = 0.
            mu[idx_story] = w_star
            idx_star = idx_story
            w_star = mu[mu > 0]

            return w_star, idx_star

        number_of_el = int(remaining_points / number_of_sets)

        idx = idx_story[:number_of_el *
                        number_of_sets].reshape(number_of_el, -1)
        X_for_nys = tm.zeros((length, number_of_sets))
        X_for_prob = tm.zeros((1, number_of_sets))
        X_for_obj = tm.zeros((1, number_of_sets))
        for i in range(number_of_el):
            idx_tmp_i = idx_story[i * number_of_sets:(i + 1) * number_of_sets]
            X_for_nys += torch.multiply(
                kernel(pt_nys, samp[idx_tmp_i]),
                mu[idx_tmp_i].unsqueeze(0)
            )
            if use_prob:
                X_for_prob += torch.multiply(
                    torch.reshape(prob[idx_tmp_i], (1, -1)), mu[idx_tmp_i].unsqueeze(0))
            if use_obj:
                X_for_obj += torch.multiply(
                    torch.reshape(obj[idx_tmp_i], (1, -1)), mu[idx_tmp_i].unsqueeze(0))

        X_tmp_tr = U_svd @ X_for_nys
        if use_prob:
            X_tmp_tr = torch.cat((X_tmp_tr, X_for_prob), 0)
        if use_obj:
            X_tmp_tr = torch.cat((X_tmp_tr, X_for_obj), 0)
        X_tmp = X_tmp_tr.T
        tot_weights = torch.sum(mu[idx], 0)
        idx_last_part = idx_story[number_of_el * number_of_sets:]

        if len(idx_last_part):
            X_mat = U_svd @ kernel(pt_nys, samp[idx_last_part])
            if use_prob:
                X_mat = torch.cat((X_mat, torch.reshape(
                    prob[idx_last_part], (1, -1))), 0)
            if use_obj:
                X_mat = torch.cat((X_mat, torch.reshape(
                    obj[idx_last_part], (1, -1))), 0)
            X_tmp[-1] += torch.multiply(
                X_mat.T,
                mu[idx_last_part].unsqueeze(1)
            ).sum(axis=0)
            tot_weights[-1] += torch.sum(mu[idx_last_part], 0)

        X_tmp = torch.divide(X_tmp, tot_weights.unsqueeze(0).T)

        # # sparsify for the case use_obj is True
        # if use_obj:
        #     X_tmp_raw = torch.clone(X_tmp[:, :n])
        #     obj_raw = X_tmp[:, -1:].reshape(-1)

        # w_star, idx_star, _, _, ERR, _, _ = Tchernychova_Lyons_CAR(
        #     X_tmp, torch.clone(tot_weights), device
        # )
        w_star, idx_star = LP(X_tmp.T, torch.clone(
            tot_weights), eigs, tm, ep_lp=ep_lp, ep_torch=ep_torch, use_obj=use_obj, use_prob=use_prob)

        idx_tomaintain = idx[:, idx_star].reshape(-1)
        idx_tocancel = tm.ones(idx.shape[1]).to(torch.bool)
        idx_tocancel[idx_star] = 0
        idx_tocancel = idx[:, idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = torch.multiply(mu[idx[:, idx_star]], w_star)
        mu_tmp = torch.divide(mu_tmp, tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)

        idx_tmp = idx_star == number_of_sets - 1
        idx_tmp = tm.arange(len(idx_tmp))[idx_tmp != 0]
        # if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp) > 0:
            mu_tmp = torch.multiply(mu[idx_last_part], w_star[idx_tmp])
            mu_tmp = torch.divide(mu_tmp, tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp
            idx_tomaintain = torch.cat([idx_tomaintain, idx_last_part])
        else:
            idx_tocancel = torch.cat([idx_tocancel, idx_last_part])
            mu[idx_last_part] = 0.

        idx_story = torch.clone(idx_tomaintain)
        remaining_points = len(idx_story)


def LP(K_, mu_, eigs_, tm, ep_lp=1e-5, ep_torch=1e-8, use_obj=True, use_prob=True):
    # We solve an LP, which is given as follows when use_obj & use_prob is True:
    # maximize K[-1] @ x (objective term)
    # subject to |K[:-2] @ x - K[:-2] @ mu| <= ep_lp * eigs (coordinate-wise)
    #            K[-2] @ x >= K[-2] @ mu (probability term)
    #            x >= 0, |1^T @ x - 1| <= ep_torch (accepting numerical errors)
    #
    # 'ep_lp' corresponds to \epislon_{LP} in the paper
    # 'ep_torch' corresponds to an acceptable numerical error to avoid infeasibility
    #
    # If you want to integrate test functions "exactly",
    # we recommend using ep_lp = ep_torch = 1e-8 or similar, depending on the precision you use

    K = K_.cpu().detach().numpy()
    mu = mu_.cpu().detach().numpy()
    eigs = eigs_.cpu().detach().numpy()
    m, n = K.shape
    mu = mu.reshape((n, 1))
    B = K @ mu
    dum = np.ones((1, n))

    num_final = int(use_obj) + int(use_prob)
    B_er = np.sqrt(eigs / max(1, m - num_final)) * ep_lp
    B_er = np.maximum(B_er, np.ones(m - num_final) * ep_torch)

    model = gp.Model("test")
    model.setParam('OutputFlag', 0)
    model.Params.LogToConsole = 0
    x = model.addMVar(n)
    model.update()

    if use_obj:
        model.setObjective(- K[m - 1, :] @ x)
        # To maximize K[m-1,:] @ x
    else:
        model.setObjective(0)
    if use_prob:
        model.addConstr(K[m - num_final] @ x >= K[m - num_final] @ mu)

    for i in range(m - num_final):
        model.addConstr(K[i, :] @ x >= B[i] - B_er[i])
        model.addConstr(K[i, :] @ x <= B[i] + B_er[i])

    model.addConstr(dum @ x >= dum @ mu - ep_torch)
    model.addConstr(dum @ x <= dum @ mu + ep_torch)
    model.addConstr(x >= 0)

    model.optimize()
    idx_star = []
    w_star = []
    if model.Status == gp.GRB.OPTIMAL:
        for i in range(n):
            if x[i].X > 0:
                idx_star.append(i)
                w_star.append(x[i].X)
    else:
        print("FAILED")
    w_star = tm.from_numpy(np.reshape(w_star, -1))
    idx_star = tm.from_numpy(np.reshape(idx_star, -1)).to(torch.long)
    return w_star, idx_star
