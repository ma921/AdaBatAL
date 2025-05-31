import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
import gpytorch
from botorch.exceptions.warnings import InputDataWarning

def predict(test_x, model):
    """
    Fast variance inference is made with LOVE via fast_pred_var().
    For accurate variance inference, you can just comment out the part.

    Input:
        - model: gpytorch.models, function of GP model.

    Output:
        - pred.mean; torch.tensor, the predictive mean
        - pred.variance; torch.tensor, the predictive variance
    """
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        try:
            with gpytorch.settings.fast_pred_var():
                pred = model.likelihood(model(test_x))
        except:
            try:
                pred = model.likelihood(model(test_x))
            except:
                warnings.warn("Cholesky failed. Adding more jitter...")
                with gpytorch.settings.cholesky_jitter(float_value=1e-2):
                    pred = model.likelihood(model(test_x))
    return pred.mean, pred.variance

def predict_mean(test_x, model):
    """
    Fast variance inference is made with LOVE via fast_pred_var().
    For accurate variance inference, you can just comment out the part.

    Input:
        - model: gpytorch.models, function of GP model.

    Output:
        - pred.mean; torch.tensor, the predictive mean
        - pred.variance; torch.tensor, the predictive variance
    """
    pred_mean, _ = predict(test_x, model)
    return pred_mean

def get_cov_cache(model):
    """
    woodbury_inv = K(Xobs, Xobs)^(-1)
    S @ S.T = woodbury_inv

    Input:
        - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py

    Output:
        - woodbury_inv: torch.tensor, the inverse of Gram matrix K(Xobs, Xobs)^(-1)
        - Xobs: torch.tensor, the observed inputs X
        - lik_var: torch.tensor, the GP likelihood noise variance
    """
    Xobs = model.train_inputs[0]
    lik_var = model.likelihood.noise
    try:
        S = model.prediction_strategy.covar_cache
    except:
        model.eval()
        mean = Xobs[0].unsqueeze(0)
        model(mean)
        S = model.prediction_strategy.covar_cache
    woodbury_inv = S @ S.T
    return woodbury_inv, Xobs, lik_var


def predictive_covariance(x, y, model):
    """
    Input:
        - x: torch.tensor, inputs x
        - y: torch.tensor, inputs y
        - model: gpytorch.models, function of GP model.

    Output:
        - cov_xy: torch.tensor, predictive covariance matrix
    """
    woodbury_inv, Xobs, lik_var = get_cov_cache(model)
    Kxy = model.covar_module.forward(x, y)
    KxX = model.covar_module.forward(x, Xobs)
    KXy = model.covar_module.forward(Xobs, y)
    cov_xy = Kxy - KxX @ woodbury_inv @ KXy

    d = min(len(x), len(y))
    cov_xy[range(d), range(d)] = cov_xy[range(d), range(d)] + lik_var
    return cov_xy

def normalise(y, Y):
    return (y - Y.mean()) / Y.std()

def set_gp(X, Y):
    train_X = X.view(-1).unsqueeze(-1)
    train_Y = normalise(Y, Y).view(-1).unsqueeze(-1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InputDataWarning)
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
    return gp