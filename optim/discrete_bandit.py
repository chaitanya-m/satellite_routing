# optim/discrete_bandit.py

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Optional


class DiscreteBanditOptimiser:
    def __init__(self, candidates: torch.Tensor):
        self.candidates = candidates.to(dtype=torch.double)  # (N, d)
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None
        self.model: Optional[SingleTaskGP] = None

        # For Hyperparameter fitting through maximising log marginal likelihood
        # We don't want to maximize marginal log likelihood without enough data... 
        # We can still condition and update the posteriors though
        self.min_fit_points = 10  
        self.num_obs = 0


    def ask(self) -> float:
        # Cold start: no data yet (or model not fit yet)
        if self.X is None or self.Y is None or self.model is None:
            idx = torch.randint(self.candidates.shape[0], (1,))
            return float(self.candidates[idx, 0].item())

        best_f = float(self.Y.max().item())
        ei = LogExpectedImprovement(self.model, best_f=best_f)

        # EI expects input shape (..., q=1, d). For a candidate set: (N, 1, d).
        Xcand = self.candidates.unsqueeze(1)
        values = ei(Xcand).squeeze(-1)  # (N,)
        idx = torch.argmax(values)
        return float(self.candidates[idx, 0].item())

    def tell(self, x: float, y: float) -> None:
        x_t = torch.tensor([[x]], dtype=self.candidates.dtype)
        y_t = torch.tensor([[y]], dtype=self.candidates.dtype)

        if self.X is None:
            self.X = x_t
            self.Y = y_t
        else:
            assert self.X is not None
            assert self.Y is not None
            X_prev = self.X
            Y_prev = self.Y
            self.X = torch.cat([X_prev, x_t], dim=0)
            self.Y = torch.cat([Y_prev, y_t], dim=0)

        self.model = SingleTaskGP(self.X, self.Y)

        self.num_obs += 1
        if self.num_obs >= self.min_fit_points:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        