import numpy as np
import torch
import torch.distributions as td


class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=12, num=8, sigmas=None, device="cpu"):
        arc = 2 * np.pi / num
        xs = [np.cos(arc * idx) * radius for idx in range(num)]
        ys = [np.sin(arc * idx) * radius for idx in range(num)]
        mus = [torch.tensor([x, y], device=device) for x, y in zip(xs, ys)]
        dim = len(mus[0])
        sigmas = (
            [torch.eye(dim, device=device) for _ in range(num)]
            if sigmas is None
            else sigmas
        )

        if batch_size % num != 0:
            raise ValueError("batch size must be devided by number of gaussian")
        self.num = num
        self.batch_size = batch_size
        self.dists = []
        for mu, sigma in zip(mus, sigmas):
            dist = td.multivariate_normal.MultivariateNormal(mu, sigma)
            dist._unbroadcasted_scale_tril = dist._unbroadcasted_scale_tril.double()
            self.dists.append(dist)

    def log_prob(self, x):
        # assume equally-weighted
        densities = [torch.exp(dist.log_prob(x.double())) for dist in self.dists]
        return torch.log(sum(densities) / len(self.dists))

    def sample(self, n=None):
        n = self.batch_size if n is None else n
        ind_sample = n / self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        samples = torch.cat(samples, dim=0)
        return samples.float()

    def sample_n(self, n):
        ind_sample = n / self.num
        samples = [dist.sample([int(ind_sample)]) for dist in self.dists]
        samples = torch.cat(samples, dim=0)
        return samples.float()


def PriorNormal(sigma_max, device):
    cov_coef = sigma_max**2
    return td.MultivariateNormal(
        torch.zeros(2, device=device), cov_coef * torch.eye(2, device=device)
    )
