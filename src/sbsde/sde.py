import math
from enum import Enum

import numpy as np
import torch


class Direction(Enum):
    FORWARD = 1
    BACKWARD = -1


class SDE:
    def __init__(self, p_data, p_prior, device, t0=1e-3, T=1, interval=100):
        self.p_data = p_data
        self.p_prior = p_prior
        self.device = device

        self.t0 = t0
        self.T = T
        self.interval = interval
        self.ts = torch.linspace(t0, T, interval, device=device)
        self.dt = T / interval

    def f(self, t, x):
        pass

    def g(self, t):
        pass

    def marginal_prob(self, t, x_0):
        pass

    def propagate(self, t, x, z, direction=Direction.FORWARD):
        sign = 1 if direction == Direction.FORWARD else -1
        drift = (self.f(t, x) + sign * self.g(t) * z) * self.dt
        diffusion = self.g(t) * torch.randn_like(x) * np.sqrt(self.dt)
        dx_t = drift + diffusion
        return x + sign * dx_t

    def propagate_s_u(self, t, x, u, s=None, direction=Direction.FORWARD):
        sign = 1 if direction == Direction.FORWARD else -1
        if direction == Direction.FORWARD:
            drift = (self.f(t, x) + self.g(t) * u) * self.dt
            diffusion = self.g(t) * torch.randn_like(x) * np.sqrt(self.dt)
            dx_t = drift + diffusion
        else:
            assert s is not None, "s must be provided for backward propagation"
            drift = (self.f(t, x) + self.g(t) * u - self.g(t) ** 2 * s) * self.dt
            diffusion = self.g(t) * torch.randn_like(x) * np.sqrt(self.dt)
            dx_t = drift + diffusion
        return x + sign * dx_t

    def propagate_s_u_2(self, t, x, u, u2=None, s=None, direction=Direction.FORWARD):
        sign = 1 if direction == Direction.FORWARD else -1
        if direction == Direction.FORWARD:
            drift = (self.f(t, x) + self.g(t) ** 2 * u) * self.dt
            diffusion = self.g(t) * torch.randn_like(x) * np.sqrt(self.dt)
            dx_t = drift + diffusion
        else:
            assert s is not None, "s must be provided for backward propagation"
            drift = (self.f(t, x) + self.g(t) ** 2 * u2 - self.g(t) ** 2 * s) * self.dt
            diffusion = self.g(t) * torch.randn_like(x) * np.sqrt(self.dt)
            dx_t = drift + diffusion
        return x + sign * dx_t

    def sample_traj(self, net, direction=Direction.FORWARD, mode="normal"):
        init_dist = self.p_data if direction == Direction.FORWARD else self.p_prior

        x = init_dist.sample().to(self.device)
        xs = torch.zeros(x.shape[0], len(self.ts), x.shape[1], device=self.device)
        zs = torch.zeros_like(xs)

        for i, t in enumerate(self.ts):
            z = net(t, x)
            if mode == "normal":
                x = self.propagate(t, x, z, direction=direction)
            else:
                x = self.propagate_s_u_2(t, x, z, direction=direction)
            xs[:, i] = x
            zs[:, i] = z
        x_term = x
        return xs, zs, x_term

    def sample_marginal(self, n, device):
        indices = (
            (torch.rand(n) * self.interval).round().long().clamp_max(self.interval - 1)
        )
        t = self.ts[indices]
        x_0 = self.p_data.sample(n).to(device)
        mean, std = self.marginal_prob(t, x_0)
        z = torch.randn_like(x_0)
        x_t = mean + std[:, None] * z
        return t, x_t, mean, std, z


class SimpleSDE(SDE):
    def __init__(self, p_data, p_prior, device, eps=1.0, t0=1e-3, T=1, interval=100):
        super().__init__(p_data, p_prior, device, t0, T, interval)
        self.eps = eps

    def f(self, t, x):
        return torch.zeros_like(x)

    def g(self, t):
        return torch.full_like(t, fill_value=self.eps)


class VESDE(SDE):
    def __init__(
        self,
        p_data,
        p_prior,
        device,
        sigma_min=1e-2,
        sigma_max=5.0,
        t0=1e-3,
        T=1,
        interval=100,
    ):
        super().__init__(p_data, p_prior, device, t0, T, interval)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def f(self, t, x):
        return torch.zeros_like(x)

    def g(self, t):
        return (
            self.sigma_min
            * (self.sigma_max / self.sigma_min) ** t
            * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
        )

    def marginal_prob(self, t, x_0):
        mean = x_0
        std = self.sigma_min**2 * (self.sigma_max / self.sigma_min) ** (2 * t)
        return mean, std


class VPSDE(SDE):
    def __init__(
        self,
        p_data,
        p_prior,
        device,
        beta_min=0.1,
        beta_max=20,
        t0=1e-3,
        T=1,
        interval=100,
    ):
        super().__init__(p_data, p_prior, device, t0, T, interval)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def f(self, t, x):
        t = t.squeeze()
        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        beta_t = self.beta_t(t)[:, None]
        return -0.5 * beta_t * x

    def g(self, t):
        return torch.sqrt(self.beta_t(t))

    def marginal_prob(self, t, x):
        beta_int = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        mean = torch.exp(-0.5 * beta_int)[:, None] * x
        std = torch.sqrt(1.0 - torch.exp(-beta_int))
        return mean, std
