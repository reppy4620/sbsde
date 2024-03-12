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

    def propagate(self, t, x, z, direction=Direction.FORWARD):
        sign = 1 if direction == Direction.FORWARD else -1
        dx_t = (self.f(t, x) + sign * self.g(t) * z) * self.dt + self.g(t) * np.sqrt(
            self.dt
        ) * torch.randn_like(x)
        return x + sign * dx_t

    def sample_traj(self, net, direction=Direction.FORWARD):
        init_dist = self.p_data if direction == Direction.FORWARD else self.p_prior

        x = init_dist.sample().to(self.device)
        xs = torch.zeros(x.shape[0], len(self.ts), x.shape[1], device=self.device)
        zs = torch.zeros_like(xs)

        for i, t in enumerate(self.ts):
            z = net(t, x)
            x = self.propagate(t, x, z, direction=direction)
            xs[:, i] = x
            zs[:, i] = z
        x_term = x
        return xs, zs, x_term


class VESDE(SDE):
    def __init__(
        self, p_data, p_prior, device, sigma_min, sigma_max, t0=1e-3, T=1, interval=100
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
