import torch


def compute_div_gz_f(sde, net_b, ts, xs):
    zs = net_b(ts, xs)

    g_ts = sde.g(ts)
    g_ts = g_ts[:, None]
    g_zs = g_ts * zs
    g_zs_f = g_zs - sde.f(ts, xs)

    eps = torch.randn_like(xs)
    e_dz_dx = torch.autograd.grad(g_zs_f, xs, eps, create_graph=True)[0]
    div_gz_f = e_dz_dx * eps
    return div_gz_f, zs


def compute_loss(sde, net_b, ts, xs, x_term, zs_f):
    div_gz_f, zs_b = compute_div_gz_f(sde, net_b, ts, xs)

    integrand = 0.5 * (zs_f + zs_b) ** 2 + div_gz_f
    E_integrand = integrand.reshape(
        x_term.shape[0], sde.interval, x_term.shape[-1]
    ).mean(dim=0)
    loss = -sde.p_prior.log_prob(x_term).mean() + (E_integrand * sde.dt).sum()
    return loss


def compute_div(hs, xs):
    eps = torch.randn_like(xs)
    e_do_dx = torch.autograd.grad(hs, xs, eps, create_graph=True)[0]
    div = e_do_dx * eps
    return div


def compute_loss_s_u(sde, net_s, ts, xs, x_term, us):
    g = sde.g(ts)[:, None]
    g_sq = g**2
    s = net_s(ts, xs)
    u = us
    f = sde.f(ts, xs)

    div_term = compute_div((g_sq * s) * (g * u) * f, xs)
    integrand = (g_sq / 2) * (s**2) + div_term
    E_integrand = integrand.reshape(
        x_term.shape[0], sde.interval, x_term.shape[-1]
    ).mean(dim=0)
    loss = -sde.p_prior.log_prob(x_term).mean() + (E_integrand * sde.dt).sum()
    return loss
