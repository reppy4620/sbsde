from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.optim as optim
from sbsde.distribution import MixMultiVariateNormal, PriorNormal
from sbsde.loss import compute_loss_s_u, compute_loss_score
from sbsde.model import ToyModel
from sbsde.sde import VESDE, VPSDE, Direction
from sbsde.utils import flatten_dim01
from tqdm.auto import trange

batch_size = 200
prior_sigma = 1.0
num_iter = 5000
sample_size = 1000


class Mode:
    SCORE = "score"
    SB = "SB"


class ZeroNet(torch.nn.Module):
    def forward(self, t, x):
        return torch.zeros_like(x)


def main(interval, sde_cls, mode):
    print(interval, sde_cls, mode)
    out_dir = Path(f"out/{interval}-{sde_cls.__name__}-{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_data = MixMultiVariateNormal(batch_size)
    p_prior = PriorNormal(prior_sigma, device)
    sde = sde_cls(p_data, p_prior, device, interval=interval)
    net = ToyModel().to(device)
    zero_net = ZeroNet().to(device)

    optimizer = optim.AdamW(net.parameters(), lr=1e-4)

    net.train()
    losses = []
    bar = trange(num_iter)
    for _ in bar:
        optimizer.zero_grad()
        if mode == Mode.SCORE:
            t, x_t, _, std, z = sde.sample_marginal(batch_size, device)
            score = net(t, x_t)
            loss = torch.mean((score * std[:, None] + z) ** 2)
        else:
            xs, _, x_term_f = sde.sample_traj(zero_net)
            _ts = sde.ts.repeat(batch_size)
            xs = flatten_dim01(xs)
            xs.requires_grad_(True)
            loss = compute_loss_score(
                sde=sde,
                net=net,
                ts=_ts,
                xs=xs,
                x_term=x_term_f,
            )

        loss.backward()
        optimizer.step()
        bar.set_postfix(loss=loss.item())
        losses.append(loss.item())

    with open(out_dir / "loss.txt", "w") as f:
        for loss in losses:
            f.write(f"{loss}\n")

    plt.plot(range(1, num_iter + 1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.title("Training Loss")
    plt.savefig(out_dir / "loss.png")
    plt.close()

    torch.save(
        {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        out_dir / "ckpt.pth",
    )

    n = sample_size
    x_data = sde.p_data.sample_n(n).to(device)
    x = x_data
    for t in sde.ts:
        with torch.no_grad():
            u = zero_net(t, x)
            x = sde.propagate_s_u(t, x, u, direction=Direction.FORWARD)
    x_T = x.detach().cpu().numpy()
    x_prior = sde.p_prior.sample_n(n)
    x = x_prior
    for t in reversed(sde.ts):
        with torch.no_grad():
            u = zero_net(t, x)
            s = net(t, x)
            x = sde.propagate_s_u(t, x, u, s, direction=Direction.BACKWARD)
    x_0 = x.detach().cpu().numpy()

    x_data = x_data.detach().cpu().numpy()
    x_prior = x_prior.detach().cpu().numpy()

    _, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[0, 0].plot(x_data[:, 0], x_data[:, 1], "o")
    axes[0, 0].set_title("p_data")
    axes[0, 1].plot(x_T[:, 0], x_T[:, 1], "o")
    axes[0, 1].set_title("p_T")
    axes[0, 2].plot(x_prior[:, 0], x_prior[:, 1], "o")
    axes[0, 2].set_title("p_prior")

    axes[1, 0].plot(x_prior[:, 0], x_prior[:, 1], "o")
    axes[1, 0].set_title("p_prior")
    axes[1, 1].plot(x_0[:, 0], x_0[:, 1], "o")
    axes[1, 1].set_title("p_0")
    axes[1, 2].plot(x_data[:, 0], x_data[:, 1], "o")
    axes[1, 2].set_title("p_data")
    for i in range(2):
        for j in range(3):
            axes[i, j].set_xlim(-20, 20)
            axes[i, j].set_ylim(-20, 20)
    plt.savefig(out_dir / "example.png")

    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    d_forward = ot.dist(x_T, x_prior)
    d_backward = ot.dist(x_0, x_data)

    emd_forward = ot.emd2(a, b, d_forward)
    emd_backward = ot.emd2(a, b, d_backward)
    print(emd_forward, emd_backward)
    with open(out_dir / "emd.csv", "a") as f:
        f.write(f"{emd_forward},{emd_backward}\n")


intervals = [200]
SDEs = [VPSDE]
modes = [Mode.SB]
for interval in intervals:
    for sde_cls in SDEs:
        for mode in modes:
            main(interval, sde_cls, mode)
