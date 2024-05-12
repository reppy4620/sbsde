from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.optim as optim
from sbsde.distribution import MixMultiVariateNormal
from sbsde.loss import compute_loss, compute_loss_s_u
from sbsde.model import ToyModel
from sbsde.sde import VESDE, VPSDE, Direction, SimpleSDE
from sbsde.utils import flatten_dim01
from tqdm.auto import trange

batch_size = 200
prior_sigma = 1.0
num_iter = 10000
sample_size = 1000
sample_interval = 1000


class Mode:
    SB = "SB"
    SU = "SU"


def main(interval, sde_cls, mode):
    print(interval, sde_cls, mode)
    out_dir = Path(f"out/{interval}-{sde_cls.__name__}-{mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_data = MixMultiVariateNormal(batch_size, device=device)
    p_prior = MixMultiVariateNormal(batch_size, radius=8, num=2, device=device)
    sde = sde_cls(p_data, p_prior, device, interval=interval)
    net1 = ToyModel().to(device)
    net2 = ToyModel().to(device)

    optimizer1 = optim.AdamW(net1.parameters(), lr=1e-4)
    optimizer2 = optim.AdamW(net2.parameters(), lr=1e-4)

    def sample_plot(iteration):
        n = sample_size
        x_data = sde.p_data.sample_n(n).to(device)
        x = x_data
        for t in sde.ts:
            with torch.no_grad():
                if mode == Mode.SB:
                    z = net1(t, x)
                    x = sde.propagate(t, x, z, direction=Direction.FORWARD)
                else:
                    u = net1(t, x)
                    x = sde.propagate_s_u(t, x, u, direction=Direction.FORWARD)
        x_T = x.detach().cpu().numpy()
        x_prior = sde.p_prior.sample_n(n)
        x = x_prior
        for t in reversed(sde.ts):
            with torch.no_grad():
                if mode == Mode.SB:
                    z_hat = net2(t, x)
                    x = sde.propagate(t, x, z_hat, direction=Direction.BACKWARD)
                else:
                    u = net1(t, x)
                    s = net2(t, x)
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
        plt.savefig(out_dir / f"example_{iteration}.png")

        a, b = np.ones((n,)) / n, np.ones((n,)) / n
        d_forward = ot.dist(x_T, x_prior)
        d_backward = ot.dist(x_0, x_data)

        emd_forward = ot.emd2(a, b, d_forward)
        emd_backward = ot.emd2(a, b, d_backward)
        with open(out_dir / f"emd_{iteration}.csv", "a") as f:
            f.write(f"{iteration},{emd_forward},{emd_backward}\n")

    net1.train()
    net2.train()
    losses = []
    bar = trange(num_iter)
    for it in bar:
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        xs, us, x_term_f = sde.sample_traj(net1)
        _ts = sde.ts.repeat(batch_size)
        xs = flatten_dim01(xs)
        us = flatten_dim01(us)

        if mode == Mode.SB:
            loss = compute_loss(
                sde=sde,
                net_b=net2,
                ts=_ts,
                xs=xs,
                x_term=x_term_f,
                zs_f=us,
            )
        else:
            loss = compute_loss_s_u(
                sde=sde, net_s=net2, ts=_ts, xs=xs, x_term=x_term_f, us=us
            )
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        bar.set_postfix(loss=loss.item())
        losses.append(loss.item())
        if (it + 1) % sample_interval == 0:
            sample_plot(it + 1)

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
            "net1": net1.state_dict(),
            "net2": net2.state_dict(),
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
        },
        out_dir / "ckpt.pth",
    )


# intervals = [50]
# SDEs = [VESDE]
# modes = [Mode.SB, Mode.SU]
intervals = [100]
SDEs = [VPSDE]
modes = [Mode.SB]
for interval in intervals:
    for sde_cls in SDEs:
        for mode in modes:
            main(interval, sde_cls, mode)
