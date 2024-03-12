from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.optim as optim
from sbsde.distribution import MixMultiVariateNormal, PriorNormal
from sbsde.loss import compute_loss
from sbsde.model import ToyModel
from sbsde.sde import VESDE, Direction
from sbsde.utils import flatten_dim01
from tqdm.auto import trange

batch_size = 1000
sigma_min = 1e-2
sigma_max = 5
num_iter = 1000
sample_size = 1000
retrain = True


def main(interval):
    print("interval =", interval)
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p_data = MixMultiVariateNormal(batch_size)
    p_prior = PriorNormal(sigma_max, device)
    sde = VESDE(p_data, p_prior, device, sigma_min, sigma_max, interval=interval)
    net_f = ToyModel().to(device)
    net_b = ToyModel().to(device)

    optimizer_f = optim.AdamW(net_f.parameters(), lr=1e-4)
    optimizer_b = optim.AdamW(net_b.parameters(), lr=1e-4)

    if (out_dir / "ckpt.pth").exists() and not retrain:
        print("Load pretrained checkpoint")
        net_f.load_state_dict(torch.load(out_dir / "ckpt.pth")["net_f"])
        net_b.load_state_dict(torch.load(out_dir / "ckpt.pth")["net_b"])

    if retrain:
        net_f.train()
        net_b.train()
        losses = []
        bar = trange(num_iter)
        for it in bar:
            optimizer_f.zero_grad()
            optimizer_b.zero_grad()
            xs_f, zs_f, x_term_f = sde.sample_traj(net_f)
            _ts = sde.ts.repeat(batch_size)
            xs_f = flatten_dim01(xs_f)
            zs_f = flatten_dim01(zs_f)
            loss = compute_loss(sde, net_b, _ts, xs_f, x_term_f, zs_f)
            loss.backward()
            optimizer_f.step()
            optimizer_b.step()
            bar.set_postfix(loss=loss.item())
            losses.append(loss.item())

        plt.plot(range(1, num_iter + 1), losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid()
        plt.title("Training Loss")

        torch.save(
            {
                "net_f": net_f.state_dict(),
                "net_b": net_b.state_dict(),
                "optimizer_f": optimizer_f.state_dict(),
                "optimizer_b": optimizer_b.state_dict(),
            },
            out_dir / f"ckpt_{interval}.pth",
        )

    n = sample_size
    x_data = sde.p_data.sample_n(n).to(device)
    x = x_data
    for t in sde.ts:
        with torch.no_grad():
            z = net_f(t, x)
            x = sde.propagate(t, x, z, Direction.FORWARD)
    x_T = x.detach().cpu().numpy()
    x_prior = sde.p_prior.sample_n(n)
    x = x_prior
    for t in reversed(sde.ts):
        with torch.no_grad():
            z = net_b(t, x)
            x = sde.propagate(t, x, z, Direction.BACKWARD)
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
    plt.savefig(out_dir / f"plot_{interval}.png")

    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    d_forward = ot.dist(x_T, x_prior)
    d_backward = ot.dist(x_0, x_data)

    emd_forward = ot.emd2(a, b, d_forward)
    emd_backward = ot.emd2(a, b, d_backward)
    print(emd_forward, emd_backward)
    with open(out_dir / "result.csv", "a") as f:
        f.write(f"{emd_forward},{emd_backward}\n")


for interval in [10, 20, 50, 100]:
    main(interval)
