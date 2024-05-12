# %%
import matplotlib.pyplot as plt
from sbsde.distribution import MixMultiVariateNormal

# %%

dist = MixMultiVariateNormal(batch_size=100, num=2)

x = dist.sample(1000).numpy()
plt.scatter(x[:, 0], x[:, 1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)

# %%
