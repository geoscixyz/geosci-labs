import numpy as np
import matplotlib.pyplot as plt

b = 2.5
x = np.linspace(0, 20, 200)

f = np.exp(-((x - b) ** 2) / 2)
fig, ax = plt.subplots(1, 1, figsize=(9, 3))
ax.plot(x, f, "k")
ax.set_ylim([-0.5, 1.5])
ax.set_xticks(np.linspace(0, 20, 21))
ax.fill_between(x, f, np.zeros_like(f), color="k")
ax.set_title("Seismic Wavelet", fontsize=13)
ax.set_xlabel("Time (ms)")

# fig.savefig('SeisWavelet.png',dpi=300)
