# import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# parameters to plot
alphas = [0.2, 0.5, 1]

# x values across open interval (0, 1)
x = np.linspace(0.001, 0.999, 1000)

# create figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# plot PDF and CDF for each alpha value
for alpha_val in alphas:
    pdf = beta.pdf(x, alpha_val, alpha_val)
    cdf = beta.cdf(x, alpha_val, alpha_val)
    label = f"$\\alpha$ = {alpha_val}"
    axes[0].plot(x, pdf, label=label)
    axes[1].plot(x, cdf, label=label)

# configure PDF plot
axes[0].set_title("Beta Distribution PDF", fontsize=14)
axes[0].set_xlabel("$\\lambda$", fontsize=12)
axes[0].set_ylabel("Probability Density", fontsize=12)
axes[0].tick_params(labelsize=12)
#axes[0].legend(fontsize=12)
axes[0].grid()
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 5)

# configure CDF plot
axes[1].set_title("Beta Distribution CDF", fontsize=14)
axes[1].set_xlabel("$\\lambda$", fontsize=12)
axes[1].set_ylabel("Cumulative Probability", fontsize=12)
axes[1].tick_params(labelsize=12)
axes[1].legend(fontsize=12)
axes[1].grid()
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

# add title and save plot
#fig.suptitle("Beta($\\alpha$, $\\alpha$) Distribution", fontsize=14)
fig.tight_layout()
plt.savefig("plot_beta.png", dpi = 300, bbox_inches = "tight")
plt.show()
