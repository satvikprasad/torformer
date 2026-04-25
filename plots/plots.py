import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.4,
})


def norm_bound_sq(gamma_max, D):
    g2 = gamma_max ** 2
    g2m1 = g2 - 1.0
    g2D = gamma_max ** (2.0 * D)
    val = g2 * (g2D - 1.0) / (g2m1 ** 2) - D / g2m1
    return np.maximum(val, 0.0)


def norm_bound(gamma_max, D):
    return np.sqrt(norm_bound_sq(gamma_max, D))


def ungained_bound(D):
    return np.sqrt(D * (D + 1.0) / 2.0)


# ── Figure 1: Norm bound vs D for various gamma_max ────────────────────
fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))

D_arr = np.arange(2, 257, 1)
D_ungained = ungained_bound(D_arr)

gamma_values = [0.7, 0.85, 0.95, 0.99, 1.01, 1.05, 1.1, 1.3]
cmap_below = plt.cm.Blues_r
cmap_above = plt.cm.Reds

gammas_below = [g for g in gamma_values if g < 1.0]
gammas_above = [g for g in gamma_values if g > 1.0]

ax1.plot(D_arr, D_ungained, color="black", linewidth=2.0, linestyle="-",
         label=r"Ungained $\sqrt{D(D+1)/2}$", zorder=10)

for i, g in enumerate(gammas_below):
    t = i / max(len(gammas_below) - 1, 1)
    color = cmap_below(0.3 + 0.5 * t)
    ax1.plot(D_arr, norm_bound(g, D_arr), color=color,
             linestyle="--", label=rf"$\gamma_{{\max}}={g}$")

for i, g in enumerate(gammas_above):
    t = i / max(len(gammas_above) - 1, 1)
    color = cmap_above(0.35 + 0.5 * t)
    ax1.plot(D_arr, norm_bound(g, D_arr), color=color,
             linestyle="-", label=rf"$\gamma_{{\max}}={g}$")

ax1.set_yscale("log")
ax1.set_xlabel(r"Dimension $D$")
ax1.set_ylabel(r"$\|\tau_D\| \;/\; \|\vec{\rho}\|$")
ax1.set_title(r"Norm bound vs.\ dimension for varying $\gamma_{\max}$")
ax1.legend(loc="upper left", framealpha=0.9, edgecolor="0.8", ncol=2)
ax1.set_xlim(2, 256)
ax1.grid(True, alpha=0.15, linewidth=0.4)
ax1.axhline(y=1, color="gray", linewidth=0.3, linestyle=":")

fig1.tight_layout()
fig1.savefig("plots/norm_vs_D.png")
print("Figure 1 saved.")


# ── Figure 2: Norm bound vs gamma_max for various D ────────────────────
fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))

gamma_arr = np.concatenate([
    np.linspace(0.3, 0.99, 200),
    np.linspace(1.01, 2.0, 200),
])

D_values = [8, 16, 32, 64, 128, 256]
cmap_d = plt.cm.viridis

for i, D in enumerate(D_values):
    t = i / (len(D_values) - 1)
    color = cmap_d(0.15 + 0.7 * t)
    bound = norm_bound(gamma_arr, D)
    ug = ungained_bound(D)
    ax2.plot(gamma_arr, bound, color=color, label=rf"$D={D}$")
    ax2.axhline(y=ug, color=color, linewidth=0.5, linestyle=":", alpha=0.5)

ax2.axvline(x=1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5,
            label=r"$\gamma_{\max}=1$")

ax2.set_yscale("log")
ax2.set_xlabel(r"Effective gain $\gamma_{\max} = g \sin\theta$")
ax2.set_ylabel(r"$\|\tau_D\| \;/\; \|\vec{\rho}\|$")
ax2.set_title(r"Norm bound vs.\ $\gamma_{\max}$ for varying $D$")
ax2.legend(loc="upper left", framealpha=0.9, edgecolor="0.8", ncol=2)
ax2.set_xlim(0.3, 2.0)
ax2.grid(True, alpha=0.15, linewidth=0.4)

fig2.tight_layout()
fig2.savefig("plots/norm_vs_gamma.png")
print("Figure 2 saved.")


# ── Figure 3: Heatmap of log norm bound in (gamma_max, D) space ────────
fig3, ax3 = plt.subplots(figsize=(5.5, 4.2))

gamma_heat = np.concatenate([
    np.linspace(0.3, 0.99, 150),
    np.linspace(1.01, 2.0, 150),
])
D_heat = np.arange(4, 257, 1)
G, DD = np.meshgrid(gamma_heat, D_heat)
Z = np.log10(np.maximum(norm_bound(G, DD), 1e-10))

cmap_heat = LinearSegmentedColormap.from_list(
    "custom", ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd",
               "#f4a261", "#e76f51", "#d62828", "#6a040f"]
)

im = ax3.pcolormesh(gamma_heat, D_heat, Z, cmap=cmap_heat, shading="auto",
                    vmin=-0.5, vmax=6)
cbar = fig3.colorbar(im, ax=ax3, label=r"$\log_{10}(\|\tau_D\|/\|\vec{\rho}\|)$",
                     pad=0.02)
cbar.ax.tick_params(labelsize=8)

ax3.axvline(x=1.0, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
ax3.set_xlabel(r"Effective gain $\gamma_{\max}$")
ax3.set_ylabel(r"Dimension $D$")
ax3.set_title(r"$\log_{10}$ norm bound in $(\gamma_{\max}, D)$ space")

fig3.tight_layout()
fig3.savefig("plots/norm_heatmap.png")
print("Figure 3 saved.")


# ── Figure 4: Scaling exponent ──────────────────────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(6.5, 3.2))

gamma_ratio = [0.7, 0.85, 0.95, 1.05, 1.1, 1.3]
D_ratio = np.arange(4, 129, 1)

for i, g in enumerate(gamma_ratio):
    ratio = norm_bound(g, D_ratio) / ungained_bound(D_ratio)
    color = cmap_below(0.3 + 0.5 * i / len(gamma_ratio)) if g < 1 else cmap_above(0.35 + 0.5 * (i - len([x for x in gamma_ratio if x < 1])) / len([x for x in gamma_ratio if x >= 1]))
    style = "--" if g < 1 else "-"
    ax4a.plot(D_ratio, ratio, color=color, linestyle=style,
              label=rf"$\gamma_{{\max}}={g}$")

ax4a.axhline(y=1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax4a.set_xlabel(r"$D$")
ax4a.set_ylabel(r"Gained / Ungained")
ax4a.set_title(r"Ratio to ungained bound")
ax4a.legend(fontsize=7, loc="best", framealpha=0.9, edgecolor="0.8")
ax4a.set_yscale("log")
ax4a.grid(True, alpha=0.15, linewidth=0.4)

D_exp = np.arange(8, 257, 1)
for i, g in enumerate(gamma_ratio):
    b = norm_bound(g, D_exp)
    log_b = np.log(b + 1e-30)
    log_D = np.log(D_exp)
    exponent = np.gradient(log_b, log_D)
    color = cmap_below(0.3 + 0.5 * i / len(gamma_ratio)) if g < 1 else cmap_above(0.35 + 0.5 * (i - len([x for x in gamma_ratio if x < 1])) / len([x for x in gamma_ratio if x >= 1]))
    style = "--" if g < 1 else "-"
    ax4b.plot(D_exp, exponent, color=color, linestyle=style,
              label=rf"$\gamma_{{\max}}={g}$")

ax4b.axhline(y=0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5,
             label=r"$\mathcal{O}(\sqrt{D})$")
ax4b.axhline(y=1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5,
             label=r"$\mathcal{O}(D)$")
ax4b.set_xlabel(r"$D$")
ax4b.set_ylabel(r"$d\log\|\tau\| / d\log D$")
ax4b.set_title(r"Scaling exponent")
ax4b.legend(fontsize=6.5, loc="best", framealpha=0.9, edgecolor="0.8", ncol=2)
ax4b.set_ylim(-0.1, 3.5)
ax4b.grid(True, alpha=0.15, linewidth=0.4)

fig4.tight_layout()
fig4.savefig("plots/norm_scaling.png")
print("Figure 4 saved.")


# ── Figure 5: log ||tau|| / ||rho|| vs D (linear x, linear y = log) ───
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(7.0, 3.5))

D_arr5 = np.arange(2, 513, 1)

# Left panel: log norm bound vs D for various gamma_max
# This reveals: gamma > 1 gives linear growth in log (i.e. exponential norm),
# gamma = 1 gives ~ log D growth, gamma < 1 gives bounded log.

gamma_values_5 = [0.5, 0.7, 0.9, 0.95, 0.99, 1.01, 1.02, 1.05, 1.1, 1.2]
gammas_below_5 = [g for g in gamma_values_5 if g < 1.0]
gammas_above_5 = [g for g in gamma_values_5 if g > 1.0]

# Ungained
ax5a.plot(D_arr5, np.log(ungained_bound(D_arr5)), color="black", linewidth=2.0,
          linestyle="-", label=r"Ungained", zorder=10)

for i, g in enumerate(gammas_below_5):
    t = i / max(len(gammas_below_5) - 1, 1)
    color = cmap_below(0.25 + 0.55 * t)
    b = norm_bound(g, D_arr5)
    ax5a.plot(D_arr5, np.log(np.maximum(b, 1e-30)), color=color,
              linestyle="--", label=rf"$\gamma_{{\max}}={g}$")

for i, g in enumerate(gammas_above_5):
    t = i / max(len(gammas_above_5) - 1, 1)
    color = cmap_above(0.3 + 0.55 * t)
    b = norm_bound(g, D_arr5)
    ax5a.plot(D_arr5, np.log(np.maximum(b, 1e-30)), color=color,
              linestyle="-", label=rf"$\gamma_{{\max}}={g}$")

ax5a.set_xlabel(r"Dimension $D$")
ax5a.set_ylabel(r"$\log(\|\tau_D\| / \|\vec{\rho}\|)$")
ax5a.set_title(r"Log norm bound vs.\ $D$")
ax5a.legend(loc="upper left", framealpha=0.9, edgecolor="0.8", fontsize=6.5, ncol=2)
ax5a.set_xlim(2, 512)
ax5a.grid(True, alpha=0.15, linewidth=0.4)

# Right panel: slope d(log bound)/dD — this is the "exponential rate"
# For gamma > 1, this converges to log(gamma_max) as D grows.
# For gamma < 1, this converges to 0.

ax5b.axhline(y=0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)

D_slope = np.arange(4, 513, 1)

for i, g in enumerate(gammas_below_5):
    t = i / max(len(gammas_below_5) - 1, 1)
    color = cmap_below(0.25 + 0.55 * t)
    b = norm_bound(g, D_slope)
    log_b = np.log(np.maximum(b, 1e-30))
    slope = np.gradient(log_b, D_slope)
    ax5b.plot(D_slope, slope, color=color, linestyle="--",
              label=rf"$\gamma_{{\max}}={g}$")

for i, g in enumerate(gammas_above_5):
    t = i / max(len(gammas_above_5) - 1, 1)
    color = cmap_above(0.3 + 0.55 * t)
    b = norm_bound(g, D_slope)
    log_b = np.log(np.maximum(b, 1e-30))
    slope = np.gradient(log_b, D_slope)
    ax5b.plot(D_slope, slope, color=color, linestyle="-",
              label=rf"$\gamma_{{\max}}={g}$")
    # Plot asymptotic rate log(gamma_max)
    ax5b.axhline(y=np.log(g), color=color, linewidth=0.5, linestyle=":",
                 alpha=0.4)

# Ungained slope
b_ug = ungained_bound(D_slope)
slope_ug = np.gradient(np.log(b_ug), D_slope)
ax5b.plot(D_slope, slope_ug, color="black", linewidth=1.5, linestyle="-",
          label="Ungained", zorder=10)

ax5b.set_xlabel(r"Dimension $D$")
ax5b.set_ylabel(r"$d\log\|\tau\| / dD$")
ax5b.set_title(r"Growth rate in $D$")
ax5b.legend(loc="upper right", framealpha=0.9, edgecolor="0.8", fontsize=6, ncol=2)
ax5b.set_xlim(4, 512)
ax5b.set_ylim(-0.01, 0.25)
ax5b.grid(True, alpha=0.15, linewidth=0.4)

fig5.tight_layout()
fig5.savefig("plots/log_norm_bound.png")
fig5.savefig("plots/log_norm_bound.pdf")
print("Figure 5 saved.")
