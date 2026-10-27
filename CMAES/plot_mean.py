import numpy as np
import matplotlib.pyplot as plt

def plot_cma_mean_variables(
    data_source="outcmaes/xmean.dat",
    mode="twinx",          # 'twinx' for dual y-axes, or 'scaled' for single axis with pc multiplier
    pc_scale_factor=25.0,  # multiplier used only if mode='scaled'
    save_path="mean_plot.png"
):
    """
    Plots the mean trajectories of pc and pl from pycma output without the 
    vertical maxiter boundary line or trailing right-margin artifacts.

    Parameters:
    -----------
    data_source : str, cma.CMADataLogger, or np.ndarray
        Path to 'outcmaes/xmean.dat', a loaded CMADataLogger instance, or an array.
    mode : str
        'twinx' (recommended: independent y-axes) or 'scaled' (pc scaled by pc_scale_factor on one axis).
    pc_scale_factor : float
        Scaling multiplier applied to pc when mode='scaled'.
    save_path : str or None
        Path to save the resulting figure (e.g., 'mean_variables.png').
    """
    # 1. Parse pycma data structure
    # In pycma's xmean.dat: col 0 = iteration, col 1 = evaluations, cols 5: = variables
    if isinstance(data_source, str):
        raw = np.loadtxt(data_source, comments="%")
    elif hasattr(data_source, "xmean"):
        raw = data_source.xmean
    elif hasattr(data_source, "data") and "xmean" in data_source.data:
        raw = data_source.data["xmean"]
    else:
        raw = np.asarray(data_source)

    if raw.shape[1] > 5:
        iterations = raw[:, 1]
        pc = raw[:, 5]  # Index 0
        pl = raw[:, 6]  # Index 1
    else:
        # Fallback if the user pre-sliced the array
        iterations = np.arange(len(raw))
        pc = raw[:, 0]
        pl = raw[:, 1]

    # Style parameters
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    color_pc = "#1f77b4"  # Default pycma blue
    color_pl = "#ff7f0e"  # Default pycma orange

    if mode == "twinx":
        # Left axis: pc
        line_pc = ax1.plot(iterations, pc, color=color_pc, linewidth=1.8, label=r"$p_c$ (mean)")
        ax1.set_xlabel("Iterations", fontsize=11)
        ax1.set_ylabel(r"$p_c$", color=color_pc, fontsize=12, fontweight="bold")
        ax1.tick_params(axis="y", labelcolor=color_pc)
        ax1.set_ylim(bottom=0, top=max(pc) * 1.15)

        # Right axis: pl
        ax2 = ax1.twinx()
        line_pl = ax2.plot(iterations, pl, color=color_pl, linewidth=1.8, label=r"$p_\ell$ (mean)")
        ax2.set_ylabel(r"$p_\ell$", color=color_pl, fontsize=12, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor=color_pl)
        ax2.set_ylim(bottom=0, top=max(pl) * 1.1)

        # Unified legend
        lines = line_pc + line_pl
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right", framealpha=0.9, edgecolor="lightgray")
        
        # Grid based on the primary axis
        ax1.grid(True, linestyle="--", alpha=0.4)

    elif mode == "scaled":
        # Single axis: pc scaled by a constant factor
        scaled_pc = pc * pc_scale_factor
        line_pl = ax1.plot(iterations, pl, color=color_pl, linewidth=1.8, label=r"$p_\ell$")
        line_pc = ax1.plot(
            iterations, 
            scaled_pc, 
            color=color_pc, 
            linewidth=1.8, 
            label=rf"$p_c \times {pc_scale_factor:g}$"
        )
        
        ax1.set_xlabel("Iterations", fontsize=11)
        ax1.set_ylabel("Parameter Value", fontsize=11)
        ax1.set_ylim(bottom=0)
        ax1.legend(loc="upper right", framealpha=0.9, edgecolor="lightgray")
        ax1.grid(True, linestyle="--", alpha=0.4)

    # Clean borders and prevent trailing whitespace / line extensions
    ax1.set_xlim(iterations[0], iterations[-1])
    ax1.set_title(r"Object Variables Mean Trajectory ($p_c, p_\ell$)", fontsize=12, pad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Point directly to pycma's output directory, or pass your logger instance
    plot_cma_mean_variables(data_source="outcmaes/xmean.dat", mode="twinx")

