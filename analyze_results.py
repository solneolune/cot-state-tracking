"""
Analysis and plotting for flip-flop experiment results.
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load(path: str):
    with open(path) as f:
        return json.load(f)

def get_values(results, model, density, condition, key="glitch_rate"):
    rows = [r for r in results
            if r["model"] == model
            and r["density"] == density
            and r["condition"] == condition]
    rows.sort(key=lambda r: r["n_events"])
    return [r["n_events"] for r in rows], [r[key] for r in rows]


CONDITION_LABELS = {
    "direct": "No CoT (Direct)",
    "generic_cot": "Generic CoT",
    "state_tracking_cot": "State-Tracking CoT",
}
CONDITION_COLORS = {
    "direct": "#e63946",
    "generic_cot": "#f4a261",
    "state_tracking_cot": "#2a9d8f",
}
CONDITION_MARKERS = {
    "direct": "o",
    "generic_cot": "s",
    "state_tracking_cot": "^",
}


# Figure 1: Main Result (glitch rate vs sequence length)

def plot_main(results, output_path="fig_main.pdf"):
    models = ["gpt-3.5-turbo", "gpt-4o-mini"]
    densities = ["sparse", "dense"]
    conditions = ["direct", "generic_cot", "state_tracking_cot"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    fig.suptitle("Glitch Rate vs. Sequence Length by Prompting Strategy",
                 fontsize=13, y=1.01)

    for row, density in enumerate(densities):
        for col, model in enumerate(models):
            ax = axes[row][col]
            for cond in conditions:
                xs, ys = get_values(results, model, density, cond, "glitch_rate")
                ax.plot(xs, ys,
                        label=CONDITION_LABELS[cond],
                        color=CONDITION_COLORS[cond],
                        marker=CONDITION_MARKERS[cond],
                        linewidth=2, markersize=6)

            ax.set_title(f"{model}\n({density})", fontsize=10)
            ax.set_xlabel("Sequence length (# events)")
            ax.set_ylabel("Glitch rate (1 − accuracy)")
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax.grid(alpha=0.3)
            if row == 0 and col == 1:
                ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


# Figure 2: Accuracy Heatmap

def plot_heatmap(results, model="gpt-3.5-turbo", density="sparse",
                 output_path="fig_heatmap.png"):
    conditions = ["direct", "generic_cot", "state_tracking_cot"]
    n_events_levels = sorted(set(r["n_events"] for r in results))

    data = np.zeros((len(conditions), len(n_events_levels)))
    for i, cond in enumerate(conditions):
        for j, n in enumerate(n_events_levels):
            rows = [r for r in results
                    if r["model"] == model
                    and r["density"] == density
                    and r["condition"] == cond
                    and r["n_events"] == n]
            if rows:
                data[i, j] = rows[0]["accuracy"]

    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(n_events_levels)))
    ax.set_xticklabels(n_events_levels)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([CONDITION_LABELS[c] for c in conditions])
    ax.set_xlabel("Sequence length (# events)")
    ax.set_title(f"Accuracy — {model} ({density})")
    plt.colorbar(im, ax=ax, label="Accuracy")

    for i in range(len(conditions)):
        for j in range(len(n_events_levels)):
            ax.text(j, i, f"{data[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if 0.3 < data[i,j] < 0.8 else "white")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()

# Entry point

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    results = load(path)

    plot_main(results, "fig_main.png")
    plot_heatmap(results, model="gpt-3.5-turbo", density="sparse", output_path="fig_heatmap_35_sparse.png")
    plot_heatmap(results, model="gpt-3.5-turbo", density="dense", output_path="fig_heatmap_35_dense.png")
    plot_heatmap(results, model="gpt-4o-mini", density="sparse", output_path="fig_heatmap_4o_sparse.png")
    plot_heatmap(results, model="gpt-4o-mini", density="dense", output_path="fig_heatmap_4o_dense.png")
