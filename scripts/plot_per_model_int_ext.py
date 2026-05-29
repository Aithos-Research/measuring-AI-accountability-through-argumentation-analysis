"""Plot per-model defence quality split by elicitation track.

Builds Figure 1 of the paper: per-model (Local+Global)/2 normalised quality,
one blue bar for r_int (reasoning track) and one orange bar for r_ext
(justification track), with the reasoning-branch panel and the CoT panel
separated by a dashed vertical line.

The Global mean here is the cell-wise mean over the three substantive
global dimensions (acceptability, relevance, sufficiency), matching the
2026-05 setup change that pulls coherence out into its own diagnostic.

GPT 5.4's r_int is rendered as "NA" because its reasoning channel is
encrypted ciphertext and not interpretable as a defended-position score.

Usage:
    python -m scripts.plot_per_model_int_ext \
        logs/deliberative_justification_v2/20260527_combined_panels_curation/reasoning
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCALE_MIN, SCALE_MAX = 1, 3
SCALE_RANGE = SCALE_MAX - SCALE_MIN

REASONING_PANEL = [
    ("gpt-5.4", "GPT 5.4"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("kimi-k2p6", "Kimi K2.6"),
    ("qwen3p6-plus", "Qwen 3.6 Plus"),
    ("glm-5p1", "GLM 5.1"),
    ("deepseek-v4-pro", "DeepSeek V4 Pro"),
    ("gemini-2.5-pro", "Gemini 2.5 Pro"),
]
COT_PANEL = [
    ("mistral-large-2512", "Mistral Large 2512"),
    ("Llama 3.3", "Llama 3.3"),
]


def cell_local_mean(local):
    nums = [v for q, dims in (local or {}).items()
            if isinstance(dims, dict)
            for v in dims.values() if isinstance(v, int)]
    return (sum(nums) / len(nums)) if nums else None


def cell_global_mean_no_coh(glob):
    nums = [v for k, v in (glob or {}).items()
            if k in ("acceptability", "relevance", "sufficiency")
            and isinstance(v, int)]
    return (sum(nums) / len(nums)) if nums else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output PNG path (default: <run_dir>/per_model_int_ext.png)")
    args = p.parse_args()

    samples_path = args.run_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))

    # Per (model, track) cell-level averages
    cells = defaultdict(list)  # (model, track) -> [(local+global)/2 normalised]
    for dil in samples["dilemmas"]:
        for tg in dil["targets"]:
            model = tg["model"].split("/")[-1]
            for tk in tg.get("tracks", []) or []:
                track = tk.get("text_kind")
                if track not in ("reasoning", "justification"):
                    continue
                for p4 in tk.get("phase4", []) or []:
                    if p4.get("skipped"):
                        continue
                    lm = cell_local_mean(p4.get("local"))
                    gm = cell_global_mean_no_coh(p4.get("global"))
                    if lm is None or gm is None:
                        continue
                    avg_raw = (lm + gm) / 2
                    avg_norm = (avg_raw - SCALE_MIN) / SCALE_RANGE
                    cells[(model, track)].append(avg_norm)

    def stat(model, track):
        xs = cells.get((model, track), [])
        if not xs:
            return None, None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return m, 0.0
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        se = (var / len(xs)) ** 0.5
        return m, se

    # Build data arrays
    full_panel = REASONING_PANEL + COT_PANEL
    labels = [lbl for _, lbl in full_panel]
    int_means, int_se = [], []
    ext_means, ext_se = [], []
    for short, _ in full_panel:
        # GPT 5.4 reasoning channel is encrypted ciphertext — render as NA.
        if short == "gpt-5.4":
            int_means.append(np.nan); int_se.append(0.0)
        else:
            m, se = stat(short, "reasoning")
            int_means.append(m if m is not None else np.nan)
            int_se.append(se or 0.0)
        m, se = stat(short, "justification")
        ext_means.append(m if m is not None else np.nan)
        ext_se.append(se or 0.0)

    # Plot
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 4.5))
    int_arr = np.array(int_means, dtype=float)
    ext_arr = np.array(ext_means, dtype=float)
    # Replace NaN with 0 for plotting but track masks for annotation
    int_plot = np.where(np.isnan(int_arr), 0.0, int_arr)
    ext_plot = np.where(np.isnan(ext_arr), 0.0, ext_arr)

    ax.bar(x - width / 2, int_plot, width,
           yerr=int_se, label=r"$r_{\mathrm{int}}$",
           color="#4C7FB6", capsize=2)
    ax.bar(x + width / 2, ext_plot, width,
           yerr=ext_se, label=r"$r_{\mathrm{ext}}$",
           color="#E08E45", capsize=2)

    # Annotate NA bars
    for i, v in enumerate(int_arr):
        if np.isnan(v):
            ax.text(x[i] - width / 2, 0.01, "NA",
                    ha="center", va="bottom",
                    fontsize=8, color="grey", fontstyle="italic")

    # Panel separator between reasoning-branch and CoT
    n_reason = len(REASONING_PANEL)
    ax.axvline(n_reason - 0.5, color="black", linestyle="--",
               linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Normalised defence quality\n(mean of Local + Global, 0--1)")
    ax.set_ylim(0, 1.0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out = args.output or (args.run_dir / "per_model_int_ext.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
