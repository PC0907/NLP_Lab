"""Stage 10: build the paper's figures from the result artifacts.

Three figures, matching docs/Paper_Skeleton.md:

  F2  probe AUROC by layer (5-fold CV) with the log-prob baselines as
      horizontals -- the "the probe works and beats free baselines" claim.
      Sources: probes/_summary.json, results/comparison.json

  F3  the controls figure, and the one a reader should be able to understand
      without the text: each variant's per-document AUROC delta against the
      answer-only probe, with 95% bootstrap CIs over documents. The claim and
      its controls in a single frame.
      Source: results/attribution_controls.json (or selection_test.json)

  F4  risk-coverage and errors-caught curves for selective regeneration -- the
      practitioner's figure.
      Source: results/selective_regeneration.json

Every figure also writes a .csv of exactly the plotted numbers, so the values
can go straight into a LaTeX table and so the figure stays checkable. A missing
input is reported and skipped rather than fatal, so this runs usefully before
every stage has finished.

Usage:
    python scripts/10_make_figures.py --config CFG [--outdir figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless cluster node
import matplotlib.pyplot as plt

from probe_extraction.config import load_config
from probe_extraction.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Colour-blind-safe (Okabe-Ito). Fixed per signal so colours mean the same
# thing in every figure.
C = {
    "probe_fused": "#0072B2",
    "probe_answer": "#009E73",
    "mean_logprob": "#E69F00",
    "min_logprob": "#D55E00",
    "random": "#999999",
    "oracle": "#333333",
    "claim": "#0072B2",
    "control": "#999999",
    "bad": "#D55E00",
}
LABEL = {
    "probe_fused": "Probe + reasoning (fused)",
    "probe_answer": "Probe (answer token)",
    "mean_logprob": "Baseline: mean log-prob",
    "min_logprob": "Baseline: min log-prob",
    "random": "Random",
    "oracle": "Oracle",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the paper's figures.")
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", type=str, default="figures")
    p.add_argument("--controls-name", type=str, default="attribution_controls.json",
                   help="Which controls file to plot (selection_test.json also works).")
    return p.parse_args()


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logger.info("  wrote %s.pdf / .png", name)


def _csv(outdir: Path, name: str, header: list[str], rows: list[list]) -> None:
    with (outdir / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _load(path: Path):
    if not path.exists():
        logger.warning("  missing: %s -- skipping this figure.", path)
        return None
    return json.load(path.open())


# ---------------------------------------------------------------------------
# F2: probe AUROC by layer vs baselines
# ---------------------------------------------------------------------------

def figure_layers(probes_summary, comparison, outdir: Path) -> None:
    if not probes_summary:
        return
    per_layer = probes_summary.get("per_layer", {})
    layers, means, stds = [], [], []
    for L, r in sorted(per_layer.items(), key=lambda kv: int(kv[0])):
        m = r.get("cv_auroc_mean")
        if m is None:
            continue
        layers.append(int(L))
        means.append(m)
        stds.append(r.get("cv_auroc_std") or 0.0)
    if not layers:
        logger.warning("  no CV AUROCs in probes/_summary.json -- skipping F2.")
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    lo = [m - s for m, s in zip(means, stds)]
    hi = [m + s for m, s in zip(means, stds)]
    ax.fill_between(layers, lo, hi, color=C["probe_answer"], alpha=0.18, linewidth=0)
    ax.plot(layers, means, "o-", color=C["probe_answer"], lw=2, ms=5,
            label="Probe (5-fold CV)")

    rows = [[L, m, s] for L, m, s in zip(layers, means, stds)]
    if comparison:
        for key, style in (("mean_logprob", "--"), ("min_logprob", ":")):
            b = comparison.get("baselines", {}).get(key, {})
            if b.get("auroc") is not None:
                ax.axhline(b["auroc"], ls=style, color=C[key], lw=1.6,
                           label=f"{LABEL[key]} ({b['auroc']:.3f})")
                rows.append([key, b["auroc"], ""])

    best_i = max(range(len(means)), key=lambda i: means[i])
    ax.annotate(f"peak: layer {layers[best_i]}\n{means[best_i]:.3f}",
                xy=(layers[best_i], means[best_i]),
                xytext=(8, -26), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#555"))

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-field error detection by layer", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    _save(fig, outdir, "F2_probe_by_layer")
    _csv(outdir, "F2_probe_by_layer", ["layer_or_baseline", "auroc", "cv_std"], rows)


# ---------------------------------------------------------------------------
# F3: the controls
# ---------------------------------------------------------------------------

PRETTY = {
    "fused_attr": "Field-localized\nattribution",
    "fused_both": "Attribution\n+ scalars",
    "ctrl_docmean": "Document mean\nof same vectors",
    "ctrl_tracemean": "Whole trace,\npooled per doc",
    "ctrl_shuffled": "Shuffled across\nfields (control)",
    "ctrl_random": "Random vectors\n(control)",
}
# Which paired comparison gives each variant's delta against `answer`.
VS_ANSWER = {
    "fused_attr": "fused_attr_vs_answer",
    "ctrl_docmean": "docmean_vs_answer",
    "ctrl_tracemean": "tracemean_vs_answer",
    "ctrl_shuffled": "shuffled_vs_answer",
    "ctrl_random": "random_vs_answer",
}


def figure_controls(controls, outdir: Path, layer: str | None = None) -> None:
    if not controls:
        return
    sig_all = controls.get("significance", {})
    if not sig_all:
        logger.warning("  no significance block -- skipping F3.")
        return
    layer = layer or sorted(sig_all.keys(), key=int)[0]
    sig = sig_all[layer]

    names, deltas, los, his, ps = [], [], [], [], []
    for v, key in VS_ANSWER.items():
        s = sig.get(key)
        if not s or s.get("mean_delta") is None:
            continue
        b = s.get("bootstrap") or {}
        names.append(PRETTY.get(v, v))
        deltas.append(s["mean_delta"])
        los.append(s["mean_delta"] - (b.get("ci_low", s["mean_delta"])))
        his.append((b.get("ci_high", s["mean_delta"])) - s["mean_delta"])
        ps.append(s.get("p_holm"))
    if not names:
        logger.warning("  no plottable comparisons -- skipping F3.")
        return

    # The claim is coloured; the controls are grey. That is the whole argument:
    # if a grey bar matches the coloured one, the claim is not what we think.
    colours = [C["claim"] if ("attribution" in n and "Document" not in n)
               else C["control"] for n in names]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = range(len(names))
    ax.bar(xs, deltas, yerr=[los, his], capsize=4, color=colours,
           edgecolor="black", linewidth=0.6, error_kw=dict(lw=1.1, ecolor="#333"))
    ax.axhline(0, color="black", lw=1.0)

    for x, (d, p) in enumerate(zip(deltas, ps)):
        if p is None:
            continue
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        top = d + his[x]
        ax.text(x, top + 0.0022, star, ha="center", fontsize=9,
                color="#111" if star != "n.s." else "#777")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Δ per-document AUROC vs answer-only")
    ax.set_title(f"Every variant adds the same 3,584 dimensions (layer {layer})\n"
                 "95% bootstrap CI over documents; Holm-corrected significance",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    _save(fig, outdir, "F3_controls")
    _csv(outdir, "F3_controls",
         ["variant", "delta_vs_answer", "ci_low", "ci_high", "p_holm"],
         [[n, d, d - l, d + h, p] for n, d, l, h, p in zip(names, deltas, los, his, ps)])


# ---------------------------------------------------------------------------
# F4: selective regeneration
# ---------------------------------------------------------------------------

def figure_regeneration(regen, outdir: Path, regime: str = "per_doc") -> None:
    if not regen:
        return
    curves = regen.get("curves", {}).get(regime)
    if not curves:
        logger.warning("  no %s curves -- skipping F4.", regime)
        return

    order = ["oracle", "probe_fused", "probe_answer", "min_logprob",
             "mean_logprob", "random"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    rows = []
    for k in order:
        c = curves.get(k)
        if not c:
            continue
        r = c["rows"]
        cov = [1 - x["actual_frac_flagged"] for x in r]
        risk = [x["selective_risk"] for x in r]
        bud = [x["budget"] for x in r]
        rec = [x["recall"] for x in r]
        dashed = k in ("oracle", "random")
        ax1.plot(cov, risk, "-" if not dashed else "--", color=C[k], lw=1.9,
                 marker="o", ms=3, label=f"{LABEL[k]} (AURC {c['aurc']:.3f})")
        ax2.plot(bud, rec, "-" if not dashed else "--", color=C[k], lw=1.9,
                 marker="o", ms=3, label=LABEL[k])
        rows += [[k, b, rr, x["recall"], x["precision"], x["selective_risk"]]
                 for b, rr, x in zip(bud, risk, r)]

    base = regen.get("error_rate")
    if base is not None:
        ax1.axhline(base, color="#666", ls=":", lw=1.2)
        # Right-aligned: the risk curve rises left-to-right, so the legend sits
        # upper-left and this label would collide with it there.
        ax1.text(0.98, base + 0.004, f"no regeneration ({base:.1%})",
                 fontsize=8, color="#444", ha="right",
                 transform=ax1.get_yaxis_transform())

    ax1.set_xlabel("Coverage (fraction of fields kept as-is)")
    ax1.set_ylabel("Error rate among kept fields")
    ax1.set_title("Risk–coverage", fontsize=11)
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    ax2.axvline(0.20, color="#999", ls=":", lw=1.2)
    # Axes-fraction y so the label never rides on the x-axis text.
    ax2.text(0.21, 0.55, "20% budget", fontsize=8, color="#444", rotation=90,
             transform=ax2.get_xaxis_transform(), va="center")
    ax2.set_xlabel("Regeneration budget (fraction of fields re-asked)")
    ax2.set_ylabel("Errors caught (recall)")
    ax2.set_title("What the budget buys", fontsize=11)
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    fig.suptitle(f"Probe-guided selective regeneration ({regime} budget)", fontsize=12)
    _save(fig, outdir, f"F4_selective_regeneration_{regime}")
    _csv(outdir, f"F4_selective_regeneration_{regime}",
         ["signal", "budget", "selective_risk", "recall", "precision", "risk"], rows)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir,
                  log_name="10_make_figures", log_to_file=cfg.logging.log_to_file)

    art = cfg.artifacts_path
    results, probes = art / "results", art / "probes"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("F2: probe by layer")
    figure_layers(_load(probes / "_summary.json"),
                  _load(results / "comparison.json"), outdir)

    logger.info("F3: controls")
    controls = _load(results / args.controls_name)
    if controls is None and args.controls_name != "attribution_controls.json":
        controls = _load(results / "attribution_controls.json")
    figure_controls(controls, outdir)

    logger.info("F4: selective regeneration")
    regen = _load(results / "selective_regeneration.json")
    for regime in ("per_doc", "global"):
        figure_regeneration(regen, outdir, regime)

    logger.info("Figures -> %s", outdir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
