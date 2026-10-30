#!/usr/bin/env python3
"""Phase 6 scaling figure and summary CSVs for biomni-code.

Reads bench/phase6_results.csv (produced by scaling_suite.py; columns: size,
variant, rep, runtime_sec, maxrss_kb, output_digest, returncode,
protected_original) and writes, into --out-dir (typically /mnt/results/):

- phase6_panels.png / phase6_panels.svg — one figure, three square panels in a
  row: (a) runtime vs input size, (b) peak RSS vs input size, (c) per-size
  digest-match fraction. Original/baseline in gray, optimized in red, mean +/-
  stddev error bars, no gridlines, bold a/b/c panel labels.
- runtime.csv, memory.csv, correctness.csv — per-size x variant summaries
  (reps, mean, std; digest matches/total/fraction).

Correctness reference per size: the majority output_digest among baseline
(protected-original) rows; falls back to the majority digest across all rows at
that size. If only one variant is present (zero accepted changes), the panels
are baseline-only and labeled as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 15
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13

COLOR_ORIGINAL = "#7F7F7F"  # gray
COLOR_OPTIMIZED = "#D62728"  # red
EXTRA_COLORS = ["#0279EE", "#75A025", "#FF9400"]

DISPLAY_NAMES = {"baseline": "Original", "optimized": "Optimized"}


def display_name(variant: str, baseline_only: bool) -> str:
    name = DISPLAY_NAMES.get(variant, variant)
    if baseline_only:
        name += " (baseline only)"
    return name


def variant_color(variant: str, idx: int) -> str:
    if variant == "baseline":
        return COLOR_ORIGINAL
    if variant == "optimized":
        return COLOR_OPTIMIZED
    return EXTRA_COLORS[idx % len(EXTRA_COLORS)]


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("returncode") != "0":
                continue
            rows.append(row)
    return rows


def reference_digests(rows: list[dict]) -> dict[str, str]:
    """Per-size reference digest: majority among baseline rows, else overall."""
    by_size_all: dict[str, Counter] = defaultdict(Counter)
    by_size_base: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        digest = row.get("output_digest") or ""
        if not digest:
            continue
        size = row["size"]
        by_size_all[size][digest] += 1
        if row["variant"] == "baseline" or row.get("protected_original") not in ("", "0", "false", "False"):
            by_size_base[size][digest] += 1
    refs = {}
    for size in by_size_all:
        pool = by_size_base.get(size) or by_size_all[size]
        refs[size] = pool.most_common(1)[0][0]
    return refs


def aggregate(rows: list[dict], refs: dict[str, str]):
    """Return sizes, variants, and per-(size, variant) summaries."""
    rt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    rss: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    dig: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        size, variant = row["size"], row["variant"]
        if row.get("runtime_sec"):
            rt[size][variant].append(float(row["runtime_sec"]))
        if row.get("maxrss_kb"):
            rss[size][variant].append(float(row["maxrss_kb"]))
        digest = row.get("output_digest") or ""
        if digest and size in refs:
            dig[size][variant].append(1 if digest == refs[size] else 0)

    def numeric(size: str):
        try:
            return float(size)
        except ValueError:
            return None

    sizes = list({row["size"] for row in rows})
    if all(numeric(s) is not None for s in sizes):
        sizes.sort(key=lambda s: float(s))
    else:
        # Non-numeric labels: order by pooled mean runtime (a scaling suite's
        # premise is that cost grows with size), falling back to label order.
        pooled = {s: [r for row in rows if row["size"] == s
                      for r in [float(row["runtime_sec"])] if row.get("runtime_sec")]
                  for s in sizes}
        sizes.sort(key=lambda s: (statistics.fmean(pooled[s]) if pooled[s] else float("inf"), s))
    variants = sorted({row["variant"] for row in rows},
                      key=lambda v: (v != "baseline", v != "optimized", v))
    return sizes, variants, rt, rss, dig


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def write_csvs(out_dir: Path, sizes, variants, rt, rss, dig) -> dict:
    paths = {}
    with (out_dir / "runtime.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["size", "variant", "reps", "runtime_mean_sec", "runtime_std_sec"])
        for s in sizes:
            for v in variants:
                vals = rt[s][v]
                if vals:
                    mean, std = mean_std(vals)
                    w.writerow([s, v, len(vals), f"{mean:.6g}", f"{std:.6g}"])
    with (out_dir / "memory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["size", "variant", "reps", "maxrss_mean_kb", "maxrss_std_kb"])
        for s in sizes:
            for v in variants:
                vals = rss[s][v]
                if vals:
                    mean, std = mean_std(vals)
                    w.writerow([s, v, len(vals), f"{mean:.6g}", f"{std:.6g}"])
    with (out_dir / "correctness.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["size", "variant", "reps", "digest_matches", "digest_total", "correctness_frac"])
        for s in sizes:
            for v in variants:
                vals = dig[s][v]
                if vals:
                    w.writerow([s, v, len(vals), sum(vals), len(vals), f"{sum(vals) / len(vals):.6g}"])
    paths.update(runtime=str(out_dir / "runtime.csv"), memory=str(out_dir / "memory.csv"),
                 correctness=str(out_dir / "correctness.csv"))
    return paths


def plot_panels(out_dir: Path, sizes, variants, rt, rss, dig) -> dict:
    baseline_only = len(variants) <= 1
    numeric = all(_is_numeric(s) for s in sizes)
    if numeric:
        xvals = [float(s) for s in sizes]
        log_x = max(xvals) / min(xvals) >= 50 if min(xvals) > 0 else False
    else:
        xvals = list(range(len(sizes)))
        log_x = False

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    # Distinct markers with the optimized marker slightly smaller, so exact
    # overlaps (e.g. 100% correctness for both variants) remain visible.
    markers = {"baseline": ("o", 10), "optimized": ("s", 7)}
    for vi, v in enumerate(variants):
        color = variant_color(v, vi)
        label = display_name(v, baseline_only)
        marker, msize = markers.get(v, ("^", 8))
        rt_m, rt_e, mem_m, mem_e, cor = [], [], [], [], []
        for s in sizes:
            m, e = mean_std(rt[s][v])
            rt_m.append(m)
            rt_e.append(e)
            m, e = mean_std(rss[s][v])
            mem_m.append(m / 1024.0)  # kB -> MB
            mem_e.append(e / 1024.0)
            vals = dig[s][v]
            cor.append(100.0 * sum(vals) / len(vals) if vals else float("nan"))
        style = dict(fmt="-" + marker, color=color, ecolor=color, elinewidth=2, capsize=5,
                     capthick=2, linewidth=3, markersize=msize, alpha=0.9, label=label)
        axes[0].errorbar(xvals, rt_m, yerr=rt_e, **style)
        axes[1].errorbar(xvals, mem_m, yerr=mem_e, **style)
        axes[2].errorbar(xvals, cor, **style)

    axes[0].set_ylabel("Runtime (s)")
    axes[1].set_ylabel("Peak RSS (MB)")
    axes[2].set_ylabel("Correctness (%)")
    axes[2].set_ylim(-5, 105)
    for letter, ax in zip("abc", axes):
        ax.set_xlabel("Input size")
        ax.set_box_aspect(1)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.28, 1.03, letter, transform=ax.transAxes, fontsize=22,
                fontweight="bold", va="bottom", ha="left")
        if numeric:
            if log_x:
                ax.set_xscale("log")
                ax.set_xticks(xvals)
                ax.set_xticklabels(sizes)
            ax.minorticks_off()
        else:
            ax.set_xticks(xvals)
            ax.set_xticklabels(sizes)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()

    png = out_dir / "phase6_panels.png"
    svg = out_dir / "phase6_panels.svg"
    fig.savefig(png, dpi=150)
    fig.savefig(svg)
    plt.close(fig)
    return {"figure_png": str(png), "figure_svg": str(svg)}


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True, help="bench/phase6_results.csv from scaling_suite.py")
    p.add_argument("--out-dir", required=True, help="Output directory (typically /mnt/results/)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.csv))
    if not rows:
        print("plot_scaling_panels: no successful rows in input CSV", file=sys.stderr)
        raise SystemExit(1)
    refs = reference_digests(rows)
    sizes, variants, rt, rss, dig = aggregate(rows, refs)
    written = write_csvs(out_dir, sizes, variants, rt, rss, dig)
    written.update(plot_panels(out_dir, sizes, variants, rt, rss, dig))
    print(json.dumps({"wrote": written, "sizes": sizes, "variants": variants}))


if __name__ == "__main__":
    main()
