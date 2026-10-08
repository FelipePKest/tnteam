#!/usr/bin/env python
"""Compare LION returns against in-distribution vs out-of-distribution teammates.

    python scripts/ood_compare.py --indist probe_data/3sv5z.pt --ood probe_data/3sv5z_ood.pt \
        --out-dir probe_data/ood

Mirrors Wang et al. (2024) Fig. 4: mean test return per teammate type with 95% CI,
in-distribution (teammate seed used in training) vs OOD (unseen teammate seed).
"""
import argparse
import json
import os

import numpy as np
import torch as th


def load(path):
    d = th.load(path, map_location="cpu", weights_only=False)
    rows = [(e["team_name"].replace("agent_", ""), int(e["n_uncontrolled"]), float(e["return"]),
             bool(e["battle_won"]), int(e["obs"].shape[0] - 1)) for e in d["episodes"]]
    return d, rows


def summarise(rows, order):
    out = {}
    for name in order:
        for k in sorted({r[1] for r in rows}) + ["all"]:
            sel = [r for r in rows if r[0] == name and (k == "all" or r[1] == k)]
            if not sel:
                continue
            ret = np.array([r[2] for r in sel])
            won = np.array([r[3] for r in sel], dtype=float)
            out[(name, k)] = {"n": len(sel), "return_mean": ret.mean(), "return_ci95": 1.96 * ret.std(ddof=1) / np.sqrt(len(sel)),
                              "win_rate": won.mean(), "ep_len": float(np.mean([r[4] for r in sel]))}
    for k in sorted({r[1] for r in rows}) + ["all"]:
        sel = [r for r in rows if k == "all" or r[1] == k]
        ret = np.array([r[2] for r in sel]); won = np.array([r[3] for r in sel], dtype=float)
        out[("mean", k)] = {"n": len(sel), "return_mean": ret.mean(), "return_ci95": 1.96 * ret.std(ddof=1) / np.sqrt(len(sel)),
                            "win_rate": won.mean(), "ep_len": float(np.mean([r[4] for r in sel]))}
    return out


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indist", required=True)
    p.add_argument("--ood", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d_in, rows_in = load(a.indist)
    d_ood, rows_ood = load(a.ood)
    order = [n.replace("agent_", "") for n, _ in sorted(d_in["team_name_to_idx"].items(), key=lambda kv: kv[1])]
    s_in, s_ood = summarise(rows_in, order), summarise(rows_ood, order)
    ks = sorted({r[1] for r in rows_in})

    print("teammate seeds: in-dist %s, OOD %s" % (d_in.get("teammate_seed") or d_in.get("teammate_seed_trained_against") or "train", d_ood.get("teammate_seed")))
    table = []
    for k in ks + ["all"]:
        print("\n== K=%s uncontrolled ==" % k)
        print("%-7s %4s  %-16s %-16s %7s %7s  %6s %6s" % ("type", "n", "in-dist return", "OOD return", "drop", "z", "win_in", "win_ood"))
        for name in order + ["mean"]:
            i, o = s_in[(name, k)], s_ood[(name, k)]
            ret_i = [r[2] for r in rows_in if (name == "mean" or r[0] == name) and (k == "all" or r[1] == k)]
            ret_o = [r[2] for r in rows_ood if (name == "mean" or r[0] == name) and (k == "all" or r[1] == k)]
            z = welch(ret_i, ret_o)
            drop = i["return_mean"] - o["return_mean"]
            print("%-7s %4d  %6.2f ± %-7.2f  %6.2f ± %-7.2f %+7.2f %+7.2f  %6.2f %6.2f" % (
                name, i["n"], i["return_mean"], i["return_ci95"], o["return_mean"], o["return_ci95"], -drop, -z, i["win_rate"], o["win_rate"]))
            table.append({"type": name, "k": k, "n_in": i["n"], "n_ood": o["n"], "return_in": i["return_mean"], "ci_in": i["return_ci95"],
                          "return_ood": o["return_mean"], "ci_ood": o["return_ci95"], "diff_ood_minus_in": -drop, "welch_z": -z,
                          "win_in": i["win_rate"], "win_ood": o["win_rate"], "len_in": i["ep_len"], "len_ood": o["ep_len"]})
    with open(os.path.join(a.out_dir, "ood_results.json"), "w") as f:
        json.dump({"table": table, "indist": a.indist, "ood": a.ood,
                   "teammate_seed_in": d_in.get("teammate_seed_trained_against"), "teammate_seed_ood": d_ood.get("teammate_seed")}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(ks) + 1, figsize=(3.0 * (len(ks) + 1) + 0.5, 2.9), sharey=True)
    n_agents = int(d_in["n_agents"])
    for ax, k in zip(axes, ks + ["all"]):
        x = np.arange(len(order) + 1)
        labels = order + ["mean"]
        mi = [s_in[(n, k)]["return_mean"] for n in labels]; ci = [s_in[(n, k)]["return_ci95"] for n in labels]
        mo = [s_ood[(n, k)]["return_mean"] for n in labels]; co = [s_ood[(n, k)]["return_ci95"] for n in labels]
        ax.bar(x - 0.2, mi, 0.4, yerr=ci, capsize=2, label="In-distribution teammates", color="#4C72B0")
        ax.bar(x + 0.2, mo, 0.4, yerr=co, capsize=2, label="OOD teammates (unseen seed)", color="#DD8452")
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.axvline(len(order) - 0.5, color="gray", lw=0.6, ls=":")
        title = "all episodes" if k == "all" else "%d controlled, %d uncontrolled" % (n_agents - k, k)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean test return")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=8, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(a.out_dir, "ood_teammates.pdf"))
    fig.savefig(os.path.join(a.out_dir, "ood_teammates.png"), dpi=150)
    print("\nwrote", os.path.join(a.out_dir, "ood_teammates.pdf"))


if __name__ == "__main__":
    main()
