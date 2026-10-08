#!/usr/bin/env python
"""Merge several probe_collect.py outputs (same checkpoint) into one file.

    python scripts/probe_merge.py --out probe_data/3sv5z.pt probe_data/3sv5z_agent_*.pt
"""
import argparse
import torch as th


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    # Files collected with --teammates <one type> label that type as index 0, so
    # the per-file indices are not comparable. Relabel from the stored team_name
    # with one canonical mapping: the full config's uncntrl_agents order (the same
    # order the controller uses when all types are present), else sorted names.
    loaded = [(path, th.load(path, map_location="cpu", weights_only=False)) for path in a.inputs]
    names = set(e["team_name"] for _, d in loaded for e in d["episodes"])
    config_names = list(loaded[0][1].get("config", {}).get("uncntrl_agents", {}).keys())
    if names <= set(config_names):
        canonical = {n: i for i, n in enumerate(config_names)}
    else:
        canonical = {n: i for i, n in enumerate(sorted(names))}
    merged = dict(loaded[0][1])
    merged["episodes"] = []
    merged["sources"] = []
    merged["team_name_to_idx"] = canonical
    for path, d in loaded:
        for key in ("checkpoint", "n_agents", "n_actions", "obs_dim", "episode_limit"):
            assert merged[key] == d[key], (key, path)
        for e in d["episodes"]:
            e = dict(e)
            e["team_idx"] = canonical[e["team_name"]]
            merged["episodes"].append(e)
        merged["sources"].append((path, len(d["episodes"]), d.get("seed")))
    merged.pop("partial", None)
    print("team_name_to_idx:", canonical)
    counts = {}
    for e in merged["episodes"]:
        counts[(e["team_name"], e["n_uncontrolled"])] = counts.get((e["team_name"], e["n_uncontrolled"]), 0) + 1
    th.save(merged, a.out)
    print("saved", len(merged["episodes"]), "episodes to", a.out)
    for k in sorted(counts):
        print("  %-12s K=%d  n=%d" % (k[0], k[1], counts[k]))


if __name__ == "__main__":
    main()
