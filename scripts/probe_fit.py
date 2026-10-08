#!/usr/bin/env python
"""Linear probes for teammate type on frozen MARIE/LION representations.

Input: the .pt file written by scripts/probe_collect.py. For every controlled
agent, every episode and every history length t in --t-grid, build features
from the first t frames (and the t actions taken at them), then fit a 5-way
logistic regression with episode-grouped cross-validation.

Feature sets
  wm        aggregation-token hidden state of the trained world model after
            blocks 0..t-1 (what reward/continuation heads read). Perceiver sees
            the K controlled streams, as in controlled-only training.
  wm_rand   same computation with a randomly initialised world model (control).
  raw5      the last 5 raw observation frames of the agent, flattened
            (the actor's input before the tokenizer).
  recon5    tokenizer-reconstructed, clamped 5-frame stack (what the actor sees).
  raw15     the last 15 raw frames flattened (same temporal window as wm).

Example:
    PYTHONPATH=src python scripts/probe_fit.py --data probe_data/3sv5z_seed123.pt \
        --out-dir probe_results/3sv5z_seed123
    PYTHONPATH=src python scripts/probe_fit.py --synthetic --out-dir probe_results/synthetic
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_common import (  # noqa: E402
    build_policy, default_marie_config, raw_stack, reconstructed_stack, world_model_hidden,
)

FEATURES = ("wm", "wm_rand", "raw5", "recon5", "raw15")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help=".pt from probe_collect.py")
    parser.add_argument("--checkpoint", default=None, help="folder with agent.th (default: the one recorded in --data)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--t-grid", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60])
    parser.add_argument("--features", nargs="+", default=list(FEATURES), choices=FEATURES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 strength for logistic regression")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="self-test on a generated dataset with a random policy")
    parser.add_argument("--synthetic-episodes", type=int, default=60)
    parser.add_argument("--cache-out", default=None, help="write extracted features to this .npz and exit (per-file extraction while collection runs)")
    parser.add_argument("--cache-in", nargs="*", default=None, help="skip extraction; concatenate these .npz caches (team labels re-derived from team_name)")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def make_synthetic(seed, n_episodes, n_agents=3, n_actions=11, obs_dim=48, n_types=5):
    """Random obs with a type-dependent drift, so a working probe beats chance."""
    rng = np.random.RandomState(seed)
    episodes = []
    for index in range(n_episodes):
        team = index % n_types
        k_unc = 1 + (index // n_types) % 2
        length = rng.randint(20, 70)
        # Type signature: a fixed offset on 3 dimensions, strong enough that a
        # correct pipeline must reach high accuracy on raw observations.
        drift = np.zeros(obs_dim)
        drift[team * 3:(team + 1) * 3] = 1.5
        obs = np.tanh(rng.randn(length, n_agents, obs_dim) * 0.5 + drift[None, None, :])
        controlled = np.zeros(n_agents, dtype=bool)
        controlled[rng.choice(n_agents, n_agents - k_unc, replace=False)] = True
        episodes.append({
            "obs": th.tensor(obs, dtype=th.float32),
            "actions": th.tensor(rng.randint(0, n_actions, size=(length, n_agents))),
            "avail_actions": th.ones(length, n_agents, n_actions, dtype=th.int),
            "reward": th.zeros(length - 1), "terminated": th.zeros(length - 1, dtype=th.uint8),
            "controlled": th.tensor(controlled), "team_idx": team, "team_name": "type%d" % team,
            "n_uncontrolled": k_unc, "return": 0.0, "battle_won": False,
        })
    return {
        "episodes": episodes, "team_name_to_idx": {"type%d" % i: i for i in range(n_types)},
        "config": default_marie_config(), "n_agents": n_agents, "n_actions": n_actions,
        "obs_dim": obs_dim, "episode_limit": 70, "checkpoint": None,
    }


@th.no_grad()
def extract(data, policy, random_policy, t_grid, features, max_episodes=None):
    rows = []  # one dict per (episode, agent, t)
    feats = {name: [] for name in features}
    episodes = data["episodes"][:max_episodes] if max_episodes else data["episodes"]
    started = time.time()
    for ep_index, ep in enumerate(episodes):
        controlled = ep["controlled"].nonzero().flatten()
        k = int(controlled.numel())
        obs_c = ep["obs"][:, controlled].transpose(0, 1).float()          # [K, L, obs]
        act_c = ep["actions"][:, controlled].transpose(0, 1).long()[..., None]  # [K, L, 1]
        length = obs_c.shape[1]
        for t in t_grid:
            if t > length - 1:  # need t complete (obs, action) blocks before the final frame
                continue
            frames = obs_c[:, :t]
            actions = act_c[:, :t]
            for name in features:
                if name == "wm":
                    value = world_model_hidden(policy, frames, actions, k)
                elif name == "wm_rand":
                    value = world_model_hidden(random_policy, frames, actions, k)
                elif name == "raw5":
                    value = raw_stack(frames, 5)
                elif name == "raw15":
                    value = raw_stack(frames, 15)
                elif name == "recon5":
                    value = reconstructed_stack(policy, frames)
                else:
                    raise ValueError(name)
                feats[name].append(value.cpu().numpy())
            for agent_slot in range(k):
                rows.append({"episode": ep_index, "agent": int(controlled[agent_slot]), "t": t,
                             "team": ep["team_idx"], "team_name": ep["team_name"],
                             "k_uncontrolled": ep["n_uncontrolled"]})
        if (ep_index + 1) % 20 == 0:
            print("[fit] features for %d/%d episodes (%.1fs)" % (ep_index + 1, len(episodes), time.time() - started), flush=True)
    rows = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    feats = {name: np.concatenate(values, 0) for name, values in feats.items()}
    return rows, feats


def load_caches(paths, team_name_to_idx=None):
    """Concatenate per-file feature caches; episodes are offset so groups stay unique
    and team labels are re-derived from team_name with one canonical mapping."""
    rows, feats, offset = {}, {}, 0
    for path in paths:
        z = np.load(path, allow_pickle=False)
        r = {k[4:]: z[k] for k in z.files if k.startswith("row_")}
        f = {k[5:]: z[k] for k in z.files if k.startswith("feat_")}
        r["episode"] = r["episode"] + offset
        offset = int(r["episode"].max()) + 1
        for k, v in r.items():
            rows.setdefault(k, []).append(v)
        for k, v in f.items():
            feats.setdefault(k, []).append(v)
    rows = {k: np.concatenate(v) for k, v in rows.items()}
    feats = {k: np.concatenate(v) for k, v in feats.items()}
    names = sorted(set(rows["team_name"].tolist()))
    if not team_name_to_idx or not set(names) <= set(team_name_to_idx):
        team_name_to_idx = {n: i for i, n in enumerate(names)}
    rows["team"] = np.array([team_name_to_idx[n] for n in rows["team_name"]])
    return rows, feats, team_name_to_idx


def probe_accuracy(x, y, groups, folds, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(groups))
    folds = min(folds, n_groups)
    if folds < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), None
    splitter = GroupKFold(n_splits=folds)
    accuracies = []
    confusion = np.zeros((int(y.max()) + 1, int(y.max()) + 1), dtype=int)
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        accuracies.append(float((pred == y[test]).mean()))
        for truth, guess in zip(y[test], pred):
            confusion[truth, guess] += 1
    return float(np.mean(accuracies)), float(np.std(accuracies)), confusion


def main():
    cli = parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)
    th.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    if cli.synthetic:
        data = make_synthetic(cli.seed, cli.synthetic_episodes)
        checkpoint_file = None
    else:
        data = th.load(cli.data, map_location="cpu")
        checkpoint_dir = cli.checkpoint or data["checkpoint"]
        checkpoint_file = os.path.join(checkpoint_dir, "agent.th")
        if not os.path.exists(checkpoint_file):
            sys.exit("agent.th not found in %s" % checkpoint_dir)

    th.set_num_threads(cli.torch_threads)
    config = data["config"]
    if cli.cache_in:
        rows, feats, team_name_to_idx = load_caches(cli.cache_in, data.get("team_name_to_idx"))
        print("[fit] loaded %d caches, %d rows, teams %s" % (len(cli.cache_in), len(rows["t"]), team_name_to_idx))
    else:
        policy = build_policy(config, data["obs_dim"], data["n_agents"], data["n_actions"], checkpoint_file)
        th.manual_seed(cli.seed + 1)
        random_policy = build_policy(config, data["obs_dim"], data["n_agents"], data["n_actions"], None)
        print("[fit] %d episodes, teams %s" % (len(data["episodes"]), data["team_name_to_idx"]))
        rows, feats = extract(data, policy, random_policy, cli.t_grid, cli.features, cli.max_episodes)
        team_name_to_idx = data["team_name_to_idx"]
        if cli.cache_out:
            os.makedirs(os.path.dirname(os.path.abspath(cli.cache_out)), exist_ok=True)
            np.savez(cli.cache_out, t_grid=np.array(cli.t_grid),
                     **{"row_" + k: v for k, v in rows.items()},
                     **{"feat_" + k: v for k, v in feats.items()})
            print("[fit] wrote feature cache", cli.cache_out, "rows", len(rows["t"]))
            return
    data["team_name_to_idx"] = team_name_to_idx
    n_types = len(team_name_to_idx)
    chance = 1.0 / n_types

    results = {"chance": chance, "t_grid": cli.t_grid, "features": {}, "by_k": {},
               "n_agents": int(data["n_agents"]), "n_episodes": len(data["episodes"]),
               "team_name_to_idx": team_name_to_idx, "C": cli.C, "folds": cli.folds}
    for name in cli.features:
        results["features"][name] = {}
        for t in cli.t_grid:
            mask = rows["t"] == t
            if mask.sum() < 2 * n_types:
                continue
            acc, std, confusion = probe_accuracy(feats[name][mask], rows["team"][mask], rows["episode"][mask], cli.folds, cli.C, cli.seed)
            results["features"][name][str(t)] = {"acc": acc, "std": std, "n": int(mask.sum())}
            if name == "wm" and confusion is not None:
                results.setdefault("confusion_wm", {})[str(t)] = confusion.tolist()
            print("[fit] %-8s t=%3d  acc=%.3f ± %.3f  (n=%d)" % (name, t, acc, std, mask.sum()))
        for k in sorted(np.unique(rows["k_uncontrolled"])):
            key = "K=%d" % k
            results["by_k"].setdefault(key, {})[name] = {}
            for t in cli.t_grid:
                mask = (rows["t"] == t) & (rows["k_uncontrolled"] == k)
                if mask.sum() < 2 * n_types:
                    continue
                acc, std, _ = probe_accuracy(feats[name][mask], rows["team"][mask], rows["episode"][mask], cli.folds, cli.C, cli.seed)
                results["by_k"][key][name][str(t)] = {"acc": acc, "std": std, "n": int(mask.sum())}

    with open(os.path.join(cli.out_dir, "probe_results.json"), "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    plot(results, data["team_name_to_idx"], cli.out_dir)
    print("[fit] wrote", os.path.join(cli.out_dir, "probe_results.json"))


def plot(results, team_name_to_idx, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"wm": "World-model latent (trained)", "wm_rand": "World-model latent (random init)",
              "raw5": "Raw obs, 5 frames", "recon5": "Reconstructed obs, 5 frames (actor input)",
              "raw15": "Raw obs, 15 frames"}
    def draw(ax, feature_dict, title=None):
        for name, series in feature_dict.items():
            ts = sorted(int(t) for t in series)
            if not ts:
                continue
            acc = np.array([series[str(t)]["acc"] for t in ts])
            std = np.array([series[str(t)]["std"] for t in ts])
            ax.plot(ts, acc, marker="o", ms=3, label=labels.get(name, name))
            ax.fill_between(ts, acc - std, acc + std, alpha=0.15)
        ax.axhline(results["chance"], color="gray", ls="--", lw=1, label="Chance")
        ax.set_xlabel("History length (environment steps)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        if title:
            ax.set_title(title, fontsize=9)

    # Pooled over K (all episodes).
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    draw(ax, results["features"])
    ax.set_ylabel("Teammate-type accuracy")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "teammate_probe_pooled.pdf"))
    fig.savefig(os.path.join(out_dir, "teammate_probe_pooled.png"), dpi=150)

    # One panel per number of uncontrolled teammates (main figure).
    by_k = results.get("by_k", {})
    if by_k:
        keys = sorted(by_k, key=lambda k: int(k.split("=")[1]))
        n_agents_total = results.get("n_agents")
        fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys) + 0.4, 3.0), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, keys):
            k = int(key.split("=")[1])
            if n_agents_total:
                title = "%d controlled, %d uncontrolled" % (n_agents_total - k, k)
            else:
                title = "%d uncontrolled teammate%s" % (k, "" if k == 1 else "s")
            draw(ax, by_k[key], title)
        axes[0].set_ylabel("Teammate-type accuracy")
        axes[-1].legend(fontsize=6.5, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "teammate_probe.pdf"))
        fig.savefig(os.path.join(out_dir, "teammate_probe.png"), dpi=150)

    confusions = results.get("confusion_wm", {})
    if confusions:
        # Show the model's own context window (15) when available, else the last t.
        t_last = "15" if "15" in confusions else max(confusions, key=int)
        matrix = np.array(confusions[t_last], dtype=float)
        matrix = matrix / matrix.sum(1, keepdims=True).clip(min=1)
        names = [n.replace("agent_", "") for n, _ in sorted(team_name_to_idx.items(), key=lambda kv: kv[1])]
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True teammate type")
        ax.set_title("World-model probe, t=%s" % t_last, fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, "%.2f" % matrix[i, j], ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > 0.5 else "black")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "teammate_probe_confusion.pdf"))


if __name__ == "__main__":
    main()
