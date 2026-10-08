# Teammate-type probe: state and TODO

Last updated: 2026-10-07 ~23:05 (probe DONE; OOD-teammate run DONE: no degradation, 18.31 vs 18.30; see notes/probe_results.md). Paper deadline: ~2026-10-09 (32 h from this note).
Owner: Mateus (local run on Mac). Code owner: Felipe.

## Goal

Test whether the frozen MARIE/LION world model encodes the *type* of the uncontrolled
teammates (IPPO, QMIX, VDN, MAPPO, IQL) in its latent state, beyond what the controlled
agents' raw observations already reveal. One figure for the results section; the paper
narrative does NOT depend on it.

Design (linear probe, no training of the world model):
- Collect labelled episodes with the trained checkpoint against the five teammate types.
  Labels come for free: `uncontrolled_team_idx` (episode-level, 0..4) and `trainable_agents`
  mask are written by `src/runners/episode_runner.py` in open-team mode.
- Features per controlled agent at history length t:
  1. world-model aggregation-token hidden state (`world_model.dynamics_sequence`, with
     `world_model.context_n_agents = K` and focal ids relabelled 0..K-1 so the Perceiver
     path matches controlled-only training; see `src/components/marie_context.py`)
  2. raw 5-frame observation stack (what the actor would see without the tokenizer)
  3. tokenizer-reconstructed 5-frame stack (what the actor actually sees)
  4. hidden state of a randomly initialised world model (chance control)
  5. optional: POAM encoder embedding, if a POAM checkpoint is available
- 5-way logistic regression, train/test split by episode, accuracy vs t. Chance = 20%.
  Key comparison: (1) vs (2). Report separately for K = 1 and K = 2 controlled agents.
- Phrase result as "teammate type recoverable from the model's view of its own agents",
  because the world model only ever saw controlled agents' streams.

## State of the machine (Mac, Apple M4 Max; terminal + Miniconda are Intel builds under Rosetta)

- [x] Conda env `adhoc` (Python 3.10, torch 2.2.2 x86) now has everything: sacred, smac, pysc2,
      gym 0.23.1, numpy 1.26.4, the MPE fork, matrix-games, sklearn, matplotlib. All repo imports OK.
      Run things as: `PYTHONPATH=src python ...`
- [x] Uncontrolled teammates downloaded AND unpacked into repo `uncntrl_agents/`
      (gitignored by git status; contains 3sv5z, 5v6, 8v9, 10v11, mpe-pp). Source:
      https://huggingface.co/datasets/carolinewang01/naht, file
      `naht_unctrl_agents_10-18-24.zip` (94 MB). 3sv5z has seeds 78590 AND 112358 (both
      used in this repo), five types each, layout matches
      `src/config/open/open_train_3sv5z.yaml` exactly, e.g.
      `uncntrl_agents/3sv5z/ippo/models/ippo_baseline_seed=78590_02-29-22-51-47/best/agent.th`.
      So `base_uncntrl_path` = `<repo>/uncntrl_agents`.
      Re-download if needed: `curl -L -o x.zip https://huggingface.co/datasets/carolinewang01/naht/resolve/main/naht_unctrl_agents_10-18-24.zip`
      (huggingface-cli on this machine has a path bug; use curl.)
- [x] Retail StarCraft II 5.0.17 installed via Battle.net at `/Applications/StarCraft II`,
      SMAC maps copied to its Maps folder. BUT: DEAD END. The game binary exits with code 1
      within a second whenever it is started outside Battle.net (tested from Claude's shell,
      from the user's own Ghostty terminal, via `open`, with/without args, clean env). No
      crash report, empty GameLogs. pysc2 therefore cannot drive it. Do not retry.
- [x] Plan B WORKS (Oct 7 18:08): Linux headless SC2 4.6.2 (same build as Felipe)
      inside an x86_64 Docker container with Rosetta emulation. Docker Desktop verified:
      amd64 containers run, Rosetta enabled, 14 CPUs, 8 GB VM memory.
      - Image `tnteam-cpu:amd64` built from `docker/Dockerfile.cpu` (python:3.10-slim amd64,
        torch 2.2.2 CPU, pinned deps from docker/requirements.txt, setuptools<70 because
        sacred needs pkg_resources, sklearn). Rebuild if needed:
        `docker build --platform linux/amd64 -f docker/Dockerfile.cpu -t tnteam-cpu:amd64 .`
      - Game unzipped at `3rdparty/StarCraftII` (Versions/Base69232), SMAC maps at
        `3rdparty/StarCraftII/Maps/SMAC_Maps` (23 maps). Zip kept at
        `3rdparty/SC2.4.6.2.69232.zip`. All gitignored.
      - SMOKE TEST PASSED: 2 episodes of 3s_vs_5z with a random-init MARIE policy against
        the real IPPO/QMIX teammates, correct labels and controlled masks, obs in [-1,1],
        0.7 min total incl. game launch. Data: `3rdparty/_smoke/sc2/data.pt`.
        Fake run (config + random agent.th): `3rdparty/_smoke/sc2/{sacred,models}/fake`,
        regenerate with `3rdparty/_smoke/make_fake_sc2.py` (and `make_fake_mpe.py`).
      - Collection command template (from repo root; note docker needs `-i` only for stdin):
        docker run --rm --platform linux/amd64 -v "$PWD":/workspace/tnteam -w /workspace/tnteam \
          tnteam-cpu:amd64 python scripts/probe_collect.py \
          --checkpoint naht_results/<...>/models/<expt>/best \
          --uncntrl-path /workspace/tnteam/uncntrl_agents \
          --episodes-per-type 20 --n-uncontrolled 1 2 --out probe_data/3sv5z.pt
        (SC2PATH and PYTHONPATH are set in the image; user_info.yaml has Mac paths, hence
        the explicit --uncntrl-path. Checkpoint path must be relative to the repo or
        /workspace/tnteam/...).
      - Fit can run either in the container or in the Mac `adhoc` env (CPU only).
      - Same game version as Felipe's training, so the version caveat disappears.
      - Retail Mac StarCraft II (26 GB in /Applications) is now unnecessary; can be
        uninstalled via Battle.net to reclaim disk.
- [x] `src/config/user_info.yaml` written (gitignored): results -> `<repo>/naht_results`,
      teammates -> `<repo>/uncntrl_agents`.
- [x] MARIE checkpoint from Felipe RECEIVED (Oct 7 ~18:40) as a self-contained bundle at
      `naht_results/marie_3sv5z_naht_seed78590_finished400k/` (177 MB, gitignored). Contents:
      - `checkpoints/marie/400344/agent.th` = full MARIEPolicy state dict (world_model 248 tensors,
        actors, critics, ema_critics; 15.2M params) + optimizer states + marie_training_state.th.
        Loads into repo `MARIEPolicy` with 0 missing / 0 unexpected keys (verified in `adhoc`).
      - `training_config.json` = the sacred config (170 keys). NOT in the `models->sacred` layout
        the collector auto-detects, so ALWAYS pass `--config <bundle>/training_config.json`.
        Key facts: seed 123, t_max 400k (final step 400344, run completed), teammate seed 78590,
        `marie_controlled_replay_only: true`, `n_uncontrolled: null` (1 or 2 sampled per episode),
        marie_stack_obs 5, matwm_max_seq_length 15, hidden_dim 64, parallel runner batch 4,
        `expt_logname marie_baseline_seed=123_10-03-20-38-21`, torch 2.5.1+cu121.
      - Recorded eval at the checkpoint (128 eps): test_return_mean 18.91 (std 4.38),
        test_battle_won_mean 0.086. Use these numbers in the paper only as "checkpoint used
        for the probe"; the learning curves come from Felipe.
      - `src/` snapshot is byte-identical to repo `src` on matwm (only user_info.yaml, MPE
        pretrained params, notebooks differ). `uncntrl_agents/3sv5z` copy (seed 78590 only) is
        identical in layout to repo `uncntrl_agents/`. `run_checkpoint.py --verify` passes.
      - Real-checkpoint smoke (Docker, 2 eps): ippo K=1 return 17.71, iql K=1 return 20.01, both
        hit the 250-step limit (kiting), ~30 s/episode incl. launch. Log: `3rdparty/_logs/sc2_real_smoke.log`.
- [ ] Docker fallback (daemon not running): repo Dockerfile is CUDA/amd64, would need edits
      to run on CPU under emulation. Only if the retail client fails.

## TODO (in order)

1. [ ] Mateus: install Battle.net + StarCraft II; add SMAC maps. (Dominates wall clock.)
2. [x] Packages installed into `adhoc`.
3. [x] Teammates unpacked to `uncntrl_agents/`; `user_info.yaml` written.
4. [x] SMAC smoke test passed inside the Docker container (see Plan B above).
5. [x] Scripts written and tested (untracked, in `scripts/`):
       - `scripts/probe_common.py`  helpers (policy build, world-model hidden, stacks)
       - `scripts/probe_collect.py` collection; auto-finds sacred config.json next to the
         checkpoint (models -> sacred/<run>/1/config.json), forces CPU/episode runner,
         restricts the controller to one teammate type at a time and a fixed K, saves .pt
       - `scripts/probe_fit.py`     features wm / wm_rand / raw5 / recon5 / raw15, 5-way
         logistic regression (sklearn, GroupKFold by episode), accuracy-vs-t plot,
         confusion matrix; `--synthetic` self-test.
       Verified: (a) synthetic self-test, raw5 -> 100%, labels flow correctly;
       (b) FULL end-to-end on MPE predator-prey with the real teammate checkpoints from the
       bundle and a random-init MARIE policy (30 episodes, 1.2 s/episode, probes at chance
       as expected for a random policy). Fake MPE run lives in session scratchpad
       `mpe_smoke/` (config + random agent.th); recreate if needed, it is a 20-line snippet.
       Known caveat: `sacred_config_for_checkpoint` assumes `.../models/<expt>/<step>`; pass
       `--config` explicitly if Felipe's layout differs.
6. [x] Felipe sent the bundle (see above).
7. [x] Collection DONE (restarted Oct 7 ~19:30, finished ~20:25): `bash scripts/probe_collect_parallel.sh 25`
       = 5 containers (one per teammate type, seeds 3001..3005), each 25 eps x K in {1,2},
       250 episodes total, outputs `probe_data/3sv5z_agent_<type>.pt`, logs
       `3rdparty/_logs/collect_agent_<type>.log`. Expected ~25-30 min. Then merge:
       `python scripts/probe_merge.py --out probe_data/3sv5z.pt probe_data/3sv5z_agent_*.pt`.
       Old plan for reference: (= 5 types x 2 K x 20 = 200 episodes).
       RESTARTED Oct 7 ~19:30 with OMP_NUM_THREADS=1 / --torch-threads 1: the first launch ran
       at 160-200 s/episode because 5 torch processes x 14 threads oversubscribed the 14
       emulated cores. With 1 thread each: 65-78 s/episode per container. Collector now
       writes `<out>.partial` every 10 episodes (atomic), so a stopped run can still be used.
       LABEL GOTCHA (fixed): with `--teammates <one type>` the controller numbers that type 0,
       so per-type files all had team_idx 0. `probe_merge.py` relabels from `team_name` using
       the full-config order (ippo 0, iql 1, mappo 2, qmix 3, vdn 4); `probe_collect.py` now
       writes canonical indices for future runs. The index never reaches the MARIE policy.
       Dry-run of merge + fit on 2 partial files (20 eps, ippo vs qmix, chance 50%) worked:
       wm 97% @t=15 vs raw15 90% vs wm_rand 60%. Noisy, but pipeline is validated end to end.
       On the Mac, `probe_fit.py` needs `--checkpoint <bundle>/checkpoints/marie/400344`
       because the data file stores the container path (/workspace/tnteam/...). Smoke test: 2 episodes (42 and 194 steps)
       in 0.7 min incl. launch, so expect ~10-20 s/episode -> ~45-70 min for 200 under
       emulation. To parallelise, launch several containers with `--teammates agent_ippo
       agent_qmix` etc. and different `--seed`, then merge the episode lists (probe_fit
       reads one .pt; a 5-line merge is needed). Run inside the Docker container, NOT the
       Mac env (the Mac game cannot be launched by pysc2).
8. [x] Fit done: `probe_data/3sv5z.pt` (merged, 250 eps), `probe_data/fit_full/` has
       probe_results.json, features.npz (cache), teammate_probe.pdf (per-K panels),
       teammate_probe_pooled.pdf, teammate_probe_confusion.pdf (t=15), fit.log.
       C sweep in `probe_data/fit_C{0.01,0.1,10}/`. Numbers, reading, draft paragraph,
       method sentence and caption: `notes/probe_results.md`. Figures copied to
       `latex/pictures/` (main checkout and paper-draft worktree).
       Re-fit without re-extracting: `probe_fit.py --data probe_data/3sv5z.pt --checkpoint
       <bundle>/checkpoints/marie/400344 --out-dir <dir> --cache-in probe_data/fit_full/features.npz`.
9. [ ] Mateus: paste the paragraph into the results section, cite Fig. teammate_probe, add
       RQ2 to intro. Optional if time: second checkpoint (another seed) to show stability.

## Related decisions already made

- MPE run for the paper: NOT feasible in 32 h (reward head collapses on near-sparse
  collision reward; VQ tokenizer cannot resolve collision threshold; actor clamps obs to
  [-1,1]). Paper is SMAC-only; MPE is future work. Note: NAHT's MPE task used reward
  shaping 0.01 (folder `mpe-pp__ts=100_shape=0.01`), so say "nearly sparse".
- Paper narrative is recorded in Claude memory (`aamas27-marie-naht-paper.md`) and in
  this repo nowhere else yet; copy to Overleaf notes if needed.

## OOD teammates (POAM Q2 analogue), started Oct 7 ~22:00

- Decision (Mateus): paper mirrors POAM's three questions; add an OOD-teammate evaluation.
- Run: `bash scripts/probe_collect_parallel.sh 25 112358 ood` -> 5 containers, teammate
  checkpoints rewritten to seed 112358 (never seen in training), 25 eps x K in {1,2} per type,
  outputs `probe_data/3sv5z_ood_agent_<type>.pt`, logs `3rdparty/_logs/collect_ood_agent_<type>.log`.
  Expected ~55 min. Collector flag: `--teammate-seed` (records teammate_seed in the file).
- Then: `python scripts/probe_merge.py --out probe_data/3sv5z_ood.pt probe_data/3sv5z_ood_agent_*.pt`
  and `python scripts/ood_compare.py --indist probe_data/3sv5z.pt --ood probe_data/3sv5z_ood.pt
  --out-dir probe_data/ood` -> table + `ood_teammates.pdf` (bars per type, in-dist vs OOD,
  95% CI, panels K=1 / K=2 / all). Self-tested with in-dist vs itself.
- [x] Run finished (~55 min). Merged `probe_data/3sv5z_ood.pt`; results `probe_data/ood/ood_results.json`,
  figure `latex/pictures/ood_teammates.pdf` (both checkouts); OOD paragraph + table in notes/probe_results.md.
- Bonus: the OOD episodes can also be fed to probe_fit.py (same features) to check whether
  the probe transfers across teammate seeds (train on 78590 episodes, test on 112358). Not
  implemented; would need a train/test split option in probe_fit.

