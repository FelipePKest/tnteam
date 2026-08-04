"""Regenerate the specialized and type-conditional MPE-PP N-k evaluations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from nk_evaluation import target_set_eval, type_conditional_classifier_runs_eval


TASK_ROOT = "./naht_results/mpe-pp"
TRAINING_SEEDS = ["38410", "78590", "93718", "112358", "1285842"]
TARGETS = ["vdn", "qmix", "iql", "mappo", "ippo"]


def run_specialized(target: str) -> None:
    target_set_eval(
        expt_path=TASK_ROOT,
        expt_basenames=["baseline"],
        env_nickname="mpe-pp",
        num_agents=3,
        algs_to_eval=[f"open_train/poam-vs-{target}"],
        target_algs=[target],
        algs_to_eval_seeds=TRAINING_SEEDS,
        target_algs_seeds=TRAINING_SEEDS,
        src_config_path="src/config/open/open_eval_default.yaml",
        dest_config_folder=f"src/config/temp/specialized_regen_{target}",
        dest_results_name="specialized_poam_50ep_exact_nk_eval",
        skip_existing=False,
        eval_seed=394823,
        load_step_type="best",
        match_training_seeds=True,
        use_condor=False,
        debug=False,
    )


def run_type_conditional(eval_seed: int) -> None:
    type_conditional_classifier_runs_eval(
        expt_path=TASK_ROOT,
        env_nickname="mpe-pp",
        num_agents=3,
        classifier_models_root=(
            f"{TASK_ROOT}/open_train/poam_lstm_classifier_only/models"
        ),
        src_config_path="src/config/open/open_type_conditional_pp.yaml",
        dest_config_folder=f"src/config/temp/type_conditional_regen_{eval_seed}",
        dest_results_name="type_conditional_lstm_nk_eval",
        n_uncontrolled_list=[1, 2],
        skip_existing=True,
        eval_seed=eval_seed,
        classifier_load_step_type="best",
        use_condor=False,
        debug=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["specialized", "type-conditional"])
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--eval-seed", type=int)
    args = parser.parse_args()

    if args.kind == "specialized":
        if args.target is None:
            parser.error("--target is required for specialized evaluations")
        run_specialized(args.target)
    else:
        if args.eval_seed is None:
            parser.error("--eval-seed is required for type-conditional evaluations")
        run_type_conditional(args.eval_seed)


if __name__ == "__main__":
    main()
