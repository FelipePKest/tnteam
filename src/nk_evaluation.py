import datetime
import json
import os
import random
import shutil
import yaml 

import matplotlib.pyplot as plt

from run_eval import perform_eval
from utils.cluster_helpers import submit_to_condor
from utils.nk_eval_utils import get_runs, random_pairs, all_pairs, get_seed_pairs, \
    is_result_written, get_necessary_agent_args, write_temp_config, ENV_CONFIGS
from utils.load_utils import get_expt_paths
                            
USER_INFO = yaml.load(open(os.path.join(os.path.dirname(__file__), "config", "user_info.yaml"), "r"), Loader=yaml.FullLoader) 

def cross_eval(expt_path: str, 
               expt_basenames: list,
               env_nickname: str, 
               num_agents: int, 
               algorithms: list,
               src_config_path: str,
               dest_config_folder: str,
               dest_results_name: str,
               skip_existing: bool = True,
               num_seed_pairs: int = 3, 
               eval_seed: int = 394820,
               load_step_type: str = "best",
               match_selfplay_seeds: bool=True, # whether to match seeds for self play evals or not
               selfplay_ony: bool=False, # whether to only evaluate self play agents (alg vs alg)
               use_condor: bool = False,
               debug: bool=False):
    # Create the open_eval folder if it doesn't exist
    open_eval_path = os.path.join(expt_path, dest_results_name)
    os.makedirs(open_eval_path, exist_ok=True)
    
    for i in range(len(algorithms)):
        for j in range(i, len(algorithms)):
            if i != j and selfplay_ony: continue
            algo_i_path = algorithms[i]
            algo_j_path = algorithms[j]
            
            # sort alphabetically based on algo name 
            algo_i_path, algo_j_path = sorted([algo_i_path, algo_j_path], key=lambda x: x.split(os.path.sep)[-1])
            algo_i_dir, algo_i = os.path.split(algo_i_path)
            algo_j_dir, algo_j = os.path.split(algo_j_path)

            print(f"\nCHECKING {algo_i} VS {algo_j}")

            # Create log folder based on algo names, alphabetically sorted
            log_folder = os.path.join(open_eval_path, f"{algo_i}-vs-{algo_j}")
            os.makedirs(log_folder, exist_ok=True)
            
            # Get all seeds of algo i and j
            runs_i = get_runs(algo_i, os.path.join(expt_path, algo_i_dir), expt_basenames)
            runs_j = get_runs(algo_j, os.path.join(expt_path, algo_j_dir), expt_basenames)

            # Get previously evaluated seed pairs and sample new if needed
            if skip_existing and os.path.exists(os.path.join(log_folder, "sacred")):
                existing_seed_pairs = get_seed_pairs(log_folder)
                print("Existing seed pairs:", existing_seed_pairs)
                print("Runs i", runs_i)
                print("Log folder: ", log_folder)
                run_pairs = random_pairs(
                        algo_i, algo_j,
                        runs_i, runs_j, 
                        num_pairs=num_seed_pairs,
                        existing_seed_pairs=existing_seed_pairs,
                        match_selfplay_seeds=match_selfplay_seeds 
                        )
                print("Generated run pairs: ", run_pairs)
                print("Number generated run pairs: ", len(run_pairs))
            else: 
                run_pairs = random_pairs(algo_i, algo_j, 
                                         runs_i, runs_j,
                                         num_pairs=num_seed_pairs,
                                         match_selfplay_seeds=match_selfplay_seeds)

            # Iterate over the paired seed lists to perform eval
            for m, (algo_i_path, algo_j_path) in enumerate(run_pairs):
                algo_i_specific_args = get_necessary_agent_args(algo_i_path)
                algo_j_specific_args = get_necessary_agent_args(algo_j_path)
                
                for k in range(1, num_agents):
                    # Rewrite config as needed
                    dest_config_path = os.path.join(f'{dest_config_folder}', 
                                f'{env_nickname}_{algo_i}-vs-{algo_j}_n-{k}_runpair{m}.yaml')
                    
                    expt_label = write_temp_config(
                                    env_nickname, 
                                    results_path=log_folder, 
                                    src_config_path=src_config_path,
                                    dest_config_path=dest_config_path,
                                    k=k, num_agents=num_agents, 
                                    algo_i_path=algo_i_path, 
                                    algo_j_path=algo_j_path, 
                                    algo_i_specific_args=algo_i_specific_args,
                                    algo_j_specific_args=algo_j_specific_args,
                                    load_step_type=load_step_type
                                    )
                    # Check if the algorithms have already been evaluated
                    if skip_existing and is_result_written(log_folder, expt_label):
                        print(f"Skipping existing evaluation for {algo_i} vs {algo_j}, seed pair {m}, k={k}")
                        if m > num_seed_pairs - 1: 
                            res_paths = get_expt_paths(base_folder=log_folder, subfolder="sacred", expt_regex=expt_label)
                            # remove paths in extra paths (are folders)
                            print(f"EXTRA SEED PAIR DISCOVERED. REMOVING EXTRA FILES FROM {res_paths}")
                            for res_path in res_paths:
                                shutil.rmtree(res_path)
                        continue
                    else:
                        print(f"Performing evaluation for {algo_i} vs {algo_j}, seed pair {m}, k={k}")
                        if use_condor:
                            perform_eval_condor(env_nickname, 
                                                dest_config_path, 
                                                log_folder=log_folder, 
                                                expt_label=expt_label,
                                                condor_log_folder=open_eval_path,
                                                eval_seed=eval_seed, 
                                                debug=debug
                                                )
                        else: # directly perform the eval
                            perform_eval(env_nickname, dest_config_path,
                                         eval_seed=eval_seed,
                                         debug=debug)
    return       
             
def target_set_eval(expt_path: str, 
               expt_basenames: list,
               env_nickname: str, 
               num_agents: int, 
               algs_to_eval: list,
               target_algs: list,
               algs_to_eval_seeds: list,
               target_algs_seeds: list,
               src_config_path: str,
               dest_config_folder: str,
               dest_results_name: str,
               skip_existing: bool = True,
               eval_seed: int = 394820,
               load_step_type: str = "best",
               use_condor: bool = False,
               debug: bool=False):
    # Create the open_eval folder if it doesn't exist
    open_eval_path = os.path.join(expt_path, dest_results_name)
    os.makedirs(open_eval_path, exist_ok=True)
    
    for i in range(len(algs_to_eval)):
        for j in range(len(target_algs)):
            algo_to_eval_path = algs_to_eval[i]
            algo_target_path = target_algs[j]
            
            algo_to_eval_dir, algo_to_eval = os.path.split(algo_to_eval_path)
            algo_target_dir, algo_target = os.path.split(algo_target_path)
            print(f"\nCHECKING {algo_to_eval} VS {algo_target}")

            # Create log folder based on algo names, alphabetically sorted
            log_folder = os.path.join(open_eval_path, f"{algo_to_eval}-vs-{algo_target}")
            os.makedirs(log_folder, exist_ok=True)
        
            # Filter by the seeds that are provided
            eval_expt_basenames_seeds = [f"{expt_basename}_seed={seed}" for expt_basename in expt_basenames for seed in algs_to_eval_seeds]
            target_expt_basenames_seeds = [f"{expt_basename}_seed={seed}" for expt_basename in expt_basenames for seed in target_algs_seeds]

            # Get all relevant seeds for the algorithms 
            runs_to_eval = get_runs(algo_to_eval, os.path.join(expt_path, algo_to_eval_dir), 
                                    expt_basenames=eval_expt_basenames_seeds)
            runs_target = get_runs(algo_target, os.path.join(expt_path, algo_target_dir), 
                                   expt_basenames=target_expt_basenames_seeds)

            # Get previously evaluated seed pairs and sample new if needed
            if skip_existing and os.path.exists(os.path.join(log_folder, "sacred")):
                existing_seed_pairs = get_seed_pairs(log_folder)
                print("Existing seed pairs:", existing_seed_pairs)
                print("Log folder: ", log_folder)
                run_pairs = all_pairs(
                        algo_to_eval, algo_target,
                        runs_to_eval, runs_target, 
                        existing_seed_pairs=existing_seed_pairs
                        ) 
            else: 
                run_pairs = all_pairs(algo_to_eval, algo_target, 
                                      runs_to_eval, runs_target)

            print("All pairs: ", run_pairs)
            # Iterate over the paired seed lists to perform eval
            for m, (algo_to_eval_path, algo_target_path) in enumerate(run_pairs):
                # assumption is that the run-specific args are either 
                # the same for agent i and j or don't matter
                eval_specific_args = get_necessary_agent_args(algo_to_eval_path)
                target_specific_args = get_necessary_agent_args(algo_target_path)
                for k in range(1, num_agents):
                    # Rewrite config as needed
                    dest_config_path = os.path.join(f'{dest_config_folder}', 
                                f'{env_nickname}_{algo_to_eval}-vs-{algo_target}_n-{k}_runpair{m}.yaml')
                    
                    expt_label = write_temp_config(
                                    env_nickname, 
                                    results_path=log_folder, 
                                    src_config_path=src_config_path,
                                    dest_config_path=dest_config_path,
                                    k=k, num_agents=num_agents, 
                                    algo_i_path=algo_to_eval_path, 
                                    algo_j_path=algo_target_path, 
                                    algo_i_specific_args=eval_specific_args,
                                    algo_j_specific_args=target_specific_args,
                                    load_step_type=load_step_type
                                    )
                    # Check if the algorithms have already been evaluated
                    if skip_existing and is_result_written(log_folder, expt_label):
                        print(f"Skipping existing evaluation for {algo_to_eval} vs {algo_target}, seed pair {m}, k={k}")
                        continue
                    else:
                        print(f"Performing evaluation for {algo_to_eval} vs {algo_target}, seed pair {m}, k={k}")
                        if use_condor:
                            perform_eval_condor(env_nickname, 
                                                dest_config_path, 
                                                log_folder=log_folder, 
                                                expt_label=expt_label,
                                                condor_log_folder=open_eval_path,
                                                eval_seed=eval_seed, 
                                                debug=debug
                                                )
                            
                        else: # directly perform the eval
                            perform_eval(env_nickname, dest_config_path,
                                         eval_seed=eval_seed,
                                         debug=debug)
    return

def write_type_conditional_temp_config(env_nickname: str,
                                       results_path: str,
                                       src_config_path: str,
                                       dest_config_path: str,
                                       classifier_checkpoint: str,
                                       classifier_cfg: dict,
                                       expert_cfgs: list,
                                       uncntrl_agent_name: str,
                                       uncntrl_agent_cfg: dict,
                                       k: int,
                                       num_agents: int):
    """Write a temp config for classifier-conditioned POAM expert evaluation."""
    with open(src_config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    conf['env'] = ENV_CONFIGS[env_nickname]["env_config"]
    conf['local_results_path'] = results_path
    conf['test_verbose'] = False
    conf['log_discounted_return'] = False
    conf['test_nepisode'] = 128
    conf['eval_mode'] = "open"
    conf['n_uncontrolled'] = k

    conf['trained_agents'] = {
        'agent_classifier_pool': {
            'agent_loader': 'type_conditional_loader',
            'n_agents_to_populate': num_agents,
            'classifier': {
                'checkpoint': classifier_checkpoint,
                'track_per_timestep': True,
                **classifier_cfg,
            },
            'teammate_types': expert_cfgs,
        }
    }
    conf['uncntrl_agents'] = {uncntrl_agent_name: uncntrl_agent_cfg}

    clean_name = uncntrl_agent_name
    if clean_name.startswith("agent_"):
        clean_name = clean_name[len("agent_"):]
    conf['label'] = f"type_conditional_{clean_name}_n-{k}"

    os.makedirs(os.path.dirname(dest_config_path), exist_ok=True)
    with open(dest_config_path, "w") as f:
        yaml.dump(conf, f)

    return conf['label']


def poam_expert_vs_type_eval(expt_path: str,
                             env_nickname: str,
                             num_agents: int,
                             expert_cfgs: list,
                             uncntrl_agents_config: dict,
                             src_config_path: str,
                             dest_config_folder: str,
                             dest_results_name: str,
                             n_uncontrolled_list=None,
                             skip_existing: bool = True,
                             eval_seed: int = 394820,
                             load_step_type: str = "best",
                             use_condor: bool = False,
                             debug: bool = False):
    """
    Evaluate each teammate-specific POAM expert directly against each uncontrolled
    agent type across the standard N-k sweep.
    """
    open_eval_path = os.path.join(expt_path, dest_results_name)
    os.makedirs(open_eval_path, exist_ok=True)
    if n_uncontrolled_list is None:
        n_uncontrolled_list = list(range(1, num_agents))

    for expert_cfg in expert_cfgs:
        expert_name = expert_cfg["name"]
        expert_model_path = expert_cfg["agent_path"]
        expert_specific_args = get_necessary_agent_args(expert_model_path)

        for uncntrl_agent_name, uncntrl_agent_cfg in uncntrl_agents_config.items():
            target_name = uncntrl_agent_name
            if target_name.startswith("agent_"):
                target_name = target_name[len("agent_"):]

            target_model_path = uncntrl_agent_cfg["agent_path"]
            target_specific_args = get_necessary_agent_args(target_model_path)
            log_folder = os.path.join(open_eval_path, f"poam-expert-{expert_name}-vs-{target_name}")
            os.makedirs(log_folder, exist_ok=True)
            print(f"\nCHECKING POAM EXPERT {expert_name} VS {target_name}")

            for k in n_uncontrolled_list:
                dest_config_path = os.path.join(
                    dest_config_folder,
                    f"{env_nickname}_poam-expert-{expert_name}-vs-{target_name}_n-{k}.yaml",
                )

                expt_label = write_temp_config(
                    env_nickname=env_nickname,
                    results_path=log_folder,
                    src_config_path=src_config_path,
                    dest_config_path=dest_config_path,
                    k=k,
                    num_agents=num_agents,
                    algo_i_path=expert_model_path,
                    algo_j_path=target_model_path,
                    algo_i_specific_args=expert_specific_args,
                    algo_j_specific_args=target_specific_args,
                    load_step_type=load_step_type,
                )

                if skip_existing and is_result_written(log_folder, expt_label):
                    print(f"Skipping existing POAM expert evaluation for {expert_name} vs {target_name}, k={k}")
                    continue

                print(f"Performing POAM expert evaluation for {expert_name} vs {target_name}, k={k}")
                if use_condor:
                    perform_eval_condor(
                        env_nickname,
                        dest_config_path,
                        log_folder=log_folder,
                        expt_label=expt_label,
                        condor_log_folder=open_eval_path,
                        eval_seed=eval_seed,
                        debug=debug,
                        alg_config="open_dummy",
                    )
                else:
                    perform_eval(
                        env_nickname,
                        dest_config_path,
                        eval_seed=eval_seed,
                        debug=debug,
                        alg_config="open_dummy",
                    )
    return


def _find_sacred_run_dir(log_folder: str, expt_label: str):
    paths = get_expt_paths(base_folder=log_folder, subfolder="sacred", expt_regex=expt_label)
    if not paths:
        return None
    return os.path.join(sorted(paths)[-1], "1")


def _split_prediction_episodes(predictions: list):
    if not predictions:
        return []
    episodes = []
    current_episode = []
    prev_key = None
    prev_timestep = None
    for entry in predictions:
        timestep = entry.get("timestep")
        env_idx = entry.get("env_idx", 0)
        current_key = (env_idx,)
        if current_episode and (
            timestep is None
            or prev_timestep is None
            or timestep < prev_timestep
            or (timestep == 0 and current_key == prev_key)
        ):
            episodes.append(current_episode)
            current_episode = []
        current_episode.append(entry)
        prev_timestep = timestep
        prev_key = current_key
    if current_episode:
        episodes.append(current_episode)
    return episodes


def _plot_prediction_episode(predictions: list, output_path: str, title: str):
    if not predictions:
        return

    episodes = _split_prediction_episodes(predictions)
    if not episodes:
        return
    episode = max(episodes, key=len)

    type_names = []
    for entry in episode:
        pred_name = entry.get("prediction_type")
        true_name = entry.get("ground_truth_type")
        if pred_name is not None and pred_name not in type_names:
            type_names.append(pred_name)
        if true_name is not None and true_name not in type_names:
            type_names.append(true_name)

    if not type_names:
        idx_values = []
        for entry in episode:
            for key in ("prediction_idx", "ground_truth_idx", "prediction", "ground_truth"):
                val = entry.get(key)
                if isinstance(val, int) and val >= 0 and val not in idx_values:
                    idx_values.append(val)
        type_names = [str(idx) for idx in sorted(idx_values)]

    type_to_y = {name: idx for idx, name in enumerate(type_names)}
    timesteps = [entry["timestep"] for entry in episode]
    pred_y = []
    true_y = []
    correct_x = []
    correct_y = []
    incorrect_x = []
    incorrect_y = []

    for entry in episode:
        pred_name = entry.get("prediction_type")
        true_name = entry.get("ground_truth_type")
        pred_idx = entry.get("prediction_idx", entry.get("prediction"))
        true_idx = entry.get("ground_truth_idx", entry.get("ground_truth"))
        if pred_name is None and isinstance(pred_idx, int) and pred_idx >= 0:
            pred_name = str(pred_idx)
        if true_name is None and isinstance(true_idx, int) and true_idx >= 0:
            true_name = str(true_idx)
        pred_y.append(type_to_y.get(pred_name, -1))
        true_y.append(type_to_y.get(true_name, -1))

        is_correct = entry.get("is_correct")
        if is_correct is None and pred_y[-1] >= 0 and true_y[-1] >= 0:
            is_correct = pred_y[-1] == true_y[-1]
        if is_correct is True:
            correct_x.append(entry["timestep"])
            correct_y.append(pred_y[-1])
        elif is_correct is False:
            incorrect_x.append(entry["timestep"])
            incorrect_y.append(pred_y[-1])

    plt.figure(figsize=(12, 5))
    plt.step(timesteps, true_y, where="post", label="Ground truth", linewidth=2.0, color="black", alpha=0.7)
    plt.plot(timesteps, pred_y, label="Predicted type", linewidth=1.5, color="tab:blue", alpha=0.65)
    if correct_x:
        plt.scatter(correct_x, correct_y, color="tab:green", s=36, label="Correct", zorder=3)
    if incorrect_x:
        plt.scatter(incorrect_x, incorrect_y, color="tab:red", s=36, label="Incorrect", zorder=3)
    plt.yticks(range(len(type_names)), type_names)
    plt.xlabel("Timestep")
    plt.ylabel("Uncontrolled type")
    plt.title(title)
    plt.grid(alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _prediction_is_correct(entry: dict):
    is_correct = entry.get("is_correct")
    if is_correct is not None:
        return bool(is_correct)

    pred_type = entry.get("prediction_type")
    true_type = entry.get("ground_truth_type")
    if pred_type is not None and true_type is not None:
        return pred_type == true_type

    pred_idx = entry.get("prediction_idx", entry.get("prediction"))
    true_idx = entry.get("ground_truth_idx", entry.get("ground_truth"))
    if pred_idx is None or true_idx is None:
        return None
    return pred_idx == true_idx


def _plot_prediction_accuracy_over_timesteps(
    predictions: list,
    output_path: str,
    title: str,
    random_chance: float = 0.2,
):
    timestep_counts = {}
    timestep_correct = {}
    for entry in predictions:
        timestep = entry.get("timestep")
        if timestep is None:
            continue
        is_correct = _prediction_is_correct(entry)
        if is_correct is None:
            continue
        timestep_counts[timestep] = timestep_counts.get(timestep, 0) + 1
        timestep_correct[timestep] = timestep_correct.get(timestep, 0) + int(is_correct)

    if not timestep_counts:
        return

    timesteps = sorted(timestep_counts)
    accuracies = [
        timestep_correct[timestep] / timestep_counts[timestep]
        for timestep in timesteps
    ]

    plt.figure(figsize=(12, 5))
    plt.plot(timesteps, accuracies, linewidth=2, color="tab:blue", label="Classifier accuracy")
    plt.axhline(
        random_chance,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Random ({random_chance:.1%})",
    )
    plt.ylim(0, 1)
    plt.xlabel("Episode Timestep")
    plt.ylabel("Classification Accuracy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _plot_type_conditional_predictions(log_folder: str, expt_label: str, num_types: int = 5):
    run_dir = _find_sacred_run_dir(log_folder, expt_label)
    if run_dir is None:
        return

    predictions_path = os.path.join(run_dir, "per_timestep_predictions.json")
    if not os.path.exists(predictions_path):
        return

    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    if not predictions:
        return

    plots_dir = os.path.join(log_folder, "prediction_plots")
    episode_output_path = os.path.join(plots_dir, f"{expt_label}_episode_predictions.png")
    episode_title = f"{expt_label}: classifier predictions over episode"
    _plot_prediction_episode(predictions, episode_output_path, episode_title)
    print(f"Wrote prediction plot to {episode_output_path}")

    accuracy_output_path = os.path.join(plots_dir, f"{expt_label}_accuracy_over_timesteps.png")
    accuracy_title = f"{expt_label}: classifier accuracy over episode timesteps"
    _plot_prediction_accuracy_over_timesteps(
        predictions,
        accuracy_output_path,
        accuracy_title,
        random_chance=1.0 / num_types,
    )
    print(f"Wrote classifier accuracy plot to {accuracy_output_path}")

def type_conditional_eval(expt_path: str,
                          env_nickname: str,
                          num_agents: int,
                          classifier_checkpoint: str,
                          classifier_cfg: dict,
                          expert_cfgs: list,
                          uncntrl_agents_config: dict,
                          src_config_path: str,
                          dest_config_folder: str,
                          dest_results_name: str,
                          n_uncontrolled_list = None,
                          skip_existing: bool = True,
                          eval_seed: int = 394820,
                          use_condor: bool = False,
                          debug: bool = False):
    """
    Evaluate a type_conditional_loader policy in the N-k setting.

    The trained side is a classifier plus a POAM expert pool. For every action
    query, the classifier predicts the uncontrolled type and the matching POAM
    expert is forced, matching open_type_conditional evaluation.
    """
    open_eval_path = os.path.join(expt_path, dest_results_name)
    os.makedirs(open_eval_path, exist_ok=True)
    if n_uncontrolled_list is None:
        n_uncontrolled_list = list(range(1, num_agents))

    for uncntrl_agent_name, uncntrl_agent_cfg in uncntrl_agents_config.items():
        clean_name = uncntrl_agent_name
        if clean_name.startswith("agent_"):
            clean_name = clean_name[len("agent_"):]

        log_folder = os.path.join(open_eval_path, f"type_conditional-vs-{clean_name}")
        os.makedirs(log_folder, exist_ok=True)
        print(f"\nCHECKING TYPE_CONDITIONAL VS {clean_name}")

        for k in n_uncontrolled_list:
            dest_config_path = os.path.join(
                dest_config_folder,
                f'{env_nickname}_type_conditional-vs-{clean_name}_n-{k}.yaml',
            )
            expt_label = write_type_conditional_temp_config(
                env_nickname=env_nickname,
                results_path=log_folder,
                src_config_path=src_config_path,
                dest_config_path=dest_config_path,
                classifier_checkpoint=classifier_checkpoint,
                classifier_cfg=classifier_cfg,
                expert_cfgs=expert_cfgs,
                uncntrl_agent_name=uncntrl_agent_name,
                uncntrl_agent_cfg=uncntrl_agent_cfg,
                k=k,
                num_agents=num_agents,
            )

            if skip_existing and is_result_written(log_folder, expt_label):
                print(f"Skipping existing type-conditional evaluation for {clean_name}, k={k}")
                _plot_type_conditional_predictions(log_folder, expt_label, num_types=len(expert_cfgs))
                continue

            print(f"Performing type-conditional evaluation for {clean_name}, k={k}")
            if use_condor:
                perform_eval_condor(env_nickname,
                                    dest_config_path,
                                    log_folder=log_folder,
                                    expt_label=expt_label,
                                    condor_log_folder=open_eval_path,
                                    eval_seed=eval_seed,
                                    debug=debug,
                                    alg_config="open_dummy")
            else:
                perform_eval(env_nickname,
                             dest_config_path,
                             eval_seed=eval_seed,
                             debug=debug,
                             alg_config="open_dummy")
                _plot_type_conditional_predictions(log_folder, expt_label, num_types=len(expert_cfgs))
    return

def perform_eval_condor(env_nickname, dest_config_path, 
                        log_folder, expt_label, condor_log_folder,
                        eval_seed, debug, alg_config="open_dummy"):
    '''Logic to evaluate algorithm i against algorithm j is wrapped into a script
    and submitted to condor.
    '''
    exec_cmd = "src/run_eval.py"
    expt_params = {
        "env": env_nickname,
        "dest_config_path": dest_config_path,
        "log_folder": log_folder,
        "expt_label": expt_label,
        "eval_seed": eval_seed,
        "alg_config": alg_config,
        "debug": debug
    }
    if debug: 
        print("NK_EVALUATION.PY: exec_cmd=", exec_cmd)
        print("NK_EVALUATION.PY: expt_params=", expt_params)
        return
    else:
        # make condor log folder if it doesn't exit
        os.makedirs(condor_log_folder, exist_ok=True)
        submit_to_condor(env_id=env_nickname, 
                         exec_cmd=exec_cmd, 
                         results_dir=condor_log_folder,
                         job_name=expt_label, 
                         expt_params=expt_params, 
                         user_email=USER_INFO["email"],
                         sleep_time=5, 
                         print_without_submit=False # print sub file
                         )        
        
if __name__ == "__main__":
    '''
    Inputs: 
        - Domain
        - Number total agents
        - List of algorithms to be evaluated against each other
        - Path to input algorithms
        - Number of run pairings to evaluate 
        
    Outputs:
        - Output + Source monitoring via Sacred: 

    Procedure: N-K eval for each pair of algorithms, where N is the number of agents and k is the number of uncontrolled agents.
        - For  I in algo list: 
            ○ For j in algo list: 
                § Create log folder based on algo names, alphabetically sorted.
                § If I, j or j, I has been evaluated already, continue. 
                § Get all seeds of algo I and j (there should be the same number). To prevent quadratic complexity in seeds, pair the seeds at random. 
                § Iterating over the paired seed lists:
                    □ For k=1, …, n-1, do: 
                        ® Rewrite config as needed
                        ® Eval algorithm I against algorithm j
                        ® (Results will be written by sacred)
                        ® Check that result was written by sacred.
                § Write result to log folder. 
        
    Does order matter? i.e. does algo pair (i, j) differ from (j, i)?
        - No, since we randomize the order of agents in the team and iterate from 1, cdots, n-1
    Does k refer to the number of agents generated by algo i or algo j? 
        - k refers to the number of uncontrolled agents, generated by algorithm j (2nd algo)
    '''
    # set seed 
    random.seed(0)
    EVAL_SEED = 394820

    DEBUG = False # toggle this to debug the script without running the evaluation
    USE_CONDOR = False # toggle this to run the evaluation on condor
    RUN_TYPE_CONDITIONAL = True
    RUN_POAM_EXPERT_VS_TYPES = False
    RUN_TARGET_SET = False
    base_path = USER_INFO['base_results_path']
    
    # expt_dir = "5v6"
    # num_agents = 5
    # env_nickname = "5v6"

    # expt_dir = "8v9" 
    # num_agents = 8
    # env_nickname = "8v9"

    # expt_dir = "10v11"
    # num_agents = 10
    # env_nickname = "10v11"

    # expt_dir = "3sv5z"
    # num_agents = 3
    # env_nickname = "3sv5z"

    # expt_dir = "mpe-pp/ts=100_shape=0.01"
    expt_dir = "mpe-pp"
    num_agents = 3
    env_nickname = "mpe-pp"


    algorithms = [
        "vdn", 
        "qmix",
        "iql",
        "mappo",
        "ippo",
        # "open_train/ippo-pqvmq_aht",
        # "open_train/poam-pqvmq_aht",
        # "open_train/ippo-pqvmq_open",
        # "open_train/poam-pqvmq_open",
        # "open_train/poam-pqvmq_open",
        ]


    if RUN_TYPE_CONDITIONAL:
        classifier_cfg = {
            "history_len": 32,
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "ff": 256,
            "dropout": 0.1,
        }
        poam_expert_cfgs = [
            {
                "name": "ippo",
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/open_train/poam-pqvmq_open/models/poam_baseline_seed=112358_04-02-03-00-00",
                "load_step": "best",
                "test_mode": True,
            },
            {
                "name": "qmix",
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/open_train/poam-pqvmq_open/models/poam_baseline_seed=112358_04-01-10-29-18",
                "load_step": "best",
                "test_mode": True,
            },
            {
                "name": "vdn",
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/open_train/poam-pqvmq_open/models/poam_baseline_seed=112358_04-01-01-07-32",
                "load_step": "best",
                "test_mode": True,
            },
            {
                "name": "mappo",
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/open_train/poam-pqvmq_open/models/poam_baseline_seed=112358_03-31-11-48-33",
                "load_step": "best",
                "test_mode": True,
            },
            {
                "name": "iql",
                "agent_loader": "poam_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/open_train/poam-pqvmq_open/models/poam_baseline_seed=112358_03-30-16-09-21",
                "load_step": "best",
                "test_mode": True,
            },
        ]
        type_conditional_uncntrl_agents = {
            "agent_ippo": {
                "agent_loader": "rnn_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/ippo/models/ippo_baseline_seed=112358_07-10-14-39-17",
                "load_step": "best",
                "n_agents_to_populate": 3,
                "test_mode": True,
            },
            "agent_qmix": {
                "agent_loader": "rnn_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/qmix/models/qmix_baseline_seed=112358_04-24-12-09-32",
                "load_step": "best",
                "n_agents_to_populate": 3,
                "test_mode": True,
            },
            "agent_vdn": {
                "agent_loader": "rnn_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/vdn/models/vdn_baseline_seed=112358_04-24-12-11-09",
                "load_step": "best",
                "n_agents_to_populate": 3,
                "test_mode": True,
            },
            "agent_mappo": {
                "agent_loader": "rnn_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/mappo/models/mappo_baseline_seed=112358_04-24-12-04-31",
                "load_step": "best",
                "n_agents_to_populate": 3,
                "test_mode": True,
            },
            "agent_iql": {
                "agent_loader": "rnn_eval_agent_loader",
                "agent_path": "naht_results/mpe-pp/iql/models/iql_baseline_seed=112358_04-24-12-11-40",
                "load_step": "best",
                "n_agents_to_populate": 3,
                "test_mode": True,
            },
        }

        if RUN_POAM_EXPERT_VS_TYPES:
            poam_expert_vs_type_eval(
                expt_path=os.path.join(base_path, expt_dir),
                env_nickname=env_nickname,
                num_agents=num_agents,
                expert_cfgs=poam_expert_cfgs,
                uncntrl_agents_config=type_conditional_uncntrl_agents,
                src_config_path="src/config/open/open_eval_default.yaml",
                dest_config_folder=f"src/config/temp/poam_expert_eval_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/",
                dest_results_name="poam_expert_nk_eval",
                n_uncontrolled_list=[1, 2],
                skip_existing=True,
                eval_seed=EVAL_SEED,
                load_step_type="best",
                use_condor=USE_CONDOR,
                debug=DEBUG,
            )
        else:
            type_conditional_eval(
                expt_path=os.path.join(base_path, expt_dir),
                env_nickname=env_nickname,
                num_agents=num_agents,
                classifier_checkpoint="/Users/felipekestelman/Git/tnteam/classifier.th",
                classifier_cfg=classifier_cfg,
                expert_cfgs=poam_expert_cfgs,
                uncntrl_agents_config=type_conditional_uncntrl_agents,
                src_config_path="src/config/open/open_type_conditional_pp.yaml",
                dest_config_folder=f"src/config/temp/type_conditional_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/",
                dest_results_name="type_conditional_nk_eval",
                n_uncontrolled_list=[1, 2],
                skip_existing=True,
                eval_seed=EVAL_SEED,
                use_condor=USE_CONDOR,
                debug=DEBUG,
            )

    if RUN_TARGET_SET:
        target_set_eval(expt_path=os.path.join(base_path, expt_dir),
                        expt_basenames=["baseline"],
                        env_nickname=env_nickname,
                        num_agents=num_agents,
                        algs_to_eval=[
                            # "open_train/ippo-pqvmq_open",
                            "open_train/poam-pqvmq_open",
                            # "open_train/poam-pqvmq_aht", # for naht vs aht comparison only
                            # "open_train/ippo-qmq-3trainseeds",
                            # "open_train/poam-qmq-3trainseeds",
                                      ],
                        target_algs=["vdn", "qmix", "iql", "mappo", "ippo"],
                        # target_algs=["vdn", "ippo"], # for ood alt train/test split
                        # algs_to_eval_seeds=["112358", "1285842", "78590", "38410", "93718"],
                        algs_to_eval_seeds=["112358"], # for in-distribution eval
                        # target_algs_seeds=["1285842", "78590", "38410", "93718"],# not eval on 112358 because that's the training set
                        target_algs_seeds=["112358"], # for in-distribution eval
                        # target_algs_seeds=["112358", "1285842", "78590", "38410", "93718"], # for alt train/test split
                        src_config_path="src/config/open/open_eval_default.yaml",
                        dest_config_folder=f"src/config/temp/temp_{datetime.datetime.now().strftime('%m-%d-%H-%M-%S')}/",
                        # dest_results_name="ood_generalization",
                        dest_results_name="in_distr_eval",
                        # dest_results_name="ood_gen_vp",
                        eval_seed=EVAL_SEED,
                        load_step_type="best",
                        use_condor=USE_CONDOR,
                        debug=DEBUG
                        )
