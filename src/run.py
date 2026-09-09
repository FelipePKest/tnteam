import os
import json
import pprint
import time
import threading
import math
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.load_utils import find_model_path
from os.path import dirname, abspath
from os import makedirs

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # Create the local results directory
    if args.local_results_path == "":
        args.local_results_path = dirname(dirname(abspath(__file__)))
    makedirs(args.local_results_path, exist_ok=True)

    # setup loggers
    logger = Logger(_log)

    if args.eval_mode == "open":
        # force sacred to dump to log every 5 seconds
        _run.beat_interval = 5

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    if args.use_tensorboard:
        tb_log_dir = os.path.join(args.local_results_path, "tb_logs", args.expt_logname)
        logger.setup_tb(tb_log_dir)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    start_time = time.time()
    while True:
        runner.run(test_mode=True)
        # when num_test_episodes have run, the test_stats buffer will be cleared
        if len(runner.test_stats) == 0:
            break

    if args.eval_mode == "open":
        sacred_log_path = os.path.join(runner.logger._run_obj.observers[0].dir, "info.json")
        while not os.path.exists(sacred_log_path):
            time.sleep(1)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()
    end_time = time.time()
    print("Evaluation took {} seconds".format(end_time - start_time))

def run_sequential(args, logger):
    print("ENV IS : ", args.env)
    args.open_train_or_eval = True if "open" in args.mac else False
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.episode_limit = env_info["episode_limit"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "actor_hidden_states": {"vshape": (args.hidden_dim,), 
                                "group": "agents"},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8}
    }
    if args.open_train_or_eval: # track a mask relating to whether agents are trainable or not
        scheme["trainable_agents"] = {"vshape": (1,), "group": "agents", "dtype": th.bool}
        scheme["uncontrolled_team_idx"] = {
            "vshape": (1,),
            "dtype": th.long,
            "episode_const": True,
        }
        if getattr(args, "n_policy_types", 0) > 0:
            scheme["policy_type"] = {"vshape": (1,), "group": "agents", "dtype": th.long}
    
    if "liam" in args.name or "poam" in args.name and not args.open_train_or_eval:
        scheme['actor_hidden_states']['vshape'] = (args.hidden_dim, 2) 
    
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    buffer_size = args.buffer_size
    is_world_model_learner = args.learner in {
        "matwm_learner", "marie_learner"
    }
    is_reference_marie = args.learner == "marie_reference_learner"
    if is_world_model_learner:
        capacity_steps = getattr(args, "matwm_replay_capacity_steps", None)
        if capacity_steps is not None:
            buffer_size = max(
                args.batch_size,
                int(math.ceil(capacity_steps / float(env_info["episode_limit"])))
            )
    buffer = ReplayBuffer(
        scheme,
        groups,
        buffer_size,
        env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    latent_ppo_batch_size = getattr(args, "matwm_ppo_batch_episodes", 0)
    use_batched_latent_ppo = (
        is_world_model_learner
        and getattr(args, "matwm_real_policy_algorithm", "awr") == "ppo"
        and latent_ppo_batch_size > args.batch_size_run
    )
    policy_buffer = None
    policy_collections = 0
    if use_batched_latent_ppo:
        policy_buffer = ReplayBuffer(
            scheme,
            groups,
            latent_ppo_batch_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    if args.use_cuda:
        mac.cuda()

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    assert args.eval_mode in ["default", "open", None]
    if args.eval_mode != "open":
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

        if args.use_cuda:
            learner.cuda()

    if args.checkpoint_path != "":
        model_path, timestep_to_load = find_model_path(args.checkpoint_path, args.load_step, logger=logger)
        logger.console_logger.info(f"Loading model from ts {timestep_to_load}, {model_path}")
        learner.load_models(model_path)
        if getattr(args, "resume_t_env_from_checkpoint", False):
            runner.t_env = timestep_to_load

    if args.eval_mode in ["default", "open"] or args.save_replay:
        runner.log_train_stats_t = runner.t_env
        evaluate_sequential(args, runner)
        logger.log_stat("episode", runner.t_env, runner.t_env)
        logger.print_recent_stats()
        logger.console_logger.info("Finished Evaluation")
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    best_test_return = -1000000
    best_test_win_rate = -1.0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    interleaved_updates = (
        is_world_model_learner
        and getattr(args, "matwm_interleaved_updates", False)
        and not use_batched_latent_ppo
        and getattr(args, "matwm_real_policy_algorithm", "awr") != "ppo"
    )
    interleaved_update_credit = 0.0
    canonical_marie = (
        args.learner == "marie_learner"
        and getattr(args, "marie_original_procedure", False)
    )
    marie_new_transitions = 0

    def train_matwm_replay_once():
        if getattr(args, "matwm_sequence_replay", False):
            sequence_length = getattr(args, "matwm_max_seq_length", 64)
            if not buffer.can_sample_sequences(args.batch_size, sequence_length):
                return False
            episode_sample = buffer.sample_sequences(
                args.batch_size,
                sequence_length,
                recency_decay=getattr(args, "matwm_replay_decay", None),
            )
        else:
            if not buffer.can_sample(args.batch_size):
                return False
            episode_sample = buffer.sample(
                args.batch_size,
                recency_decay=getattr(args, "matwm_replay_decay", None),
            )
        max_ep_t = episode_sample.max_t_filled()
        episode_sample = episode_sample[:, :max_ep_t]
        if episode_sample.device != args.device:
            episode_sample.to(args.device)
        learner.train(episode_sample, runner.t_env, episode)
        return True

    def train_matwm_after_step(steps_collected):
        nonlocal interleaved_update_credit
        update_ratio = getattr(args, "matwm_updates_per_env_step", 1.0)
        if update_ratio is None:
            update_ratio = 1.0
        interleaved_update_credit += steps_collected * float(update_ratio)
        while interleaved_update_credit >= 1.0:
            if not train_matwm_replay_once():
                # Do not create a large catch-up burst while replay is warming up.
                interleaved_update_credit = 0.0
                break
            interleaved_update_credit -= 1.0

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        step_callback = train_matwm_after_step if interleaved_updates else None
        episode_batch, _ = runner.run(
            test_mode=False, step_callback=step_callback
        ) # batch_size_run eps collected
        if is_reference_marie:
            learner.train_episode(episode_batch, runner.t_env, episode)
        else:
            buffer.insert_episode_batch(episode_batch)
        if is_reference_marie:
            pass
        elif canonical_marie:
            marie_new_transitions += runner.env_steps_this_run
            update_interval = getattr(args, "marie_new_samples_per_update", 100)
            minimum_replay = getattr(args, "marie_min_replay_steps", 1000)
            if buffer.marie_transition_count() < minimum_replay:
                # Upstream retains only enough credit for the first update;
                # it does not execute a catch-up burst after replay warm-up.
                marie_new_transitions = min(
                    marie_new_transitions, update_interval
                )
            if (
                marie_new_transitions >= update_interval
                and buffer.marie_transition_count() >= minimum_replay
            ):
                learner.train_from_replay(buffer, runner.t_env, episode)
                marie_new_transitions = 0
        elif use_batched_latent_ppo:
            policy_buffer.insert_episode_batch(episode_batch)
            policy_collections += 1
            if policy_buffer.can_sample(latent_ppo_batch_size):
                policy_batch = policy_buffer.sample(latent_ppo_batch_size)
                max_policy_t = policy_batch.max_t_filled()
                policy_batch = policy_batch[:, :max_policy_t]
                if policy_batch.device != args.device:
                    policy_batch.to(args.device)
                learner.train_real_ppo(policy_batch, runner.t_env, episode)

                recency_decay = getattr(args, "matwm_replay_decay", None)
                if recency_decay is not None:
                    recency_decay = recency_decay ** env_info["episode_limit"]
                queued_updates = (
                    getattr(args, "matwm_updates_per_collect", 1)
                    * policy_collections
                )
                for _ in range(queued_updates):
                    episode_sample = buffer.sample(
                        args.batch_size, recency_decay=recency_decay
                    )
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]
                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)
                    learner.train_world_only(
                        episode_sample, runner.t_env, episode
                    )
                policy_buffer.clear()
                policy_collections = 0
        elif not interleaved_updates and buffer.can_sample(args.batch_size): # when batch_size eps collected
            use_latent_ppo = (
                is_world_model_learner
                and getattr(args, "matwm_real_policy_algorithm", "awr") == "ppo"
            )
            if use_latent_ppo:
                policy_batch = episode_batch
                max_policy_t = policy_batch.max_t_filled()
                policy_batch = policy_batch[:, :max_policy_t]
                if policy_batch.device != args.device:
                    policy_batch.to(args.device)
                learner.train_real_ppo(policy_batch, runner.t_env, episode)

            recency_decay = (
                getattr(args, "matwm_replay_decay", None)
                if is_world_model_learner else None
            )
            if recency_decay is not None:
                # Replay entries are episodes; preserve the paper's per-step decay.
                recency_decay = recency_decay ** env_info["episode_limit"]
            updates_per_collect = 1
            if is_world_model_learner:
                updates_per_env_step = getattr(
                    args, "matwm_updates_per_env_step", None
                )
                if updates_per_env_step is None:
                    updates_per_collect = getattr(
                        args, "matwm_updates_per_collect", 1
                    )
                else:
                    updates_per_collect = max(
                        1,
                        int(math.ceil(
                            runner.env_steps_this_run * updates_per_env_step
                        )),
                    )
            for _ in range(updates_per_collect):
                if (
                    is_world_model_learner
                    and getattr(args, "matwm_sequence_replay", False)
                ):
                    episode_sample = buffer.sample_sequences(
                        args.batch_size,
                        getattr(args, "matwm_max_seq_length", 64),
                        recency_decay=getattr(args, "matwm_replay_decay", None),
                    )
                else:
                    episode_sample = buffer.sample(
                        args.batch_size, recency_decay=recency_decay
                    )

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)
                if use_latent_ppo:
                    learner.train_world_only(
                        episode_sample, runner.t_env, episode
                    )
                else:
                    learner.train(episode_sample, runner.t_env, episode)

            if args.on_policy:
                buffer.clear()

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size_run)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            test_batches = []
            for _ in range(n_test_runs):
                test_batch, mean_test_return = runner.run(test_mode=True)
                test_batches.append(test_batch)

            # Evaluate classifier on test episodes (if learner supports it)
            if hasattr(learner, 'test') and test_batches:
                # Test on each batch and average
                test_accs = []
                for tb in test_batches:
                    max_ep_t = tb.max_t_filled()
                    tb_truncated = tb[:, :max_ep_t]
                    if tb_truncated.device != args.device:
                        tb_truncated.to(args.device)
                    acc = learner.test(tb_truncated, runner.t_env, log=False)
                    if acc is not None:
                        test_accs.append(acc)
                if test_accs:
                    mean_test_acc = sum(test_accs) / len(test_accs)
                    logger.log_stat("classifier_test_acc", mean_test_acc, runner.t_env)
                    logger.console_logger.info(f"Classifier test accuracy: {mean_test_acc:.4f}")

            # save best checkpoint
            assert mean_test_return is not None
            test_win_rate = getattr(runner, "last_test_battle_won", -1.0)
            marie_selection = getattr(args, "marie_original_procedure", False)
            is_better = (
                (test_win_rate, mean_test_return)
                > (best_test_win_rate, best_test_return)
                if marie_selection else mean_test_return > best_test_return
            )
            if is_better:
                best_test_return = mean_test_return
                best_test_win_rate = test_win_rate
                save_path = os.path.join(args.local_results_path, "models", args.expt_logname, "best")
                os.makedirs(save_path, exist_ok=True)
                # make json file with best_test_return
                with open(os.path.join(save_path, "best_info.json"), 'w') as f:
                    json.dump({
                        "best_test_return": best_test_return,
                        "best_test_win_rate": best_test_win_rate,
                        "best_ts": str(runner.t_env),
                    }, f)
                logger.console_logger.info("Saving models to {}".format(save_path))
                learner.save_models(save_path)
        
        # save at regular intervals 
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.expt_logname, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    if hasattr(learner, "close"):
        learner.close()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
