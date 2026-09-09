"""Isolated runtime for the upstream MARIE implementation.

The reference checkout and EPyMARL both expose top-level packages named
``utils`` and ``agent``.  Running MARIE in a child process avoids importing
either repository through the other's module namespace.
"""

import multiprocessing as mp
import os
import traceback

import numpy as np


class MARIEReferenceService:
    def __init__(self, args, obs_dim):
        spec = {
            "root": getattr(args, "marie_reference_root", "/home/jupyter-jphuser3/MARIE"),
            "obs_dim": int(obs_dim),
            "n_agents": int(args.n_agents),
            "n_actions": int(args.n_actions),
            "device": args.device,
            "seed": int(args.seed),
            "run_dir": os.path.abspath(args.local_results_path),
        }
        ctx = mp.get_context("spawn")
        self.parent, child = ctx.Pipe()
        self.process = ctx.Process(target=_serve, args=(child, spec), daemon=True)
        self.process.start()
        status, value = self.parent.recv()
        if status == "error":
            raise RuntimeError("Reference MARIE startup failure:\n" + value)

    def _request(self, command, payload=None):
        if not self.process.is_alive():
            raise RuntimeError("The reference MARIE service stopped unexpectedly")
        self.parent.send((command, payload))
        status, value = self.parent.recv()
        if status == "error":
            raise RuntimeError("Reference MARIE failure:\n" + value)
        return value

    def reset(self, batch_size=1):
        self._request("reset", int(batch_size))

    def act(self, observations, avail_actions):
        return self._request("act", (observations, avail_actions))

    def train(self, rollout):
        return self._request("train", rollout)

    def save(self, path):
        self._request("save", path)

    def load(self, path):
        self._request("load", path)

    def close(self):
        if getattr(self, "process", None) is not None and self.process.is_alive():
            try:
                self._request("close")
            finally:
                self.process.join(timeout=10)


def _serve(connection, spec):
    try:
        import sys
        import random
        import torch

        root = os.path.abspath(spec["root"])
        os.chdir(root)
        sys.path.insert(0, root)
        # A spawned child imports EPyMARL's entry point first. Remove its
        # colliding packages before importing the reference checkout.
        for name in list(sys.modules):
            if name == "utils" or name.startswith(("agent.", "configs.", "networks.", "environments.")):
                sys.modules.pop(name, None)
        for name in ("agent", "configs", "networks", "environments"):
            sys.modules.pop(name, None)

        from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
        from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
        from environments import Env
        import wandb

        # EPyMARL/Sacred owns experiment logging. This only satisfies the
        # reference learner's internal wandb.log calls without a second run.
        wandb.init(mode="disabled")

        seed = spec["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        learner_config = DreamerLearnerConfig()
        controller_config = DreamerControllerConfig()
        for config in (learner_config, controller_config):
            config.IN_DIM = spec["obs_dim"]
            config.ACTION_SIZE = spec["n_actions"]
            config.NUM_AGENTS = spec["n_agents"]
            config.CONTINUOUS_ACTION = False
            config.ACTION_SPACE = None
            config.ENV_TYPE = Env.STARCRAFT
            config.tokenizer_type = "vq"
            config.ema_decay = 0.8
        learner_config.DEVICE = spec["device"]
        learner_config.sample_temperature = "inf"
        learner_config.use_ce_for_r = False
        learner_config.use_ce_for_end = False
        learner_config.use_ce_for_av_action = True
        learner_config.critic_average_r = False
        learner_config.RUN_DIR = spec["run_dir"]
        controller_config.temperature = 1.0

        learner = learner_config.create_learner()
        controller = controller_config.create_controller()
        controller.receive_params(learner.params())
        controller_stacks = []

        def reset_controller_stacks(batch_size):
            nonlocal controller_stacks
            from copy import deepcopy
            controller_stacks = []
            for _ in range(batch_size):
                controller.init_rnns()
                controller_stacks.append(deepcopy(controller.stack_obs))

        reset_controller_stacks(1)
        connection.send(("ok", True))

        while True:
            try:
                command, payload = connection.recv()
            except EOFError:
                return
            try:
                if command == "ready":
                    result = True
                elif command == "reset":
                    reset_controller_stacks(payload or 1)
                    result = True
                elif command == "act":
                    obs, avail = payload
                    if len(controller_stacks) != len(obs):
                        reset_controller_stacks(len(obs))
                    actions = []
                    entropies = []
                    for batch_index in range(len(obs)):
                        controller.stack_obs = controller_stacks[batch_index]
                        action, entropy = controller.step(
                            torch.as_tensor(obs[batch_index:batch_index + 1], dtype=torch.float32),
                            torch.as_tensor(avail[batch_index:batch_index + 1], dtype=torch.float32),
                            None,
                        )
                        controller_stacks[batch_index] = controller.stack_obs
                        actions.append(action.argmax(-1).cpu().numpy())
                        entropies.append(entropy.cpu().numpy())
                    result = (np.stack(actions), np.stack(entropies))
                elif command == "train":
                    previous_train_count = learner.train_count
                    learner.step(payload)
                    # Upstream workers receive new parameters after an actual
                    # learner update, not after every collected episode.
                    if learner.train_count != previous_train_count:
                        controller.receive_params(learner.params())
                    result = {
                        "train_count": learner.train_count,
                        "accum_samples": learner.accum_samples,
                        "replay_size": len(learner.mamba_replay_buffer),
                    }
                elif command == "save":
                    os.makedirs(payload, exist_ok=True)
                    learner.save(os.path.join(payload, "reference_marie.pt"))
                    result = True
                elif command == "load":
                    checkpoint = torch.load(
                        os.path.join(payload, "reference_marie.pt"),
                        map_location=learner_config.DEVICE,
                    )
                    learner.tokenizer.load_state_dict(checkpoint["tokenizer"])
                    learner.model.load_state_dict(checkpoint["model"])
                    learner.actor.load_state_dict(checkpoint["actor"])
                    learner.critic.load_state_dict(checkpoint["critic"])
                    controller.receive_params(learner.params())
                    result = True
                elif command == "close":
                    connection.send(("ok", True))
                    return
                else:
                    raise ValueError(f"Unknown command: {command}")
                connection.send(("ok", result))
            except Exception:
                connection.send(("error", traceback.format_exc()))
    except Exception:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
