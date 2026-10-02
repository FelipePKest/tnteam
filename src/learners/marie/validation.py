"""MARIE validation implementation."""

import torch as th
from components.episode_buffer import EpisodeBatch
from components.marie_context import compact_controlled, controlled_group
from modules.world_models.marie.convergence import ModelConvergence, ValidationPlateau
from modules.world_models.marie.validation import collect_validation


class MARIEValidation:
    """MARIE validation methods shared by the learner adapter."""

    def observe_model_convergence(self, batches, t_env):
        if getattr(self.args, "marie_policy_only", False):
            return  # An explicit policy-only request is not a reversible freeze.
        reversible = isinstance(self.convergence, ValidationPlateau)
        if self.convergence is None or (self.policy_only and not (reversible and self.convergence.frozen)):
            return
        if self.convergence_batches is None:
            # Keep the first held-out episodes fixed for comparable measurements.
            # Store plain tensor dictionaries so they can be checkpointed portably.
            self.convergence_batches = []
            for batch in batches:
                self.convergence_batches.append({
                    "scheme": {k:v for k,v in batch.scheme.items() if k != "filled"},
                    "groups": batch.groups, "batch_size": batch.batch_size,
                    "max_seq_length": batch.max_seq_length,
                    "transitions": {k:v.detach().cpu().clone() for k,v in batch.data.transition_data.items()},
                    "episodes": {k:v.detach().cpu().clone() for k,v in batch.data.episode_data.items()},
                })
        fixed = []
        for saved in self.convergence_batches:
            batch = EpisodeBatch(saved["scheme"], saved["groups"], saved["batch_size"], saved["max_seq_length"])
            batch.data.transition_data = saved["transitions"]
            batch.data.episode_data = saved["episodes"]
            fixed.append(batch)
        stats = self.validate_world_model(fixed, t_env, prefix="marie_fixed_validation")
        keys = ["tokenizer_reconstruction_mae"] + [
            "h%d/%s" % (h, metric) for h in (1, 5, 15)
            for metric in ("observation_mae", "reward_mae", "return_mae")
        ]
        if not reversible and any("marie_fixed_validation/"+key not in stats for key in keys):
            return  # insufficient held-out horizon; never infer convergence
        metrics = {key: stats.get("marie_fixed_validation/"+key, float('nan')) for key in keys}
        if reversible:
            cached = getattr(self, "_fresh_validation", None)
            fresh_stats = (cached[1] if cached is not None and cached[0] == t_env
                           else self.validate_world_model(batches, t_env))
            fresh = {key: fresh_stats.get("marie_validation/"+key, float('nan')) for key in keys}
            diagnostics, frozen = self.convergence.observe(metrics, fresh, self.marie_update_events)
        else:
            diagnostics, frozen = self.convergence.observe(
                ModelConvergence.snapshot(self.world_model), metrics, self.marie_update_events)
        for key, value in diagnostics.items():
            self.logger.log_stat("marie_convergence/"+key, value, t_env)
        if reversible and self.policy_only and not frozen:
            self.policy_only = False
            self.world_model.requires_grad_(True)
            self.world_model.train()
            self.logger.console_logger.info(
                "MARIE validation recovery at t_env=%s: resuming tokenizer and world-model updates", t_env)
        if frozen and not self.policy_only:
            self.policy_only = True
            self.world_model.requires_grad_(False)
            self.world_model.eval()
            self.logger.console_logger.info(
                "MARIE convergence plateau at t_env=%s: freezing tokenizer and world model; "
                "continuing agent-only training. This is a heuristic plateau, not proof of optimality.", t_env)

    @th.no_grad()
    def validate_world_model(self, batches, t_env, prefix="marie_validation"):
        """Open-loop predictions on fresh evaluation episodes, never replayed.

        Follow recorded actions with the same cached stochastic dynamics used
        by policy imagination. Report each horizon separately; no universal
        sufficiency threshold is assumed. Preserve training RNG and modes.
        """
        modes = [(module, module.training) for module in self.world_model.modules()]
        records = {h: [] for h in (1, 5, 15)}
        reconstruction_errors = []
        device = next(self.world_model.parameters()).device
        devices = [device.index or 0] if device.type == "cuda" else []
        try:
            self.world_model.eval()
            with th.random.fork_rng(devices=devices):
                th.manual_seed(1729)
                validation_size = int(getattr(self.args, "marie_validation_batch_size", 32))
                if getattr(self.args, "marie_controlled_replay_only", False):
                    groups = {}
                    for batch in batches:
                        # Team composition can differ between evaluation batches.
                        for row in range(batch.batch_size):
                            compact = compact_controlled(batch[row:row+1])
                            groups.setdefault(compact["obs"].shape[2], []).append(compact)
                    for count, grouped in sorted(groups.items()):
                        with controlled_group(self, count):
                            measured, errors = collect_validation(
                                self.world_model, grouped, count, validation_size)
                        for horizon in records:
                            records[horizon].extend(measured[horizon])
                        reconstruction_errors.extend(errors)
                else:
                    records, reconstruction_errors = collect_validation(
                        self.world_model, batches, self.n_agents, validation_size)

        finally:
            for module, training in modes:
                module.training = training
        result = {}
        base_prefix = prefix
        if reconstruction_errors:
            result[base_prefix+"/tokenizer_reconstruction_mae"] = sum(reconstruction_errors)/len(reconstruction_errors)
        for horizon, rows in records.items():
            if not rows:
                continue
            prefix = base_prefix + "/h%d/" % horizon
            for key in rows[0]:
                if not key.startswith("terminal_"):
                    result[prefix+key] = sum(row[key] for row in rows)/len(rows)
            tp = sum(row["terminal_tp"] for row in rows)
            predicted = sum(row["terminal_predicted"] for row in rows)
            actual = sum(row["terminal_actual"] for row in rows)
            result[prefix+"terminal_actual"] = actual
            result[prefix+"terminal_predicted"] = predicted
            if predicted:
                result[prefix+"terminal_precision"] = tp/predicted
            if actual:
                result[prefix+"terminal_recall"] = tp/actual
            result[prefix+"windows"] = len(rows)
        for key, value in result.items():
            self.logger.log_stat(key, value, t_env)
        if base_prefix == "marie_validation":
            self._fresh_validation = (t_env, result)
        return result
