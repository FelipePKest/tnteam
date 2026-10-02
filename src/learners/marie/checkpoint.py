"""MARIE checkpoint implementation."""

import os
import torch as th
from modules.world_models.marie.convergence import ValidationPlateau


class MARIECheckpoint:
    """MARIE checkpoint methods shared by the learner adapter."""

    def save_models(self, path):
        super().save_models(path)
        th.save(
            self.tokenizer_optimiser.state_dict(),
            os.path.join(path, "tokenizer_opt.th"),
        )
        th.save(
            {
                "marie_update_events": self.marie_update_events,
                "train_calls": self.train_calls,
                "model_threshold_streak": self.model_threshold_streak,
                "model_reduced_since": self.model_reduced_since,
                "model_last_eval_t": self.model_last_eval_t,
                "convergence": self.convergence.state_dict() if self.convergence else None,
                "convergence_mode": self.convergence_mode,
                "convergence_batches": self.convergence_batches,
                "validation_batch_size": int(getattr(self.args, "marie_validation_batch_size", 32)),
            },
            os.path.join(path, "marie_training_state.th"),
        )

    def load_models(self, path):
        if (
            not os.path.exists(os.path.join(path, "agent.th"))
            and os.path.exists(os.path.join(path, "reference_marie.pt"))
        ):
            self.mac.load_models(path)
            return
        super().load_models(path)
        tokenizer_path = os.path.join(path, "tokenizer_opt.th")
        if os.path.exists(tokenizer_path):
            self.tokenizer_optimiser.load_state_dict(
                th.load(tokenizer_path, map_location="cpu")
            )
        state_path = os.path.join(path, "marie_training_state.th")
        if os.path.exists(state_path):
            state = th.load(state_path, map_location="cpu")
            self.marie_update_events = state.get("marie_update_events", 0)
            self.train_calls = state.get("train_calls", 0)
            self.model_threshold_streak = state.get("model_threshold_streak", 0)
            self.model_reduced_since = state.get("model_reduced_since")
            self.model_last_eval_t = state.get("model_last_eval_t")

            if self.convergence is not None and state.get("convergence") is not None:
                same_mode = state.get("convergence_mode", "parameters") == self.convergence_mode
                if same_mode:
                    self.convergence.load_state_dict(state["convergence"])
                if isinstance(self.convergence, ValidationPlateau):
                    # Replay is rebuilt on restart; don't mix plateau histories.
                    # Frozen checkpoints retain their recovery baseline.
                    if (not self.convergence.frozen or state.get("validation_batch_size", 1)
                            != int(getattr(self.args, "marie_validation_batch_size", 32))):
                        self.convergence.reset()
                elif (not self.convergence.frozen and
                    state.get("validation_batch_size", 1) != int(
                        getattr(self.args, "marie_validation_batch_size", 32))):
                    # Batched sampling changes the stochastic realization.
                    # Start a new plateau window rather than mixing protocols.
                    self.convergence.previous = None
                    self.convergence.history = []
                    self.convergence.streak = 0
                    self.convergence.last_event = None
                self.convergence_batches = state.get("convergence_batches")
                if self.convergence.frozen:
                    self.policy_only = True
                    self.world_model.requires_grad_(False)
                    self.world_model.eval()

        self.set_optimizer_foreach(getattr(self.args, "marie_optimizer_foreach", False))
