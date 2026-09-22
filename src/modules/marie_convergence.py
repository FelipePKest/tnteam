"""Conservative, checkpointable plateau detection; not a proof of optimality."""
import math

import torch as th


class ModelConvergence:
    def __init__(self, window=3, patience=2, parameter_tolerance=0.01,
                 validation_tolerance=0.02, min_events=20):
        if window < 2 or patience < 1 or min_events < 1:
            raise ValueError('Convergence window >= 2, patience/min_events >= 1 required')
        if not (0 < parameter_tolerance < 1 and 0 < validation_tolerance < 1):
            raise ValueError('Convergence tolerances must lie between zero and one')
        self.window = window
        self.patience = patience
        self.parameter_tolerance = parameter_tolerance
        self.validation_tolerance = validation_tolerance
        self.min_events = min_events
        self.previous = None
        self.history = []
        self.streak = 0
        self.frozen = False
        self.last_event = None

    @staticmethod
    def snapshot(model):
        groups = {'tokenizer': {}, 'codebook': {}, 'world_model': {}}
        for name, value in model.named_parameters():
            group = 'tokenizer' if name.startswith(('encoder.', 'decoder.')) else 'world_model'
            groups[group][name] = value.detach().float().cpu().clone()
        # EMA codes change without an optimizer step. Include the actual codebook,
        # not assignment-count accumulators whose scale depends on minibatches.
        for name, value in model.named_buffers():
            if name.endswith('_codebook.embed'):
                groups['codebook'][name] = value.detach().float().cpu().clone()
        return groups

    def observe(self, snapshot, metrics, event):
        if self.frozen or (self.last_event is not None and event <= self.last_event):
            return {}, self.frozen
        diagnostics = {}
        if self.previous is not None:
            changes = {}
            for group, tensors in snapshot.items():
                old = self.previous[group]
                numerator = sum(float((v - old[k]).double().square().sum()) for k, v in tensors.items())
                denominator = sum(float(v.double().square().sum()) for v in old.values())
                changes[group] = math.sqrt(numerator / max(denominator, 1e-24))
            self.history.append({'changes': changes, 'metrics': metrics, 'event': event,
                                 'start_event': self.last_event})
            self.history = self.history[-self.window:]
            finite = all(math.isfinite(v) for v in metrics.values()) and all(
                math.isfinite(v) for v in changes.values())
            if not finite:
                self.history = []
                self.streak = 0
            if len(self.history) == self.window:
                stable = self.history[-1]['event'] - self.history[0]['start_event'] >= self.min_events
                # Sum interval changes so movement out and back cannot cancel.
                for group in changes:
                    movement = sum(row['changes'][group] for row in self.history)
                    diagnostics[group + '_relative_movement'] = movement
                    stable &= movement <= self.parameter_tolerance
                for key in metrics:
                    values = [row['metrics'][key] for row in self.history]
                    spread = (max(values)-min(values))/max(abs(sum(values)/len(values)), 1e-6)
                    diagnostics[key + '_relative_spread'] = spread
                    stable &= math.isfinite(spread) and spread <= self.validation_tolerance
                self.streak = self.streak + 1 if stable else 0
                self.frozen = self.streak >= self.patience
        self.previous = snapshot
        self.last_event = event
        diagnostics['stable_windows'] = self.streak
        diagnostics['frozen'] = int(self.frozen)
        return diagnostics, self.frozen

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ('previous', 'history', 'streak', 'frozen', 'last_event')}

    def load_state_dict(self, state):
        for key in ('previous', 'history', 'streak', 'frozen', 'last_event'):
            setattr(self, key, state[key])
