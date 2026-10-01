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


class ValidationPlateau:
    """Reversible freeze based on rolling component errors, never weights.

    Fresh errors use their own pre-freeze baseline, not the fixed distribution.
    """
    def __init__(self, window=3, patience=6, min_delta=.02, min_events=20,
                 deterioration=.15, recovery_patience=3):
        if window < 1 or patience < 1 or min_events < 1 or recovery_patience < 1:
            raise ValueError('Validation window, patience and min_events must be positive')
        if not 0 < min_delta < 1 or not 0 < deterioration < 1:
            raise ValueError('Validation thresholds must be in (0, 1)')
        self.settings = dict(window=window, patience=patience, min_delta=min_delta,
                             min_events=min_events, deterioration=deterioration,
                             recovery_patience=recovery_patience)
        self.reset()

    def reset(self):
        self.history = []
        self.fresh_history = []
        self.best = {}
        self.bad = {}
        self.baseline = {}
        self.recovery_streak = 0
        self.frozen = False
        self.last_event = None
        self.start_event = None

    @staticmethod
    def _mean(rows):
        return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}

    def observe(self, metrics, fresh, event):
        if self.last_event is not None and event <= self.last_event:
            return {}, self.frozen
        self.last_event = event
        valid = (bool(metrics) and metrics.keys() == fresh.keys()
                 and all(math.isfinite(v) and v >= 0 for v in [*metrics.values(), *fresh.values()]))
        if self.history and self.history[0].keys() != metrics.keys():
            valid = False
        if not valid:
            was_frozen = self.frozen
            self.reset()
            self.last_event = event
            return {'invalid_validation': 1, 'thawed': int(was_frozen), 'frozen': 0}, False
        if self.start_event is None:
            self.start_event = event
        self.history.append(dict(metrics))
        self.fresh_history.append(dict(fresh))
        self.history = self.history[-self.settings['window']:]
        self.fresh_history = self.fresh_history[-self.settings['window']:]
        diagnostics = {'frozen': int(self.frozen), 'thawed': 0}
        if len(self.history) < self.settings['window']:
            return diagnostics, self.frozen
        averaged, fresh_mean = self._mean(self.history), self._mean(self.fresh_history)
        if self.frozen:
            degraded = any(fresh_mean[k] > v + self.settings['deterioration'] * max(v, 1e-6)
                           for k, v in self.baseline.items())
            self.recovery_streak = self.recovery_streak + 1 if degraded else 0
            diagnostics['recovery_streak'] = self.recovery_streak
            for key, baseline in self.baseline.items():
                diagnostics[key + '_fresh_relative_increase'] = (fresh_mean[key] - baseline) / max(baseline, 1e-6)
            if self.recovery_streak >= self.settings['recovery_patience']:
                self.reset()
                self.last_event = event
                diagnostics.update(thawed=1, frozen=0)
            return diagnostics, self.frozen
        for key, value in averaged.items():
            best = self.best.get(key)
            if best is None or best - value > self.settings['min_delta'] * max(best, 1e-6):
                self.best[key], self.bad[key] = value, 0
            else:
                self.bad[key] += 1
            diagnostics[key + '_no_improvement_checks'] = self.bad[key]
        degraded = any(averaged[k] > v + self.settings['deterioration'] * max(v, 1e-6)
                       for k, v in self.best.items())
        enough_updates = event - self.start_event >= self.settings['min_events']
        self.frozen = (enough_updates and not degraded
                       and min(self.bad.values()) >= self.settings['patience'])
        if self.frozen:
            self.baseline = fresh_mean
        diagnostics.update(no_improvement_checks=min(self.bad.values()),
                           deterioration_blocked=int(degraded), frozen=int(self.frozen))
        return diagnostics, self.frozen

    def state_dict(self):
        import copy
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        import copy
        if state.get('settings') != self.settings:
            self.reset()
            return False
        self.__dict__.update(copy.deepcopy(state))
        return True
