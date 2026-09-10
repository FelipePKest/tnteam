import torch as th
import numpy as np
from types import SimpleNamespace as SN


class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        # Ensure device is a torch.device object for compatibility with older PyTorch versions
        self.device = th.device(device) if isinstance(device, str) else device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)
            # Ensure vshape is a flat tuple of ints (handle nested tuples)
            vshape = tuple(int(v) for v in vshape)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (int(groups[group]),) + vshape
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((int(batch_size),) + shape, dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((int(batch_size), int(max_seq_length)) + shape, dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        device = th.device(device) if isinstance(device, str) else device
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            if type(v) == list or type(v) == np.ndarray:
                v = th.tensor(np.array(v), dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        self.marie_sample_visits = {
            "tokenizer": th.zeros(
                buffer_size, max_seq_length, dtype=th.long, device="cpu"
            ),
            "model": th.zeros(
                buffer_size, max_seq_length, dtype=th.long, device="cpu"
            ),
        }

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            inserted = slice(
                self.buffer_index, self.buffer_index + ep_batch.batch_size
            )
            for visits in self.marie_sample_visits.values():
                visits[inserted].zero_()
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def marie_transition_count(self):
        return sum(
            max(0, int(self.data.transition_data["filled"][episode].sum()) - 1)
            for episode in range(self.episodes_in_buffer)
        )

    def _marie_candidates(self, sequence_length):
        candidates = []
        for episode in range(self.episodes_in_buffer):
            filled = int(
                self.data.transition_data["filled"][episode].sum().item()
            )
            # A sequence with L transitions requires L+1 observations.
            last_start = filled - sequence_length - 1
            for start in range(max(0, last_start + 1)):
                candidates.append((episode, start))
        return candidates

    def can_sample_marie(self, batch_size, sequence_length):
        return len(self._marie_candidates(sequence_length)) >= batch_size

    def sample_marie(
        self, batch_size, sequence_length, mode="model", temperature="inf"
    ):
        """Visit-balanced transition sampling used by canonical MARIE.

        Sampling is without replacement. Visit counts are maintained
        independently for tokenizer and world-model draws and reset whenever
        an episode slot is overwritten.
        """
        if mode == "policy":
            return self._sample_marie_policy(
                batch_size, sequence_length + 1
            )
        if mode not in self.marie_sample_visits:
            raise ValueError("Unknown MARIE replay mode: {}".format(mode))
        candidates = self._marie_candidates(sequence_length)
        if len(candidates) < batch_size:
            raise ValueError(
                "Not enough MARIE transitions: {} available, {} requested".format(
                    len(candidates), batch_size
                )
            )
        visits = None
        if mode == "policy":
            # The upstream episode dataset draws actor contexts uniformly and
            # does not share tokenizer/world-model visit counters.
            probabilities = None
        else:
            visits = np.asarray([
                int(self.marie_sample_visits[mode][episode, start])
                for episode, start in candidates
            ], dtype=np.float64)
        if visits is not None and (
            temperature == "inf" or temperature == float("inf")
        ):
            if visits.sum() == 0:
                probabilities = np.full(len(visits), 1.0 / len(visits))
            else:
                probabilities = 1.0 - visits / visits.sum()
                probabilities /= probabilities.sum()
        elif visits is not None:
            logits = -visits / float(temperature)
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
        selected = np.random.choice(
            len(candidates), batch_size, replace=False, p=probabilities
        )
        result_scheme = {
            key: value for key, value in self.scheme.items() if key != "filled"
        }
        result = EpisodeBatch(
            result_scheme, self.groups, batch_size, sequence_length + 1,
            preprocess=None, device=self.device,
        )
        for output, candidate_index in enumerate(selected):
            episode, start = candidates[candidate_index]
            stop = start + sequence_length + 1
            for key, value in self.data.transition_data.items():
                result.data.transition_data[key][output] = value[
                    episode, start:stop
                ]
            for key, value in self.data.episode_data.items():
                result.data.episode_data[key][output] = value[episode]
            if mode != "policy":
                self.marie_sample_visits[mode][episode, start] += 1
        return result

    def _sample_marie_policy(self, batch_size, observation_length):
        """Match MultiAgentEpisodesDataset endpoint sampling and left padding."""
        if self.episodes_in_buffer == 0:
            raise ValueError("Cannot sample MARIE policy contexts from empty replay")
        result_scheme = {
            key: value for key, value in self.scheme.items() if key != "filled"
        }
        result = EpisodeBatch(
            result_scheme, self.groups, batch_size, observation_length,
            preprocess=None, device=self.device,
        )
        # Upstream uses random.choices: episodes are uniform with replacement.
        episodes = np.random.choice(
            self.episodes_in_buffer, batch_size, replace=True
        )
        for output, episode in enumerate(episodes):
            filled = int(
                self.data.transition_data["filled"][episode].sum().item()
            )
            # EPyMARL also marks the observation after the terminal action as
            # filled. The reference episodic MARIE buffer contains transition
            # records only, so that final observation is not an endpoint.
            transition_count = max(1, filled - 1)
            stop = np.random.randint(1, transition_count + 1)
            start = stop - observation_length
            source_start = max(0, start)
            destination_start = source_start - start
            copied = stop - source_start
            destination = slice(destination_start, destination_start + copied)
            source = slice(source_start, stop)
            for key, value in self.data.transition_data.items():
                result.data.transition_data[key][output, destination] = value[
                    episode, source
                ]
            for key, value in self.data.episode_data.items():
                result.data.episode_data[key][output] = value[episode]
        return result

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size, recency_decay=None):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            probabilities = None
            if recency_decay is not None:
                indices = np.arange(self.episodes_in_buffer)
                if self.episodes_in_buffer < self.buffer_size:
                    ages = self.episodes_in_buffer - 1 - indices
                else:
                    # buffer_index always points at the next (oldest) slot.
                    ages = (self.buffer_index - 1 - indices) % self.buffer_size
                probabilities = np.power(float(recency_decay), ages.astype(np.float64))
                probabilities /= probabilities.sum()
            ep_ids = np.random.choice(
                self.episodes_in_buffer, batch_size, replace=False, p=probabilities
            )
            return self[ep_ids]

    def sample_sequences(self, batch_size, sequence_length, recency_decay=None):
        """Sample non-overlapping transition chunks with transition-age weighting."""
        candidates = []
        weights = []
        for ep_id in range(self.episodes_in_buffer):
            filled = int(self.data.transition_data["filled"][ep_id].sum().item())
            transitions = max(0, filled - 1)
            if transitions == 0:
                continue
            if self.episodes_in_buffer < self.buffer_size:
                episode_age = self.episodes_in_buffer - 1 - ep_id
            else:
                episode_age = (self.buffer_index - 1 - ep_id) % self.buffer_size
            for start in range(0, transitions, sequence_length):
                stop = min(start + sequence_length, transitions)
                candidates.append((ep_id, start, stop))
                age = episode_age * self.max_seq_length + transitions - stop
                weights.append(
                    1.0 if recency_decay is None else float(recency_decay) ** age
                )

        if len(candidates) < batch_size:
            raise ValueError(
                "Not enough replay sequences: {} available, {} requested".format(
                    len(candidates), batch_size
                )
            )
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        selected = np.random.choice(
            len(candidates), batch_size, replace=False, p=probabilities
        )
        result_scheme = {
            key: value for key, value in self.scheme.items() if key != "filled"
        }
        result = EpisodeBatch(
            result_scheme,
            self.groups,
            batch_size,
            sequence_length + 1,
            preprocess=None,
            device=self.device,
        )
        for out_id, candidate_id in enumerate(selected):
            ep_id, start, stop = candidates[candidate_id]
            count = stop - start + 1
            for key, value in self.data.transition_data.items():
                result.data.transition_data[key][out_id, :count] = value[
                    ep_id, start:stop + 1
                ]
            if stop == transitions:
                result.data.transition_data["terminated"][
                    out_id, stop - start - 1
                ] = 1
            for key, value in self.data.episode_data.items():
                result.data.episode_data[key][out_id] = value[ep_id]
        return result

    def can_sample_sequences(self, batch_size, sequence_length):
        available = 0
        for ep_id in range(self.episodes_in_buffer):
            filled = int(self.data.transition_data["filled"][ep_id].sum().item())
            transitions = max(0, filled - 1)
            available += int(np.ceil(transitions / float(sequence_length)))
        return available >= batch_size
    
    def clear(self):
        '''Clear the replay buffer'''
        self.buffer_index = 0
        self.episodes_in_buffer = 0
        for k, v in self.data.transition_data.items():
            v.zero_()
        for k, v in self.data.episode_data.items():
            v.zero_()
        for visits in self.marie_sample_visits.values():
            visits.zero_()
            
    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())
