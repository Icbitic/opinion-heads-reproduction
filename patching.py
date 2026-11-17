"""Final-position query-head output replacement, with a fixed prefix cache."""
from contextlib import contextmanager

import torch
from transformers.cache_utils import Cache, DynamicLayer


class ReplayLayer(DynamicLayer):
    """Immutable prefix plus an overwrite-only final-token slot for each row."""

    def __init__(self, original, batch_size):
        super().__init__()
        key, value = original.keys, original.values
        self.prefix_length = original.get_seq_length()
        self.stored_length = key.shape[-2]
        self.is_sliding = original.is_sliding
        self.sliding_window = getattr(original, "sliding_window", None)
        shape = (batch_size, key.shape[1], self.stored_length + 1, key.shape[-1])
        self.keys = torch.empty(shape, dtype=key.dtype, device=key.device)
        self.values = torch.empty(shape, dtype=value.dtype, device=value.device)
        self.keys[..., :-1, :].copy_(key.expand(batch_size, -1, -1, -1))
        self.values[..., :-1, :].copy_(value.expand(batch_size, -1, -1, -1))
        self.is_initialized = True
        self.dtype, self.device = key.dtype, key.device

    def get_seq_length(self):
        return self.prefix_length

    def get_mask_sizes(self, cache_position):
        return self.stored_length + cache_position.shape[0], self.prefix_length - self.stored_length

    def get_max_cache_shape(self):
        return self.sliding_window if self.is_sliding else -1

    def update(self, key_states, value_states, cache_kwargs=None):
        if key_states.shape[-2] != 1 or key_states.shape[0] != self.keys.shape[0]:
            raise ValueError("Replay accepts one final token at the fixed batch size")
        if cache_kwargs and cache_kwargs.get("cache_position") is not None:
            if cache_kwargs["cache_position"].numel() != 1:
                raise ValueError("Replay requires one final position")
        self.keys[..., -1:, :].copy_(key_states)
        self.values[..., -1:, :].copy_(value_states)
        return self.keys, self.values


def replay_cache(prefix, batch_size):
    return Cache(layers=[ReplayLayer(layer, batch_size) for layer in prefix.layers])


class HeadPatcher:
    def __init__(self, model, device, heads):
        self.model, self.device = model, device
        self.head_dim = model.model.layers[0].self_attn.head_dim
        self.capture = None
        self.requests = {}
        self.donor = None
        self.handles = []
        for index in sorted({layer for layer, _ in heads}):
            projection = model.model.layers[index].self_attn.o_proj
            if projection.in_features != model.config.num_attention_heads * self.head_dim:
                raise ValueError("Unexpected concatenated query-head shape")
            self.handles.append(projection.register_forward_pre_hook(self._hook(index)))

    def _hook(self, layer):
        def hook(module, args):
            value = args[0]
            if self.capture is not None:
                self.capture[layer] = value[:, -1:, :].detach().clone()
            requests = self.requests.get(layer, [])
            if not requests:
                return None
            patched = value.clone()
            for row, head in requests:
                selection = slice(head * self.head_dim, (head + 1) * self.head_dim)
                patched[row, -1, selection] = self.donor[layer][0, 0, selection]
            return (patched, *args[1:])
        return hook

    @contextmanager
    def intervene(self, head_sets, donor):
        if self.requests:
            raise RuntimeError("Nested interventions")
        self.donor = donor
        for row, heads in enumerate(head_sets):
            for layer, head in heads:
                self.requests.setdefault(layer, []).append((row, head))
        try:
            yield
        finally:
            self.requests, self.donor = {}, None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    @torch.inference_mode()
    def prefill(self, ids):
        inputs = torch.tensor([ids[:-1]], device=self.device)
        return self.model(input_ids=inputs, use_cache=True, logits_to_keep=1,
                          return_dict=True).past_key_values

    @torch.inference_mode()
    def last(self, ids, cache, batch_size, capture=False):
        length = len(ids) - 1
        if cache.get_seq_length() != length:
            raise ValueError("Prefix cache length mismatch")
        inputs = torch.full((batch_size, 1), ids[-1], dtype=torch.long, device=self.device)
        positions = torch.full((batch_size, 1), length, dtype=torch.long, device=self.device)
        mask = torch.ones((batch_size, length + 1), dtype=torch.long, device=self.device)
        self.capture = {} if capture else None
        try:
            output = self.model(input_ids=inputs, attention_mask=mask, position_ids=positions,
                                cache_position=positions[0], past_key_values=cache,
                                use_cache=True, logits_to_keep=1, return_dict=True)
            logits = output.logits[:, -1, :].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite logits")
            return logits, self.capture
        finally:
            self.capture = None

    @torch.inference_mode()
    def full(self, ids):
        inputs = torch.tensor([ids], device=self.device)
        return self.model(input_ids=inputs, use_cache=False, logits_to_keep=1,
                          return_dict=True).logits[:, -1, :].float()
