"""Checkpoint compatibility for runs produced by the former MARIE service."""

import torch as th


def _load_complete(module, translated, label):
    current = module.state_dict()
    compatible = {}
    for name, value in translated.items():
        if name in current and current[name].shape == value.shape:
            compatible[name] = value
    required = {name for name, _ in module.named_parameters()}
    missing = sorted(required - compatible.keys())
    if missing:
        raise RuntimeError(
            f"Cannot convert {label}; missing compatible tensors: {missing}"
        )
    module.load_state_dict(compatible, strict=False)


def _translate_world(tokenizer, model):
    translated = {}
    for name, value in tokenizer.items():
        if name.startswith("encoder."):
            translated["encoder.encoder." + name[len("encoder."):]] = value
        elif name.startswith("decoder."):
            translated[name] = value
        elif name.startswith("codebook."):
            translated[
                "encoder.quantizer." + name[len("codebook."):]
            ] = value

    direct = {
        "embedder.embedding_tables.0.weight": "action_token_embedding.weight",
        "embedder.embedding_tables.1.weight": "world_token_embedding.weight",
        "head_observations.head_module.0.weight": "next_token_head.0.weight",
        "head_observations.head_module.0.bias": "next_token_head.0.bias",
        "head_observations.head_module.2.weight": "next_token_head.2.weight",
        "head_observations.head_module.2.bias": "next_token_head.2.bias",
    }
    for source_prefix, target_prefix in (
        ("head_rewards.head_module.", "reward."),
        ("head_ends.head_module.", "continuation."),
        ("heads_avail_actions.head_module.dists.", "next_availability."),
    ):
        for name, value in model.items():
            if name.startswith(source_prefix):
                translated[target_prefix + name[len(source_prefix):]] = value
    for source, target in direct.items():
        if source in model:
            translated[target] = model[source]
    if "pos_emb.weight" in model:
        translated["position"] = model["pos_emb.weight"].unsqueeze(0)

    transformer_parts = {
        "ln1.": "attention_norm.",
        "ln2.": "feed_forward.0.",
        "attn.query.": "query.",
        "attn.key.": "key.",
        "attn.value.": "value.",
        "attn.proj.": "attention_output.",
        "mlp.0.": "feed_forward.1.",
        "mlp.2.": "feed_forward.4.",
    }
    for name, value in model.items():
        if name.startswith("transformer.blocks.") and ".attn.mask" not in name:
            remainder = name[len("transformer.blocks."):]
            layer, remainder = remainder.split(".", 1)
            for source, target in transformer_parts.items():
                if remainder.startswith(source):
                    translated[
                        f"sequence_model.layers.{layer}." + target
                        + remainder[len(source):]
                    ] = value
                    break
        elif name.startswith("transformer.ln_f."):
            translated[
                "sequence_model.norm." + name[len("transformer.ln_f."):]
            ] = value

    perceiver_direct = {
        "perattn.latents": "aggregator.query",
        "perattn.cross_attn_blocks.0.fn.": "aggregator.cross_attention.",
        "perattn.cross_attn_blocks.0.norm_context.":
            "aggregator.cross_context_norm.",
        "perattn.cross_attn_blocks.0.norm.": "aggregator.cross_query_norm.",
        "perattn.cross_attn_blocks.1.fn.net.": "aggregator.cross_ff.net.",
        "perattn.cross_attn_blocks.1.norm.": "aggregator.cross_ff_norm.",
    }
    for name, value in model.items():
        if name == "perattn.latents":
            translated[perceiver_direct[name]] = value
            continue
        for source, target in perceiver_direct.items():
            if source.endswith(".") and name.startswith(source):
                translated[target + name[len(source):]] = value
                break
        if not name.startswith("perattn.layers."):
            continue
        remainder = name[len("perattn.layers."):]
        layer, branch, remainder = remainder.split(".", 2)
        branch_mapping = {
            ("0", "fn."): f"aggregator.latent_attention.{layer}.",
            ("0", "norm."): f"aggregator.latent_norms.{layer}.",
            ("1", "fn.net."): f"aggregator.latent_ff.{layer}.net.",
            ("1", "norm."): f"aggregator.latent_ff_norms.{layer}.",
        }
        for (expected_branch, source), target in branch_mapping.items():
            if branch == expected_branch and remainder.startswith(source):
                translated[target + remainder[len(source):]] = value
                break
    return translated


def load_reference_marie_checkpoint(policy, checkpoint_path):
    """Load a legacy ``reference_marie.pt`` without importing external code."""
    checkpoint = th.load(checkpoint_path, map_location="cpu")
    expected = {"tokenizer", "model", "actor", "critic"}
    if not expected.issubset(checkpoint):
        raise RuntimeError(
            f"{checkpoint_path} is not a MARIE reference checkpoint"
        )

    _load_complete(
        policy.world_model,
        _translate_world(checkpoint["tokenizer"], checkpoint["model"]),
        "MARIE tokenizer/world model",
    )
    actor = {
        "net." + name[len("feedforward_model."):]: value
        for name, value in checkpoint["actor"].items()
        if name.startswith("feedforward_model.")
    }
    _load_complete(policy.actors[0], actor, "MARIE actor")

    critic = {}
    for name, value in checkpoint["critic"].items():
        if name.startswith("feedforward_model."):
            critic["value." + name[len("feedforward_model."):]] = value
        elif name.startswith("_attention_stack.encoder."):
            critic["attention." + name[len("_attention_stack.encoder."):]] = value
        elif name.startswith("embed."):
            critic[name] = value
    _load_complete(policy.critics[0], critic, "MARIE critic")
    policy.ema_critics.load_state_dict(policy.critics.state_dict())
