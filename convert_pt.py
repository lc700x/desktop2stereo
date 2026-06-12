#!/usr/bin/env python3
"""Convert Depth Anything metric v1 checkpoints from LiheYoung/Depth-Anything.

The Space checkpoint is a training checkpoint with keys like
``core.core.pretrained.*`` plus extra metric-bin head tensors.  The Hugging Face
``DepthAnythingForDepthEstimation`` model only uses the Depth Anything core, so
this script maps that core to the HF key names, splits DINOv2 qkv tensors, and
drops only the known extra metric head tensors.

By default the output is a plain ``DepthAnythingForDepthEstimation`` model with
relative depth values. That is the simpler HF path and it avoids the metric
head contract mismatch from the Space checkpoint.
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    DepthAnythingConfig,
    DepthAnythingForDepthEstimation,
    Dinov2Config,
    DPTImageProcessor,
    ZoeDepthConfig,
    ZoeDepthForDepthEstimation,
)


METRIC_EXTRAS = (
    "conv2.",
    "seed_bin_regressor.",
    "seed_projector.",
    "projectors.",
    "attractors.",
    "conditional_log_binomial.",
)


ZOE_EXTRA_BUFFERS = (
    "conditional_log_binomial.log_binomial_transform.k_idx",
    "conditional_log_binomial.log_binomial_transform.K_minus_1",
)


def get_dpt_config(model_name: str, force_relative: bool = False) -> DepthAnythingConfig:
    """Create a DepthAnythingConfig matching the HF Depth Anything layouts."""
    model_name = model_name.lower()

    if "small" in model_name:
        out_indices = [3, 6, 9, 12] if "v2" in model_name else [9, 10, 11, 12]
        backbone_name = "facebook/dinov2-small"
        fusion_hidden_size = 64
        neck_hidden_sizes = [48, 96, 192, 384]
    elif "base" in model_name:
        out_indices = [3, 6, 9, 12] if "v2" in model_name else [9, 10, 11, 12]
        backbone_name = "facebook/dinov2-base"
        fusion_hidden_size = 128
        neck_hidden_sizes = [96, 192, 384, 768]
    elif "large" in model_name:
        out_indices = [5, 12, 18, 24] if "v2" in model_name else [21, 22, 23, 24]
        backbone_name = "facebook/dinov2-large"
        fusion_hidden_size = 256
        neck_hidden_sizes = [256, 512, 1024, 1024]
    else:
        raise NotImplementedError(f"Model not supported: {model_name}")

    backbone_config = Dinov2Config.from_pretrained(
        backbone_name,
        out_indices=out_indices,
        apply_layernorm=True,
        reshape_hidden_states=False,
    )

    is_metric = "metric" in model_name and not force_relative
    config = DepthAnythingConfig(
        reassemble_hidden_size=backbone_config.hidden_size,
        patch_size=backbone_config.patch_size,
        backbone_config=backbone_config,
        fusion_hidden_size=fusion_hidden_size,
        neck_hidden_sizes=neck_hidden_sizes,
        depth_estimation_type="metric" if is_metric else "relative",
        max_depth=(20 if "indoor" in model_name else 80) if is_metric else None,
    )
    config.architectures = ["DepthAnythingForDepthEstimation"]
    return config


def get_zoedepth_config(model_name: str) -> ZoeDepthConfig:
    """Create the metric ZoeDepth config matching the Space checkpoint."""
    model_name = model_name.lower()
    if "large" not in model_name:
        raise NotImplementedError("Only the large metric-v1 Space checkpoint is supported for ZoeDepth conversion")

    max_depth = 20.0 if "indoor" in model_name else 80.0
    backbone_config = Dinov2Config.from_pretrained(
        "facebook/dinov2-large",
        out_indices=[21, 22, 23, 24],
        apply_layernorm=True,
        reshape_hidden_states=False,
    )

    config = ZoeDepthConfig(
        backbone_config=backbone_config,
        readout_type="ignore",
        neck_hidden_sizes=[256, 512, 1024, 1024],
        fusion_hidden_size=256,
        num_relative_features=32,
        bottleneck_features=256,
        bin_embedding_dim=128,
        num_attractors=[16, 8, 4, 1],
        bin_centers_type="softplus",
        bin_configurations=[{"name": "outdoor" if "outdoor" in model_name else "indoor", "n_bins": 64, "min_depth": 0.001, "max_depth": max_depth}],
    )
    config.architectures = ["ZoeDepthForDepthEstimation"]
    return config


def create_rename_keys(config: DepthAnythingConfig | ZoeDepthConfig, head_prefix: str = "head") -> list[tuple[str, str]]:
    """Return original Depth Anything key names and their HF names."""
    rename_keys = [
        ("pretrained.cls_token", "backbone.embeddings.cls_token"),
        ("pretrained.mask_token", "backbone.embeddings.mask_token"),
        ("pretrained.pos_embed", "backbone.embeddings.position_embeddings"),
        ("pretrained.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"),
        ("pretrained.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"),
    ]

    for i in range(config.backbone_config.num_hidden_layers):
        rename_keys.extend(
            [
                (f"pretrained.blocks.{i}.ls1.gamma", f"backbone.encoder.layer.{i}.layer_scale1.lambda1"),
                (f"pretrained.blocks.{i}.ls2.gamma", f"backbone.encoder.layer.{i}.layer_scale2.lambda1"),
                (f"pretrained.blocks.{i}.norm1.weight", f"backbone.encoder.layer.{i}.norm1.weight"),
                (f"pretrained.blocks.{i}.norm1.bias", f"backbone.encoder.layer.{i}.norm1.bias"),
                (f"pretrained.blocks.{i}.norm2.weight", f"backbone.encoder.layer.{i}.norm2.weight"),
                (f"pretrained.blocks.{i}.norm2.bias", f"backbone.encoder.layer.{i}.norm2.bias"),
                (f"pretrained.blocks.{i}.mlp.fc1.weight", f"backbone.encoder.layer.{i}.mlp.fc1.weight"),
                (f"pretrained.blocks.{i}.mlp.fc1.bias", f"backbone.encoder.layer.{i}.mlp.fc1.bias"),
                (f"pretrained.blocks.{i}.mlp.fc2.weight", f"backbone.encoder.layer.{i}.mlp.fc2.weight"),
                (f"pretrained.blocks.{i}.mlp.fc2.bias", f"backbone.encoder.layer.{i}.mlp.fc2.bias"),
                (
                    f"pretrained.blocks.{i}.attn.proj.weight",
                    f"backbone.encoder.layer.{i}.attention.output.dense.weight",
                ),
                (
                    f"pretrained.blocks.{i}.attn.proj.bias",
                    f"backbone.encoder.layer.{i}.attention.output.dense.bias",
                ),
            ]
        )

    rename_keys.extend(
        [
            ("pretrained.norm.weight", "backbone.layernorm.weight"),
            ("pretrained.norm.bias", "backbone.layernorm.bias"),
        ]
    )

    for i in range(4):
        rename_keys.extend(
            [
                (f"depth_head.projects.{i}.weight", f"neck.reassemble_stage.layers.{i}.projection.weight"),
                (f"depth_head.projects.{i}.bias", f"neck.reassemble_stage.layers.{i}.projection.bias"),
            ]
        )
        if i != 2:
            rename_keys.extend(
                [
                    (f"depth_head.resize_layers.{i}.weight", f"neck.reassemble_stage.layers.{i}.resize.weight"),
                    (f"depth_head.resize_layers.{i}.bias", f"neck.reassemble_stage.layers.{i}.resize.bias"),
                ]
            )

    refinenet_mapping = {1: 3, 2: 2, 3: 1, 4: 0}
    for i in range(1, 5):
        j = refinenet_mapping[i]
        rename_keys.extend(
            [
                (f"depth_head.scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"),
                (f"depth_head.scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.weight",
                    f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit1.conv1.bias",
                    f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.weight",
                    f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit1.conv2.bias",
                    f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.weight",
                    f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit2.conv1.bias",
                    f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.weight",
                    f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight",
                ),
                (
                    f"depth_head.scratch.refinenet{i}.resConfUnit2.conv2.bias",
                    f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias",
                ),
            ]
        )

    for i in range(4):
        rename_keys.append((f"depth_head.scratch.layer{i + 1}_rn.weight", f"neck.convs.{i}.weight"))

    rename_keys.extend(
        [
            ("depth_head.scratch.output_conv1.weight", f"{head_prefix}.conv1.weight"),
            ("depth_head.scratch.output_conv1.bias", f"{head_prefix}.conv1.bias"),
            ("depth_head.scratch.output_conv2.0.weight", f"{head_prefix}.conv2.weight"),
            ("depth_head.scratch.output_conv2.0.bias", f"{head_prefix}.conv2.bias"),
            ("depth_head.scratch.output_conv2.2.weight", f"{head_prefix}.conv3.weight"),
            ("depth_head.scratch.output_conv2.2.bias", f"{head_prefix}.conv3.bias"),
        ]
    )

    return rename_keys


def create_metric_head_rename_keys() -> list[tuple[str, str]]:
    rename_keys = [
        ("conv2.weight", "metric_head.conv2.weight"),
        ("conv2.bias", "metric_head.conv2.bias"),
        ("seed_bin_regressor._net.0.weight", "metric_head.seed_bin_regressor.conv1.weight"),
        ("seed_bin_regressor._net.0.bias", "metric_head.seed_bin_regressor.conv1.bias"),
        ("seed_bin_regressor._net.2.weight", "metric_head.seed_bin_regressor.conv2.weight"),
        ("seed_bin_regressor._net.2.bias", "metric_head.seed_bin_regressor.conv2.bias"),
        ("seed_projector._net.0.weight", "metric_head.seed_projector.conv1.weight"),
        ("seed_projector._net.0.bias", "metric_head.seed_projector.conv1.bias"),
        ("seed_projector._net.2.weight", "metric_head.seed_projector.conv2.weight"),
        ("seed_projector._net.2.bias", "metric_head.seed_projector.conv2.bias"),
        ("conditional_log_binomial.mlp.0.weight", "metric_head.conditional_log_binomial.mlp.0.weight"),
        ("conditional_log_binomial.mlp.0.bias", "metric_head.conditional_log_binomial.mlp.0.bias"),
        ("conditional_log_binomial.mlp.2.weight", "metric_head.conditional_log_binomial.mlp.2.weight"),
        ("conditional_log_binomial.mlp.2.bias", "metric_head.conditional_log_binomial.mlp.2.bias"),
    ]

    for i in range(4):
        rename_keys.extend(
            [
                (f"projectors.{i}._net.0.weight", f"metric_head.projectors.{i}.conv1.weight"),
                (f"projectors.{i}._net.0.bias", f"metric_head.projectors.{i}.conv1.bias"),
                (f"projectors.{i}._net.2.weight", f"metric_head.projectors.{i}.conv2.weight"),
                (f"projectors.{i}._net.2.bias", f"metric_head.projectors.{i}.conv2.bias"),
                (f"attractors.{i}._net.0.weight", f"metric_head.attractors.{i}.conv1.weight"),
                (f"attractors.{i}._net.0.bias", f"metric_head.attractors.{i}.conv1.bias"),
                (f"attractors.{i}._net.2.weight", f"metric_head.attractors.{i}.conv2.weight"),
                (f"attractors.{i}._net.2.bias", f"metric_head.attractors.{i}.conv2.bias"),
            ]
        )

    return rename_keys


def resolve_input_checkpoint(input_pt: str) -> Path:
    path = Path(input_pt)
    if path.is_dir():
        outdoor = path / "depth_anything_metric_depth_outdoor.pt"
        if outdoor.exists():
            return outdoor
        checkpoints = sorted(path.glob("*.pt"))
        if len(checkpoints) == 1:
            return checkpoints[0]
        names = ", ".join(p.name for p in checkpoints) or "none"
        raise FileNotFoundError(f"Could not choose a checkpoint in {path}. Found: {names}")
    return path


def unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dict, got {type(checkpoint)!r}")

    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError("Checkpoint state dict keys must be strings")

    return dict(checkpoint)


def strip_known_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "core.core.", "core."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def split_qkv(state_dict: dict[str, torch.Tensor], config: DepthAnythingConfig) -> None:
    hidden_size = config.backbone_config.hidden_size
    for i in range(config.backbone_config.num_hidden_layers):
        weight_key = f"pretrained.blocks.{i}.attn.qkv.weight"
        bias_key = f"pretrained.blocks.{i}.attn.qkv.bias"
        if weight_key not in state_dict or bias_key not in state_dict:
            raise KeyError(f"Missing qkv tensors for transformer block {i}")

        qkv_weight = state_dict.pop(weight_key)
        qkv_bias = state_dict.pop(bias_key)
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.weight"] = qkv_weight[:hidden_size, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.query.bias"] = qkv_bias[:hidden_size]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.weight"] = qkv_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.key.bias"] = qkv_bias[hidden_size : hidden_size * 2]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.weight"] = qkv_weight[-hidden_size:, :]
        state_dict[f"backbone.encoder.layer.{i}.attention.attention.value.bias"] = qkv_bias[-hidden_size:]


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: DepthAnythingConfig | ZoeDepthConfig,
    architecture: str,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    normalized = {}
    for key, value in state_dict.items():
        key = strip_known_prefix(key)
        if key.startswith("head."):
            key = "depth_head." + key[len("head.") :]
        normalized[key] = value
    state_dict = normalized

    dropped = []
    if architecture == "depth_anything":
        dropped = sorted(key for key in state_dict if key.startswith(METRIC_EXTRAS))
        for key in dropped:
            state_dict.pop(key)
        head_prefix = "head"
    elif architecture == "zoedepth":
        dropped = [key for key in ZOE_EXTRA_BUFFERS if key in state_dict]
        for key in dropped:
            state_dict.pop(key)
        head_prefix = "relative_head"
    else:
        raise ValueError(f"Unknown target architecture: {architecture}")

    split_qkv(state_dict, config)

    rename_keys = create_rename_keys(config, head_prefix=head_prefix)
    if architecture == "zoedepth":
        rename_keys.extend(create_metric_head_rename_keys())

    for src, dst in rename_keys:
        if src not in state_dict:
            raise KeyError(f"Missing expected tensor: {src}")
        state_dict[dst] = state_dict.pop(src)

    return state_dict, dropped


def default_processor() -> DPTImageProcessor:
    return DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        ensure_multiple_of=14,
        keep_aspect_ratio=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )


def convert_pt_to_hf(
    input_pt: str,
    output_dir: str,
    model_name: str,
    save_processor: bool = True,
    force_relative: bool = True,
    depth_anything_compatible: bool = True,
) -> None:
    checkpoint_path = resolve_input_checkpoint(input_pt)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = unwrap_state_dict(checkpoint)

    if depth_anything_compatible:
        config = get_dpt_config(model_name, force_relative=force_relative)
        model = DepthAnythingForDepthEstimation(config)
        architecture = "depth_anything"
    else:
        config = get_zoedepth_config(model_name)
        model = ZoeDepthForDepthEstimation(config)
        architecture = "zoedepth"

    converted, dropped = convert_state_dict(state_dict, config, architecture=architecture)

    expected_keys = set(model.state_dict())
    unexpected_after_filter = sorted(set(converted) - expected_keys)
    if unexpected_after_filter:
        raise RuntimeError(f"Unexpected converted keys: {unexpected_after_filter}")

    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Conversion failed. Missing keys: {missing}; unexpected keys: {unexpected}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    if save_processor:
        default_processor().save_pretrained(output_path)

    print(f"Dropped {len(dropped)} unsupported tensors: {dropped}")
    if isinstance(config, ZoeDepthConfig):
        print(f"Saved as ZoeDepth metric depth; max_depth={config.bin_configurations[0]['max_depth']}")
    else:
        print(f"Saved as {config.depth_estimation_type} depth; max_depth={config.max_depth}")
    print(f"Saved Hugging Face model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pt",
        default=r"E:\Download\depth-anything-metric-v1",
        help="Path to the .pt checkpoint, or a folder containing depth_anything_metric_depth_outdoor.pt",
    )
    parser.add_argument(
        "--output_dir",
        default=r"E:\Download\depth-anything-metric-v1\hf",
        help="Directory to save the Hugging Face model",
    )
    parser.add_argument(
        "--model_name",
        default="depth-anything-metric-outdoor-large",
        help="Model variant, e.g. depth-anything-metric-outdoor-large",
    )
    parser.add_argument("--no_processor", action="store_true", help="Do not save a DPTImageProcessor")
    parser.add_argument(
        "--relative_head",
        action="store_true",
        help="With --depth_anything_compatible, save a relative config for diagnostics.",
    )
    args = parser.parse_args()

    convert_pt_to_hf(
        args.input_pt,
        args.output_dir,
        args.model_name,
        save_processor=not args.no_processor,
        force_relative=True,
        depth_anything_compatible=True,
    )
