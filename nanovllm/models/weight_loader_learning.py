from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from nanovllm.models.model_learning import TinyCausalLM

@dataclass(frozen=True)
class LoadReport:
	num_tensors: int
	num_parameters: int

def qwen3_weight_map(num_layers: int, *, tie_word_embeddings: bool = False) -> dict[str, str]:
	if type(num_layers) is not int or num_layers <= 0:
		raise ValueError("num_layers must be a positive integer")

	mapping = {"model.embed_tokens.weight": "embed_tokens.weight"}

	layer_suffixes = ("input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")

	for layer_index in range(num_layers):
		for suffix in layer_suffixes:
			source = f"model.layers.{layer_index}.{suffix}"
			target = f"layers.{layer_index}.{suffix}"
			mapping[source] = target

	mapping["model.norm.weight"] = "norm.weight"

	if not tie_word_embeddings:
		mapping["lm_head.weight"] = "lm_head.weight"

	return mapping

@torch.no_grad()
def load_qwen3_weights(model:TinyCausalLM, checkpoint: Mapping[str, torch.Tensor]) -> LoadReport:
	if not isinstance(checkpoint, Mapping):
		raise ValueError("checkpoint must be a mapping")

	if any(type(name) is not str for name in checkpoint):
		raise ValueError("Checkpoint names must be strings")

	mapping = qwen3_weight_map(model.num_layers, tie_word_embeddings=model.tie_word_embeddings)
	targets = dict(model.named_parameters())

	mapped_targets = set(mapping.values())
	actual_targets = set(targets)

	if mapped_targets != actual_targets:
		unmapped = sorted(actual_targets - mapped_targets)
		nonexistent = sorted(mapped_targets - actual_targets)

		raise ValueError("Model parameters do not match loader mapping: " f"unmapped={unmapped}, nonexistent={nonexistent}")

	expected_sources = set(mapping)
	actual_sources = set(checkpoint)

	optional_tied_lm_head = (model.tie_word_embeddings and "lm_head.weight" in checkpoint)

	actual_sources = set(checkpoint)

	if optional_tied_lm_head:
		tied_lm_head = checkpoint["lm_head.weight"]
		embed_weight = checkpoint["model.embed_tokens.weight"]

		if tied_lm_head.shape != embed_weight.shape:
			raise ValueError("Tied lm_head.weight shape does not match " "model.embed_tokens.weight")

		if not torch.equal(tied_lm_head, embed_weight):
			raise ValueError("Tied lm_head.weight differs from embedding weight")

		actual_sources.remove("lm_head.weight")

	missing = sorted(expected_sources - actual_sources)
	unexpected = sorted(actual_sources - expected_sources)

	if missing or unexpected:
		raise ValueError(f"Checkpoint keys mismatch: " f"missing={missing}, unexpected={unexpected}")

	supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}

	for source_name, target_name in mapping.items():
		source = checkpoint[source_name]
		target = targets[target_name]

		if not isinstance(source, torch.Tensor):
			raise ValueError(f"{source_name}: weight must be a tensor")

		if source.layout != torch.strided:
			raise ValueError(f"{source_name}: only dense tensors are supported")

		if source.device.type != "cpu":
			raise ValueError(f"{source_name}: source weight must be on CPU")

		if source.dtype not in supported_dtypes:
			raise ValueError(f"{source_name}: unsupported dtype {source.dtype}")

		if(target.device.type != "cpu" or target.dtype != torch.float32):
			raise ValueError(f"{target_name}: target must be CPU float32")

		if source.shape != target.shape:
			raise ValueError(f"{source_name}: shape mismatch, " f"expected={tuple(target.shape)}, " f"got={tuple(source.shape)}")

		if not torch.isfinite(source).all().item():
			raise ValueError(f"{source_name}: weight contains NaN or infinity")

	for source_name, target_name in mapping.items():
		targets[target_name].copy_(checkpoint[source_name])

	model.eval()

	return LoadReport(num_tensors=len(mapping), num_parameters=sum(parameter.numel() for parameter in targets.values()))

def load_qwen3_safetensors(model: TinyCausalLM, path: str | Path) -> LoadReport:
	path = Path(path)

	if not path.is_file():
		raise ValueError(f"Safetensors file does not exist: {path}")

	try:
		from safetensors.torch import load_file
	except ImportError as exc:
		raise RuntimeError("safetensors is required; install it with " "uv pip install --python .venv/bin/python safetensors") from exc

	checkpoint = load_file(str(path), device="cpu")

	return load_qwen3_weights(model, checkpoint)
