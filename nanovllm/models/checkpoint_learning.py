import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import torch

from nanovllm.models.model_learning import TinyCausalLM
from nanovllm.models.weight_loader_learning import (LoadReport, load_qwen3_weights)

class SafetensorsCheckpoint(Mapping):
	def __init__(self, directory: str | Path):
		self.directory = Path(directory).resolve()
		self._stack = None
		self._files = {}
		self._weight_map = {}

	def __enter__(self):
		from safetensors import safe_open

		if self._stack is not None:
			raise RuntimeError("Checkpoint is already open")

		if not self.directory.is_dir():
			raise ValueError(f"Model directory does not exist: {self.directory}")

		self._stack = ExitStack()

		try:
			index_path = (self.directory / "model.safetensors.index.json")
			single_path = self.directory / "model.safetensors"

			expected_map = None

			if index_path.is_file():
				with index_path.open("r", encoding="utf-8") as file:
					index = json.load(file)

				if not isinstance(index, dict):
					raise ValueError("Checkpoint index must be an object")

				expected_map = index.get("weight_map")

				if(not isinstance(expected_map, dict) or not expected_map):
					raise ValueError("Checkpoint index must contain a nonempty weight_map")

				if any(type(name) is not str or not name or type(filename) is not str or not filename for name, filename in expected_map.items()):
					raise ValueError("weight_map names and filenames must be strings")

				filenames = sorted(set(expected_map.values()))

			elif single_path.is_file():
				filenames = ["model.safetensors"]

			else:
				raise ValueError("No model.safetensors or " "model.safetensors.index.json found")

			actual_map = {}

			for filename in filenames:
				relative_path = Path(filename)

				if relative_path.is_absolute():
					raise ValueError("Shard filename must be relative to model directory")

				path = (self.directory / relative_path).resolve()

				if not path.is_relative_to(self.directory):
					raise ValueError(f"Shard lies outside model directory: {filename}")

				if path.suffix != ".safetensors":
					raise ValueError(f"Unexpected shard file type: {filename}")

				if not path.is_file():
					raise ValueError(f"Checkpoint shard does not exist: {filename}")

				handle = self._stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
				self._files[filename] = handle

				for name in handle.keys():
					if name in actual_map:
						raise ValueError(f"Duplicate weight across shards: {name}")

					actual_map[name] = filename

			if expected_map is not None and actual_map != expected_map:
				missing = sorted(set(expected_map) - set(actual_map))
				unexpected = sorted(set(actual_map) - set(expected_map))
				wrong_shard = sorted(name for name in set(expected_map) & set(actual_map) if expected_map[name] != actual_map[name])

				raise ValueError("Checkpoint index does not match shard contents: " f"missing={missing}, " f"unexpected={unexpected}, " f"wrong_shard={wrong_shard}")

			self._weight_map = actual_map
			return self

		except BaseException:
			self._close()
			raise

	def __exit__(self, exc_type, exc_value, traceback):
		self._close()
		return False

	def _close(self):
		if self._stack is not None:
			self._stack.close()

		self._stack = None
		self._files.clear()
		self._weight_map.clear()

	def __getitem__(self, name: str) -> torch.Tensor:
		if self._stack is None:
			raise RuntimeError("Checkpoint must be used inside a with block")

		filename = self._weight_map[name]
		return self._files[filename].get_tensor(name)

	def __iter__(self):
		return iter(self._weight_map)

	def __len__(self):
		return len(self._weight_map)

def load_qwen3_directory(model: TinyCausalLM, directory: str | Path) -> LoadReport:
	with SafetensorsCheckpoint(directory) as checkpoint:
		return load_qwen3_weights(model, checkpoint)
