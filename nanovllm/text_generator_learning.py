import json
import secrets
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm.cpu_config_learning import CpuConfig
from nanovllm.engine.sampling_engine_learning import SamplingEngine
from nanovllm.engine.sampling_runner_learning import SamplingModelRunner
from nanovllm.layers.sampler_learning import SamplingOptions
from nanovllm.models.checkpoint_learning import load_qwen3_directory
from nanovllm.models.model_config_learning import Qwen3ModelConfig
from nanovllm.models.model_learning import TinyCausalLM

@dataclass(frozen=True)
class GenerationResult:
	request_id: int
	prompt_token_ids: tuple[int, ...]
	generated_token_ids: tuple[int, ...]
	text: str
	seed: int | None
	options: SamplingOptions

	@property
	def prompt_tokens(self) -> int:
		return len(self.prompt_token_ids)

	@property
	def generated_tokens(self) -> int:
		return len(self.generated_token_ids)

class TextGenerator:
	def __init__(self, model_dir: str | Path, *, max_model_len: int = 1024, num_kvcache_blocks: int = 4, max_num_batched_tokens: int = 128, policy: str = "alternating") -> None:
		if policy not in ("prefill_first", "alternating"):
			raise ValueError("policy must be 'prefill_first' or 'alternating'")

		self.model_dir = Path(model_dir).expanduser().resolve()

		if not self.model_dir.is_dir():
			raise FileNotFoundError(f"Model directory does not exist: {self.model_dir}")

		config_path = self.model_dir / "config.json"

		with config_path.open(encoding="utf-8") as f:
			raw_config = json.load(f)

		cpu_config = CpuConfig(max_num_seqs=1, max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len, num_kvcache_blocks=num_kvcache_blocks, kvcache_block_size=256)
		cpu_config.validate_model_limits(max_position_embeddings=raw_config["max_position_embeddings"])

		model_config = Qwen3ModelConfig.from_json(config_path)

		self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)

		eos = self._tokenizer.eos_token_id
		vocab_size = raw_config["vocab_size"]

		if type(eos) is not int or not 0 <= eos < vocab_size:
			raise ValueError("Tokenizer must provide a valid EOS token id")

		configured_eos = raw_config.get("eos_token_id")

		if type(configured_eos) is int:
			model_eos_ids = (configured_eos)
		elif isinstance(configured_eos, list):
			if(not configured_eos or any(type(value) is not int for value in configured_eos)):
				raise ValueError("Invalid model EOS configuration")

			model_eos_ids = tuple(dict.fromkeys(configured_eos))
		else:
			raise ValueError("Model must define EOS token ids")

		if model_eos_ids != eos:
			raise ValueError("This interface currently requires one model EOS id " "matching the tokenizer")

		self.config = CpuConfig(max_num_seqs=cpu_config.max_num_seqs, max_num_batched_tokens=cpu_config.max_num_batched_tokens, max_model_len=cpu_config.max_model_len, num_kvcache_blocks=cpu_config.num_kvcache_blocks, kvcache_block_size=cpu_config.kvcache_block_size, eos=eos)

		self._vocab_size = vocab_size
		self.kv_cache_bytes = self.config.estimate_kv_cache_bytes(num_layers=raw_config["num_hidden_layers"], num_kv_heads=raw_config["num_key_value_heads"], head_dim=raw_config["head_dim"])

		self._model = TinyCausalLM(**model_config.model_kwargs(num_blocks=self.config.num_kvcache_blocks, block_size=self.config.kvcache_block_size))

		self.load_report = load_qwen3_directory(self._model, self.model_dir)
		self._model.eval()

		runner = SamplingModelRunner(self._model)
		self._engine = SamplingEngine(self.config, runner, policy=policy)

		self._busy = False
		self._failed = False

	def generate(self, text: str, *, max_tokens: int = 64, options: SamplingOptions | None = None, seed: int | None = None, ignore_eos: bool = False) -> GenerationResult:
		if not isinstance(text, str):
			raise TypeError("text must be a string")

		if not text.strip():
			raise ValueError("text must not be empty or whitespace")

		token_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]

		return self._generate_token_ids(token_ids, max_tokens=max_tokens, options=options, seed=seed, ignore_eos=ignore_eos)

	def generate_chat(self, messages: list[dict[str, str]], *, max_tokens: int = 64, options: SamplingOptions | None = None, seed: int | None = None, ignore_eos: bool = False) -> GenerationResult:
		checked_messages = self._validate_messages(messages)

		prompt = self._tokenizer.apply_chat_template(checked_messages, tokenize=False, add_generation_prompt=True)

		token_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]

		return self._generate_token_ids(token_ids, max_tokens=max_tokens, options=options, seed=seed, ignore_eos=ignore_eos)

	@staticmethod
	def _validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
		if not isinstance(messages, list) or not messages:
			raise ValueError("messages must be a nonempty list")

		checked = []

		for index, message in enumerate(messages):
			if not isinstance(message, dict):
				raise TypeError(f"messages[{index}] must be a dict")

			if set(message) != {"role", "content"}:
				raise ValueError(f"messages[{index}] must contain only role and content")

			role = message["role"]
			content = message["content"]

			if not isinstance(role, str) or role not in ("system", "user", "assistant"):
				raise ValueError(f"Unsupported role in messages[{index}]")

			if not isinstance(content, str) or not content.strip():
				raise ValueError(f"messages[{index}].content must be a nonempty string")

			checked.append({"role": role, "content": content})

		return checked

	def _generate_token_ids(self, token_ids: list[int], *, max_tokens: int, options: SamplingOptions | None, seed: int | None, ignore_eos: bool) -> GenerationResult:
		if self._failed:
			raise RuntimeError("A previous execution failed. " "Create a new TextGenerator before generating again.")

		if self._busy:
			raise RuntimeError("TextGenerator is already processing a request")

		if type(ignore_eos) is not bool:
			raise TypeError("ignore_eos must be a bool")

		if options is None:
			options = SamplingOptions(mode="sample", temperature=0.8, top_k=20, top_p=0.9)
		elif not isinstance(options, SamplingOptions):
			raise TypeError("options must be SamplingOptions")

		if options.mode == "greedy":
			if seed is not None:
				raise ValueError("Greedy requests do not use a seed")
		else:
			if seed is None:
				seed = secrets.randbits(63)
			elif type(seed) is not int or not 0 <= seed < 2**63:
				raise ValueError("seed must be an integer in [0, 2**63)")

		self.config.validate_request(prompt_tokens=len(token_ids), max_tokens=max_tokens)

		if any(type(token_id) is not int or not 0 <= token_id < self._vocab_size for token_id in token_ids):
			raise ValueError("Tokenizer returned an invalid token id")

		self._busy = True

		try:
			request_id = self._engine.add_request(list(token_ids), max_tokens=max_tokens, ignore_eos=ignore_eos, options=options, seed=seed)

			event_tokens = []

			with torch.inference_mode():
				while not self._engine.is_finished():
					for event in self._engine.step():
						if event.seq_id != request_id:
							raise RuntimeError("Unexpected request id")

						if event.token_id is not None:
							event_tokens.append(event.token_id)

			generated = tuple(self._engine.take_result(request_id))

			if tuple(event_tokens) != generated:
				raise RuntimeError("Streamed tokens do not match completed result")

			if len(generated) > max_tokens:
				raise RuntimeError("Generated output exceeds max_tokens")

			text = self._tokenizer.decode(generated, skip_special_tokens=True)

			return GenerationResult(request_id=request_id, prompt_token_ids=tuple(token_ids), generated_token_ids=generated, text=text, seed=seed, options=options)

		except Exception:
			self._failed = True
			raise

		finally:
			self._busy = False
