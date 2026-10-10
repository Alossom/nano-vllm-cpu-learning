from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
	WAITING = auto()
	RUNNING = auto()
	FINISHED = auto()

class Sequence:
	block_size = 256
	_id_count = count()

	def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):

		if not token_ids:
			raise ValueError("A sequence must contain at least one token")

		if sampling_params is None:
			sampling_params = SamplingParams()

		if sampling_params.max_tokens <= 0:
			raise ValueError("max_tokens must be positive")

		self.seq_id = next(self._id_count)
		self.status = SequenceStatus.WAITING

		self.token_ids = list(token_ids)
		self.num_prompt_tokens = len(self.token_ids)

		self.num_cached_tokens = 0
		self.num_scheduled_tokens = 0
		self.is_prefill = True

		self.block_table: list[int] = []

		self.temperature = sampling_params.temperature
		self.max_tokens = sampling_params.max_tokens
		self.ignore_eos = sampling_params.ignore_eos

	def __len__(self) -> int:
		return len(self.token_ids)

	def __getitem__(self, index):
		return self.token_ids[index]

	@property
	def num_tokens(self) -> int:
		return len(self.token_ids)

	@property
	def last_token(self) -> int:
		return self.token_ids[-1]

	@property
	def num_completion_tokens(self) -> int:
		return self.num_tokens - self.num_prompt_tokens

	@property
	def prompt_token_ids(self) -> list[int]:
		return self.token_ids[:self.num_prompt_tokens]

	@property
	def completion_token_ids(self) -> list[int]:
		return self.token_ids[self.num_prompt_tokens:]

	@property
	def is_finished(self) -> bool:
		return self.status == SequenceStatus.FINISHED

	@property
	def num_blocks(self) -> int:
		return(self.num_tokens + self.block_size - 1) // self.block_size

	@property
	def num_cached_blocks(self) -> int:
		return self.num_cached_tokens // self.block_size

	@property
	def last_block_num_tokens(self) -> int:
		return(self.num_tokens - (self.num_blocks - 1) * self.block_size)

	def block(self, index: int) -> list[int]:
		if not 0 <= index < self.num_blocks:
			raise IndexError(f"Invalid logical block index: {index}")

		start = index * self.block_size
		end = min(start + self.block_size, self.num_tokens)
		return self.token_ids[start:end]

	def append_token(self, token_id: int):
		if self.is_finished:
			raise RuntimeError("Cannot append to a finished sequence")

		self. token_ids.append(token_id)
