# Nano vLLM Baseline
本文记录上游 nano-vllm（提交 bb823b3）的初始环境与 benchmark 方法分析。Python 3.14.4、CUDA 硬件要求等描述不代表当前 CPU 学习版。本文未提供本机实际 GPU 性能测量结果；当前安装与使用方式以 README 为准。

仓库版本：
bb823b3

Python：
当前为 3.14.4，不满足项目要求。
项目要求 Python >=3.10, <3.13。
建议使用 Python 3.11。

模型：
Qwen3-0.6B

模型路径：
~/huggingface/Qwen3-0.6B/

硬件要求：
需要 NVIDIA CUDA GPU。
代码证据：torch.set_default_device("cuda")。
当前机器只有集成显卡，无法完成真实 CUDA 推理 benchmark。

执行配置：
tensor_parallel_size = 1
max_model_len = 4096
max_num_seqs = 512
max_num_batched_tokens = 16384
kvcache_block_size = 256
enforce_eager = False

Benchmark 配置：
请求数 = 256
输入长度 = 100–1024 token，随机生成
输出长度 = 100–1024 token，随机生成
输入形式 = 随机 token ID，不经过 tokenizer
warmup = 1 次
正式计时 = llm.generate() 调用前后
吞吐 = 计划输出 token 数 / 总耗时

已测指标：
总输出 token 数、总耗时、输出吞吐

未测指标：
TTFT、TPOT、p50/p99、显存峰值、GPU 利用率、
prefill/decode 分项耗时
