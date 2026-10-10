# Nano-vLLM CPU Learning

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 学习与扩展的 CPU 推理参考实现，已运行真实 Qwen3-0.6B。用于学习推理调度、KVCache 管理及模型执行，不定位为生产级高性能引擎。

## 已实现

- CPU float32 Qwen3 推理、配置解析、单文件及分片 safetensors 加载、Embedding/LM Head 权重绑定。
- 底层多请求调度、chunked prefill、prefill-first 与交替调度策略。
- 分页 KVCache、完整块前缀共享、引用计数、内存不足时的抢占与恢复。
- Greedy、temperature、top-k、top-p 及请求独立随机数状态。
- `TextGenerator` 普通文本与聊天接口，以及命令行入口。该高层接口目前为单线程同步调用。

## 安装与运行

已验证环境：WSL Linux x86_64、Python 3.11.16、uv 0.12.10、PyTorch 2.14.0+cpu、Transformers 5.16.1。

克隆本仓库后，在仓库根目录执行：

```bash
uv python install 3.11.16
uv sync --locked --python 3.11.16
```

准备本地 Qwen3-0.6B 模型目录，包含配置、tokenizer 文件及 safetensors 权重。权重不随本仓库分发。将下方路径替换为实际模型目录：

```bash
uv run --locked python -m nanovllm \
  --model /absolute/path/to/Qwen3-0.6B \
  --max-tokens 96 --seed 2026 \
  "简要解释一下 KV cache 的作用。"
```

默认使用聊天模板与随机采样。`--raw` 用于普通文本续写；`--greedy` 使用确定性选择，不能同时传入 seed 或采样参数。完整选项见 `uv run --locked python -m nanovllm --help`。

CPU PyTorch 来源由项目的 uv 配置指定；普通 pip 安装不会自动采用此来源。

## 验证与限制

已通过手动实验验证：完整与分块计算一致性、前缀缓存复用、抢占恢复、请求级采样，以及真实模型生成。一个短样例的 greedy 输出与 Hugging Face 参考实现逐 token 一致；这不代表所有输入均已全面对齐。自动化回归测试与 CI 尚未整理。

默认上下文上限为 1024 token，包含输入和输出预算。Qwen3-0.6B 的四个 KV block 在当前 float32 配置下占用 224 MiB，另需模型权重及临时计算内存。

暂不支持量化、RoPE scaling、滑动窗口注意力或 HTTP 服务。Qwen3 可能先输出思考内容，较小输出预算不保证完整回答。尚无 CPU 性能基准，也不宣称性能优于原项目。

安装与生成检查见 VALIDATION.md。

## 来源与许可证

本项目基于 Xingkai Yu 的 nano-vllm，保留原 MIT 许可证及版权声明，详见 `LICENSE`。`*_learning.py` 是学习扩展模块；原 GPU 源码保留，但不属于当前 CPU 安装和验证范围。模型权重遵循各自的许可证。