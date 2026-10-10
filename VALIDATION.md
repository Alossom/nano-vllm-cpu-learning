# CPU 版本验证

在仓库根目录执行以下命令。需要已准备本地 Qwen3-0.6B 模型，将 `MODEL_DIR` 替换为实际路径。

## 安装与生成检查

```bash
uv lock --check
uv sync --locked --python 3.11.16
git diff --check

MODEL_DIR=/home/x1o/ai-infra/models/Qwen3-0.6B \
uv run --locked python -I - <<'PY'
import os
import sys
import torch
from nanovllm import TextGenerator, SamplingOptions

assert torch.version.cuda is None
assert "nanovllm.engine.model_runner" not in sys.modules
g = TextGenerator(os.environ["MODEL_DIR"])

text = "KV cache 的主要作用是"
a = g.generate(text, max_tokens=8, seed=2026, ignore_eos=True)
b = g.generate(text, max_tokens=8, seed=2026, ignore_eos=True)
assert a.options.mode == b.options.mode == "sample"
assert a.seed == b.seed == 2026
assert a.generated_tokens == b.generated_tokens == 8
assert a.generated_token_ids == b.generated_token_ids
assert a.request_id != b.request_id
print("CPU 正式入口、连续请求与固定 seed：通过")

messages = [{"role": "user", "content": "简要解释一下 KV cache 的作用。"}]
original = [dict(message) for message in messages]
c = g.generate_chat(
    messages, max_tokens=4,
    options=SamplingOptions(mode="greedy"), ignore_eos=True,
)
assert c.seed is None and c.options.mode == "greedy"
assert c.generated_tokens == 4
assert messages == original
assert len({a.request_id, b.request_id, c.request_id}) == 3
print("Chat、greedy 与输入不变性：通过")
print("随机生成文本:", a.text)
print("Chat 生成文本:", c.text)
PY
```

## 验证范围

以上命令是安装与生成的冒烟检查，不替代完整单元测试。

开发期间还手动验证了分块 prefill、前缀共享、抢占恢复、请求级随机采样和模型数值一致性。这些专项实验尚未全部整理为仓库内可重复执行的检查。

目前没有自动化 CI、全面 Hugging Face 数值对照或 CPU 性能基准。短输出可能只有 Qwen3 的思考开头，不保证包含完整回答。