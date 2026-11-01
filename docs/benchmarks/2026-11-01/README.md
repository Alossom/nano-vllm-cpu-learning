# CPU 性能测试：2026-11-01

[阅读完整性能测试报告](性能测试报告.md)

测试版本：`ecddefc5c4a1e36ffc85d65c5ce040d84340cc4d`。共 60 次正式测量；项目源码未为测试而修改。

- [汇总统计与正确性检查](summary.json)
- [CPU 项目测试数据](custom.json)
- [交错顺序的调度复测和整批 Prefill 数据](custom_confirm.json)
- [Transformers eager 基线](hf_eager.json)
- [Transformers SDPA 基线](hf_sdpa.json)
- [CPU 算子分析](profile.json)
- [测试环境](environment.json)
- [主测试复现代码](REPRODUCE.md)
- [确认测试复现代码](CONFIRM_REPRODUCE.md)
- [算子分析与汇总复现代码](ANALYSIS_REPRODUCE.md)

复现代码保留当次实测使用的路径。请先安装项目锁定的依赖，并自行准备 Qwen3-0.6B 权重；根据本机位置调整仓库、模型与输出目录，提前创建输出目录。各测试阶段应顺序执行，避免同时争用 CPU。分析汇总代码应在四组性能测试及算子分析全部完成后运行。本目录不包含模型权重。

原版对照使用仓库保留的原版调度器、Sequence 和 BlockManager 接入同一个 CPU 执行后端，不代表原版 CUDA 引擎的速度。HF 为另一组 CPU 参照。报告中的数据仅适用于明确列出的硬件、负载和测量口径。
