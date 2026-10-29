# LLaMA-Factory（精简源码）

本目录是项目内置的LLaMA-Factory训练运行时，仅用于AgentDoG模型的SFT。上游版本和固定提交见[UPSTREAM.md](UPSTREAM.md)，许可证见[LICENSE](LICENSE)。

保留内容：

- `src/llamafactory/`：完整训练运行时；
- `pyproject.toml`与`MANIFEST.in`：安装配置；
- `data/dataset_info.json`：本项目数据集注册表；
- `examples/deepspeed/ds_z2_config.json`：全参数SFT使用的ZeRO-2配置；
- `requirements/deepspeed.txt`：DeepSpeed额外依赖。

GPU机器安装：

```bash
cd LLaMA-Factory
pip install -e .
pip install -r requirements/deepspeed.txt
```

训练数据注册和AgentDoG训练YAML由本项目后续阶段生成，不在此目录保存上游demo、文档站、测试、Docker或CI文件。
