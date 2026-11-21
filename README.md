# CoreCPU-Lab

**313 · 前端软件系统组织思想 × 神经网络计算机制 × 学习规则**

这是一个可运行、可测量、可复现的 PyTorch 研究实验室，不是已经证明有效的新通用智能模型。

## 一套仓库，四种架构

| ID | 组织方式 | 计算机制 | 语言任务 |
|---|---|---|---|
| A | sequential (独立层) | conventional | 独立 Transformer 块 |
| B | corecpu (动态专家共享) | conventional | 共享因果注意力+稀疏专家 |
| C | sequential (独立层) | SEC | 独立状态演化 Cell，无自注意力 |
| D | corecpu (动态专家共享) | SEC | 稀疏共享 SEC，无自注意力 |

STRUCTURED：四模型共用同一外部键值读写先验、输入输出协议及同一生成器。LANGUAGE：四模型共享字节词表与下字节预测目标。数学上两类组织方式与两类计算机制被组合成 2×2 实验，但不是说各组能天然做到严格等参数/等计算。

**保留旧研究**：`legacy/v0.1/core_cpu.py`、`benchmark.py`、`architecture.md` 从既有 CoreCPU 架构文档中原样提取，旧版本代码不进行改写。新模块化 B 是与其思想保持一致的新实现，不是声称与旧模型逐权重兼容。

## 1. 安装

Python 3.10+（建议 3.11+）。CPU 可以直接运行；RTX 5060 Laptop 8GB 请安装与本机驱动兼容的官方 CUDA PyTorch wheel。

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m corecpu_lab selftest
python -m pytest
```

Windows PowerShell 安装命令也可以使用：`python -m pip install -e ".[dev]"`。

## 2. 跑 A/B/C/D 的最小训练

```bash
python -m corecpu_lab train --config configs/A_baseline.yaml --steps 2 --batch 8 --eval-batches 1 --out results/A_smoke
python -m corecpu_lab train --config configs/B_corecpu.yaml --steps 2 --batch 8 --eval-batches 1 --out results/B_smoke
python -m corecpu_lab train --config configs/C_new_compute.yaml --steps 2 --batch 8 --eval-batches 1 --out results/C_smoke
python -m corecpu_lab train --config configs/D_full_new.yaml --steps 2 --batch 8 --eval-batches 1 --out results/D_smoke
```

正式实验请移除 `--steps 2`，采用多随机种子对照，不能把烟雾测试分数作为真实性能。

```bash
python -m corecpu_lab benchmark --config configs/B_corecpu.yaml --seeds 313 101 202 303 --steps 700 --out results/structured_factorial
```

`benchmark` 会强制按 A、B、C、D 顺序执行，结果写入 `results.csv` 和各自运行目录；遇到已有结果目录会拒绝覆盖。

## 3. 训练和预测

```bash
python -m corecpu_lab eval --checkpoint results/B_smoke/checkpoint.pt --batches 5
python -m corecpu_lab predict --checkpoint results/B_smoke/checkpoint.pt --input example.json
python -m corecpu_lab train --config configs/D_full_new.yaml --out results/D_long
# 续训到总步骤 1400（必须保持其他训练配置一致）
python -m corecpu_lab train --config configs/D_full_new.yaml --steps 1400 --out results/D_long --resume results/D_long/checkpoint.pt
```

`predict` 不调用标签生成器。预测正确与否取决于训练，不是按 Python 固定规则直接算出答案。

## 4. 语言模型（先做演示验证）

```bash
python -m corecpu_lab train --config configs/A_baseline_language.yaml --steps 2 --out results/A_lm
python -m corecpu_lab train --config configs/B_corecpu_language.yaml --steps 2 --out results/B_lm
python -m corecpu_lab train --config configs/C_new_compute_language.yaml --steps 2 --out results/C_lm
python -m corecpu_lab train --config configs/D_full_new_language.yaml --steps 2 --out results/D_lm
python -m corecpu_lab generate --checkpoint results/D_lm/checkpoint.pt --prompt "CoreCPU " --tokens 40
```

以上配置默认 `demo: true`，使用重复合成文本，仅验证工程收敛/生成链路，不代表自然语言能力。要训练真实 UTF-8 语料：

```bash
python -m corecpu_lab train --config configs/D_full_new_language.yaml --text data/corpus.txt --out results/D_real_lm
```

语料按 90/10 连续切分，做进一步研究应引入文档级去重、独立测试集和版权合规的数据。

## 5. 学习规则

- `bp`：标准端到端梯度学习，A/B/C/D 的主基线。
- `local`：仅用于 STRUCTURED，逐步骤损失并在步间截断梯度；不是“完全没有反向传播”。
- `elig`：只用于 SEC+STRUCTURED，完全不执行 `backward()`；手工更新分类器及 SEC 的 `out` 投影。其他 SEC/路由/读取器参数固定，因此是学习机制的**局限性实验**，不是能端到端训练整个 CoreCPU 的 e-prop 实现。
- `ff-demo`：独立 Forward-Forward 局部层演示，不与 A/B/C/D 主实验混为一谈。

```bash
python -m corecpu_lab train --config configs/learning/D_local.yaml --steps 2 --out results/D_local
python -m corecpu_lab train --config configs/learning/D_elig.yaml --steps 2 --out results/D_elig
python -m corecpu_lab ff-demo --steps 30
```

## 6. 评测的硬边界

结构化分为 `iid`、`pair`（留出运算邻接组合）、`long`（6 步）和 `mod`（只测模数 5/7）。训练模数为 4/6/8，训练链长 1–3。所有模型使用同一合成分布和匹配的固定评估随机数。

主要报告末步、逐步、全步骤正确率，读取地址正确率、参数数、训练时间、评估吞吐、GPU 峰值显存。对比必须补做等参数、等 FLOPs、等时间和多尺度实验；仅固定 `hidden_dim` 不等价于计算预算匹配。

**禁止**通过把标准答案放入 `forward`、修改测试集、只报告成功种子或夸大玩具任务结果来评价架构。

## 7. 文档和运行结构

`docs/ARCHITECTURE.md`：核心架构；`docs/EXPERIMENTS.md`：基线和科研协议；`docs/COMPUTE.md`：SEC 机制与实现；`docs/LEARNING.md`：局部与非全局学习；`docs/ROADMAP.md`：下一步；`legacy/v0.1/`：原始代码快照。

仓库中没有模型权重，也没有声称已经在你的 Windows RTX 5060 上通过实测。每次训练产生自己的 checkpoint、配置、日志、版本环境和四拆分评估 JSON。

## 8. 速度、显存与二因子交互分析

```bash
python -m corecpu_lab profile --checkpoint results/D_smoke/checkpoint.pt --lengths 3 6 12 --batch 4 --repeats 20
python -m corecpu_lab analyze --csv results/structured_factorial/results.csv --out results/structured_factorial/analysis.json
```

`profile` 是不含数据生成的前向耗时，不测训练速度、不反映模型质量；CUDA 读数为本进程当前模型的峰值。`analyze` 只汇总配对种子的描述统计和交互项，不将单次交互项当作统计显著性证据。

合入之前 GitHub 仓库的步骤见 `docs/MIGRATION.md`。连续集成配置见 `.github/workflows/ci.yml`。
