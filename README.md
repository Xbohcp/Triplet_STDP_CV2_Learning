# 🧠 Triplet_STDP_CV2_Learning

![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-4B8BBE)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-7A5CFA)

一个用于验证生物合理学习规则的 PyTorch 实验项目。项目基于 MNIST，比较标准反向传播和一种正值激活、截断误差传播、活动依赖权重更新规则之间的权重变化方向差异。

> 核心问题：提出的生物合理误差传播与权重更新规则，是否能在深度学习任务中产生接近 BP 的权重更新方向？

核心输出是一张趋势图：

```text
横轴: epoch
纵轴: BP 权重更新向量 与 生物规则权重更新向量 的平均夹角
```

夹角越小，说明两种学习规则在当前网络状态下给出的权重变化方向越接近。

---

## 📚 目录

- [项目亮点](#-项目亮点)
- [实验设置](#-实验设置)
- [快速开始](#-快速开始)
- [正式实验](#-正式实验)
- [推荐实验组合](#-推荐实验组合)
- [脚本参数](#-脚本参数)
- [手动运行](#-手动运行)
- [项目结构](#-项目结构)
- [结果解读](#-结果解读)
- [文档导航](#-文档导航)

---

## ✨ 项目亮点

- 🧪 **同批次对比**：每个 batch 都在同一个网络状态上计算 BP 更新和生物规则更新。
- 📐 **方向指标清晰**：用权重变化向量夹角衡量两种学习规则的接近程度。
- 🧬 **正值激活约束**：所有层使用 `softplus(x) + eps`，满足激活为正的建模要求。
- 🧰 **一键脚本**：自动创建环境、安装依赖、运行测试、执行实验并输出图表。
- 📝 **完整文档**：包含算法映射、实验指南、参数解释和结果解读。

---

## 🔬 实验设置

原始需求来自 [Coding_prompt.tex](Coding_prompt.tex)。当前实现如下：

| 项目 | 设置 |
| --- | --- |
| 平台 | PyTorch |
| 数据集 | MNIST |
| 网络结构 | `784 -> 500 -> 500 -> 500 -> 10` |
| 隐藏层数量 | 3 |
| 每个隐藏层神经元 | 500 |
| 激活函数 | `softplus(x) + eps`，保证输出为正 |
| 分类损失 | `CrossEntropyLoss(log(y_L), target)` |
| 对比指标 | 每层权重变化向量夹角，以及所有层的平均夹角 |
| 默认训练规则 | `biological` |
| 默认截断函数 | `zeta(x) = 0.5 * tanh(x)` |
| 默认更新函数 | `f(epsilon_l, y_l) = epsilon_l * y_l` |

核心公式：

```text
epsilon_l = zeta(y_l^-1 * sigma'(x_l) * W_(l+1)^T (y_(l+1) * epsilon_(l+1)))
Delta W_l = -eta * f(epsilon_l, y_l) * y_(l-1)^T
```

---

## 🚀 快速开始

进入项目目录：

```bash
cd ~/IPARA/3-RESOURCES/emacs/config/github/Triplet_STDP_CV2_Learning
```

运行快速检查：

```bash
./scripts/run_experiment.sh --smoke
```

这个命令会自动完成：

1. 创建或复用 `.venv`
2. 安装 `requirements.txt` 中的依赖
3. 运行单元测试
4. 用 64 个 MNIST 样本跑 1 个 epoch
5. 生成 CSV 和趋势图

输出文件：

```text
outputs/smoke_metrics.csv
outputs/smoke_angle_trend.png
```

---

## 🏁 正式实验

使用默认网络结构运行 20 个 epoch：

```bash
./scripts/run_experiment.sh --epochs 20 --open
```

不自动打开图片：

```bash
./scripts/run_experiment.sh --epochs 20
```

输出文件：

| 文件 | 含义 |
| --- | --- |
| `outputs/angle_metrics.csv` | 每个 epoch 的 loss、accuracy、平均夹角、各层夹角 |
| `outputs/angle_trend.png` | 平均夹角和各层夹角随 epoch 变化的曲线 |

---

## 🧭 推荐实验组合

### 1. 生物规则训练轨迹

```bash
./scripts/run_experiment.sh \
  --epochs 20 \
  --train-rule biological \
  --output-csv outputs/biological_metrics.csv \
  --output-plot outputs/biological_angle_trend.png
```

观察网络由生物规则训练时，生物规则更新方向与 BP 更新方向的夹角如何变化。

### 2. BP 训练轨迹

```bash
./scripts/run_experiment.sh \
  --epochs 20 \
  --train-rule bp \
  --output-csv outputs/bp_metrics.csv \
  --output-plot outputs/bp_angle_trend.png
```

观察网络由标准 BP 训练时，生物规则是否仍能给出接近 BP 的更新方向。

### 3. 初始状态方向比较

```bash
./scripts/run_experiment.sh \
  --epochs 1 \
  --train-rule none \
  --output-csv outputs/no_update_metrics.csv \
  --output-plot outputs/no_update_angle_trend.png
```

只比较随机初始化附近的方向关系，不更新参数。

---

## ⚙️ 脚本参数

查看完整帮助：

```bash
./scripts/run_experiment.sh --help
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--smoke` | 关闭 | 快速检查，1 epoch、64 个样本 |
| `--quick` | 关闭 | 中等检查，2 epoch、1024 个样本 |
| `--setup-only` | 关闭 | 只创建环境并安装依赖 |
| `--no-tests` | 关闭 | 跳过 pytest |
| `--open` | 关闭 | 实验结束后打开趋势图 |
| `--epochs N` | `20` | 训练 epoch 数 |
| `--batch-size N` | `128` | batch size |
| `--learning-rate X` | `0.01` | 学习率 |
| `--hidden-layers N` | `3` | 隐藏层数量 |
| `--hidden-dim N` | `500` | 每个隐藏层神经元数量 |
| `--train-rule RULE` | `biological` | `biological`、`bp` 或 `none` |
| `--scale-mode MODE` | `tanh` | `tanh` 或 `clamp` |
| `--update-function NAME` | `epsilon_times_activation` | 生物规则中的 `f(epsilon_l, y_l)` |
| `--update-bias` | 关闭 | 同步更新 bias |
| `--limit-samples N` | 空 | 只使用前 N 个 MNIST 样本 |
| `--device DEVICE` | 自动 | `cpu`、`cuda` 或 `mps` |

---

## 🛠️ 手动运行

如果不使用一键脚本，也可以手动创建环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

手动运行实验：

```bash
python -m triplet_stdp_cv2_learning.train_mnist \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 0.01 \
  --train-rule biological \
  --output-csv outputs/angle_metrics.csv \
  --output-plot outputs/angle_trend.png
```

运行测试：

```bash
python -m pytest tests/test_learning_rules.py -q
```

---

## 🗂️ 项目结构

```text
Triplet_STDP_CV2_Learning/
  Coding_prompt.tex
  README.md
  requirements.txt
  scripts/
    run_experiment.sh
  triplet_stdp_cv2_learning/
    __init__.py
    model.py
    learning_rules.py
    train_mnist.py
  tests/
    test_learning_rules.py
  docs/
    algorithm_details.md
    experiment_guide.md
```

| 路径 | 作用 |
| --- | --- |
| `triplet_stdp_cv2_learning/model.py` | 正值激活 MLP 与 forward cache |
| `triplet_stdp_cv2_learning/learning_rules.py` | BP 更新、生物误差传播、生物更新、夹角统计 |
| `triplet_stdp_cv2_learning/train_mnist.py` | MNIST 实验入口，保存 CSV 和趋势图 |
| `scripts/run_experiment.sh` | 一键环境准备、测试、训练脚本 |
| `tests/test_learning_rules.py` | 核心数学规则的单元测试 |
| `docs/algorithm_details.md` | 公式到代码的详细对应关系 |
| `docs/experiment_guide.md` | 实验设计、结果解释、扩展建议 |

---

## 📊 结果解读

CSV 中最重要的列：

| 列名 | 含义 |
| --- | --- |
| `epoch` | 当前 epoch |
| `train_loss` | 当前 epoch 平均训练损失 |
| `train_accuracy` | 当前 epoch 平均训练准确率 |
| `mean_angle_degrees` | 所有层权重更新夹角的平均值 |
| `layer_1_angle_degrees` 到 `layer_4_angle_degrees` | 每一层的夹角 |

重点观察：

- `mean_angle_degrees` 是否随 epoch 下降。
- 不同层的夹角是否同步下降。
- 输出层和底层输入层的夹角是否明显不同。
- `--train-rule biological` 与 `--train-rule bp` 两条参数轨迹上的趋势是否一致。

---

## 🧾 文档导航

- 📐 [算法实现说明](docs/algorithm_details.md)
- 🧪 [实验指南](docs/experiment_guide.md)
- 📄 [原始需求](Coding_prompt.tex)

---

## ✅ 当前验证

最近一次 smoke run 验证链路：

```bash
./scripts/run_experiment.sh --smoke
```

已验证内容：

- 依赖版本可导入
- 单元测试通过
- 默认 3 层、每层 500 神经元网络可以完成小样本训练
- CSV 和趋势图可以正常生成
